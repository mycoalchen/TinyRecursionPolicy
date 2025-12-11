from typing import Iterable, Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
from models.common import trunc_normal_init_
from models.layers import rms_norm, LinearSwish, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding
from einops import rearrange, repeat

IGNORE_LABEL_ID = -100

# State passed between inner loops

@dataclass
class GTRMLatents:
    z_H: torch.Tensor
    z_L: torch.Tensor


# State passed betewen outer loops
@dataclass
class GTRMState:
    latents: GTRMLatents

    steps: torch.Tensor
    halted: torch.Tensor

    current_data: Dict[str, torch.Tensor]


class GTRMConfig(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int # 1
    vocab_size: int

    H_cycles: int
    L_cycles: int

    H_layers: int # ignored
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    
    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"

    # Alexia: added
    mlp_t: bool = False # use mlp on L instead of transformer
    puzzle_emb_len: int = 16 # if non-zero, its specified to this value
    no_ACT_continue: bool =  True # No continue ACT loss, only use the sigmoid of the halt which makes much more sense
    log_sigma_head_init_bias: float = -5.0 # Initial bias for log_sigma head (negative = small sigma initially)

    # Added to GTRM
    q_head_input_detached: bool
    q_head_input_form: str # "intermediate output" for un-embedded z_H or "first puzzle emb" for z_H[:, 0, :] (original)
    H_deterministic_mode: str # "separate weights" or "skip noise" or "False"

    force_max_steps_at_eval: bool # If True, model always runs full number of inference steps during eval
    time_embeddings: bool

class GTRMBlock(nn.Module):
    """
    (B, L, D) -> B, L, D
    """
    def __init__(self, config: GTRMConfig) -> None:
        super().__init__()

        self.config = config
        if self.config.mlp_t:
            self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
            self.mlp_t = SwiGLU( # "MLP in sequence's time dimension"
                hidden_size=self.config.seq_len + self.puzzle_emb_len, # L
                expansion=config.expansion,
            ) # L -> L
        else:
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=False
            )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # Post Norm
        if self.config.mlp_t:
            hidden_states = rearrange(hidden_states, "B L D -> B D L")
            out = self.mlp_t(hidden_states)
            hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
            hidden_states = rearrange(hidden_states, "B D L -> B L D")
        else:
            # Self Attention
            hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        # Fully Connected
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
        return hidden_states

class GTRMReasoningModule(nn.Module):
    """
    (B, L, D) -> (B, L, D)
    """
    def __init__(self, config: GTRMConfig, layers: List[GTRMBlock], gaussian: bool):
        super().__init__()
        self.config = config
        self.layers = torch.nn.ModuleList(layers)
        self.gaussian = gaussian
        if gaussian:
            self.mu_head = CastedLinear(
                self.config.hidden_size,
                self.config.hidden_size,
                bias=True,
            )
            # sigma in log-space to guarantee positive sigma
            self.log_sigma_head = CastedLinear(
                self.config.hidden_size,
                self.config.hidden_size,
                bias=True,
            )
            # might help early stability to start with small sigma
            with torch.no_grad():
                self.log_sigma_head.bias.fill_(self.config.log_sigma_head_init_bias)

        if config.time_embeddings:
            embed_init_std = 1.0 / math.sqrt(self.config.hidden_size)
            self.inner_step_emb = CastedEmbedding(
                num_embeddings = self.config.L_cycles + 1,
                embedding_dim = self.config.hidden_size,
                init_std = embed_init_std,
                cast_to = getattr(torch, self.config.forward_dtype)
            )
            self.act_step_emb = CastedEmbedding(
                num_embeddings = self.config.halt_max_steps,
                embedding_dim = self.config.hidden_size,
                init_std = embed_init_std,
                cast_to = getattr(torch, self.config.forward_dtype)
            )

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, inner_step_num: torch.Tensor, act_step_num: torch.Tensor, skip_noise: bool = False, **kwargs) -> torch.Tensor:
        """
        skip_noise lets us reuse mu for both z_L and z_H but only add mu * eps to z_L
        """
        hidden_states = hidden_states + input_injection
        if self.config.time_embeddings:
            hidden_states = hidden_states + self.inner_step_emb(inner_step_num)[:, None, :]
            hidden_states = hidden_states + self.act_step_emb(act_step_num)[:, None, :]
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        if self.gaussian:
            mu = self.mu_head(hidden_states)
            if not skip_noise:
                log_sigma = self.log_sigma_head(hidden_states)
                hidden_states = mu + torch.exp(log_sigma) * torch.randn_like(log_sigma)
            else:
                hidden_states = mu
        return hidden_states

class GTRM_Inner(nn.Module):
    def __init__(self, config: GTRMConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O

        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head      = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        if config.q_head_input_form == "first puzzle emb":
            self.q_head       = CastedLinear(self.config.hidden_size, 2, bias=True)
        elif config.q_head_input_form == "intermediate output":
            self.q_head = CastedLinear(self.config.seq_len * self.config.vocab_size, 2, bias=True)
        else:
            raise ValueError(f"Unknown q_head_input_form: {config.q_head_input_form}. Must be 'first puzzle emb' or 'intermediate output'")

        self.q_head_input_form = config.q_head_input_form
        self.q_head_input_detached = config.q_head_input_detached

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)  if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len  # ceil div
        if self.config.puzzle_emb_ndim > 0:
            # Zero init puzzle embeddings
            self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                    batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)

        # LM Blocks
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        else:
            pass

        # Reasoning Layers
        self.L_level = GTRMReasoningModule(config=self.config, layers=[GTRMBlock(self.config) for _i in range(self.config.L_layers)], gaussian=True)
        if self.config.H_deterministic_mode == "separate weights":
            self.H_level = GTRMReasoningModule(config=self.config, layers=[GTRMBlock(self.config) for _i in range(self.config.L_layers)], gaussian=False)

        # Initial states
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        """
        input: (B, seq_len)
        puzzle_identifiers: (B)
        """

        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32)) # (B,)

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers) # (B, D)
            #                emb_len        * D                       - D = (emb_len - 1) * D
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]

            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count)) # (B, (embed_len) * D) with 0's on the right

            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)

        # Position embeddings
        if self.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Scale
        return self.embed_scale * embedding

    def initial_latents(self, batch_size: int):
        return GTRMLatents(
            z_H = repeat(self.H_init, "D -> B L D", B=batch_size, L = self.config.seq_len + self.puzzle_emb_len),
            z_L = repeat(self.L_init, "D -> B L D", B=batch_size, L = self.config.seq_len + self.puzzle_emb_len)
        )

    def reset_halted_latents(self, reset_flag: torch.Tensor, latents: GTRMLatents):
        return GTRMLatents(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, latents.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, latents.z_L),
        )

    def iterate_H(self, z_H, z_L, inner_step_num, act_step_num, seq_info):
        """
        Map z_H + z_L -> z_H
        """
        if self.config.H_deterministic_mode == "skip noise":
            z_H = self.L_level(z_H, z_L, inner_step_num, act_step_num, skip_noise=True, **seq_info)
        elif self.config.H_deterministic_mode == "separate weights":
            z_H = self.H_level(z_H, z_L, inner_step_num, act_step_num, **seq_info)
        else:
            assert self.config.H_deterministic_mode == "False"
            z_H = self.L_level(z_H, z_L, inner_step_num, act_step_num, **seq_info)
        return z_H

    def forward(self, latents: GTRMLatents, batch: Dict[str, torch.Tensor], act_step_num: torch.Tensor) -> Tuple[GTRMLatents, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # Forward iterations
        z_H, z_L = latents.z_H, latents.z_L
        # H_cycles-1 without grad
        with torch.no_grad():
            for _H_step in range(self.config.H_cycles-1):
                for _L_step in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + input_embeddings,
                        torch.ones_like(act_step_num) * _L_step, act_step_num, **seq_info)
                z_H = self.iterate_H(z_H, z_L, 
                    torch.ones_like(act_step_num) * self.config.L_cycles, act_step_num, seq_info)
        # 1 with grad
        for _L_step in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings, 
                torch.ones_like(act_step_num) * _L_step, act_step_num, **seq_info)
        z_H = self.iterate_H(z_H, z_L, 
            torch.ones_like(act_step_num) * self.config.L_cycles, act_step_num, seq_info)

        # LM Outputs
        new_latents = GTRMLatents(z_H=z_H.detach(), z_L=z_L.detach())  # New latents no grad
        # z_H shape (B, puzzle_emb_len + seq_len, D)
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:] # (B, seq_len, vocab_size)
        if self.q_head_input_form == "intermediate output":
            q_head_input = rearrange(output, "B L V -> B (L V)")
        elif self.q_head_input_form == "first puzzle emb":
            q_head_input = z_H[:, 0, :] # shape (B, D)
        else:
            raise ValueError("Unknown q_head_input_form", self.q_head_input_form)
        if self.q_head_input_detached:
            q_head_input = q_head_input.detach()
        q_logits = self.q_head(q_head_input).to(torch.float32)
        return new_latents, output, (q_logits[..., 0], q_logits[..., 1])


class GTRM(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = GTRMConfig(**config_dict)
        self.inner = GTRM_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_state_train(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]
        return GTRMState(
            latents=self.inner.initial_latents(batch_size),
            steps=torch.zeros((batch_size, ), dtype=torch.int32),
            halted=torch.ones((batch_size, ), dtype=torch.bool),
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )

    def initial_state_eval(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]
        return GTRMState(
            latents=self.inner.initial_latents(batch_size),
            steps=torch.zeros((batch_size, ), dtype=torch.int32),
            halted=torch.zeros((batch_size, ), dtype=torch.bool),
            current_data=batch
        )

    def _forward_train(self, 
        state: GTRMState, 
        batch: Dict[str, torch.Tensor]
    )-> Tuple[GTRMState, Dict[str, torch.Tensor]]:

        # Prepare halted batch positions for new puzzles
        latents = self.inner.reset_halted_latents(state.halted, state.latents)
        steps = torch.where(state.halted, 0, state.steps)
        data = {
            k: torch.where(
                state.halted.view(
                    (-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v,
            )
            for k, v in state.current_data.items()
        }

        # Run the model to get new latents and outputs
        new_latents, logits, (q_halt_logits, _) = self.inner(latents, data, steps)
        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
        }

        # Handle step update logic with exploration
        with torch.no_grad():
            steps = steps + 1
            halted = (steps >= self.config.halt_max_steps) | (q_halt_logits > 0)
            min_halt_steps = (
                torch.rand_like(
                    q_halt_logits) < self.config.halt_exploration_prob
                    ) * torch.randint_like(steps, low=2, high=self.config.halt_max_steps + 1)
            halted = halted & (steps >= min_halt_steps)

        # Update model state
        new_latents = GTRMLatents(z_H=new_latents.z_H.detach(), z_L=new_latents.z_L.detach())
        return GTRMState(latents=new_latents, steps=steps, halted=halted, current_data=data), outputs
    
    def _forward_eval(self, 
        state: GTRMState, 
        batch: Dict[str, torch.Tensor],
        prev_outputs: Optional[Dict[str, torch.Tensor]]
    )-> Tuple[GTRMState, Dict[str, torch.Tensor]]:

        active = ~state.halted
        old_z_L, old_z_H = state.latents.z_L, state.latents.z_H

        # Run the model to get new latents and outputs
        new_latents, logits, (q_halt_logits, _) = self.inner(state.latents, batch, state.steps)
        if not prev_outputs:
            outputs = {
                "logits": logits,
                "q_halt_logits": q_halt_logits,
            }
        else:
            outputs = {
                "logits": torch.where(active[:, None, None], logits, prev_outputs["logits"]),
                "q_halt_logits": torch.where(active, q_halt_logits, prev_outputs["q_halt_logits"]),
            }

        # Handle step update logic
        steps = state.steps
        with torch.no_grad():
            steps = torch.where(active, steps + 1, steps)
            halted = state.halted | (steps >= self.config.halt_max_steps) | (q_halt_logits > 0)
        
        # Update model state
        new_latents = GTRMLatents(
            z_H=torch.where(active[:, None, None], new_latents.z_H, old_z_H).detach(), 
            z_L=torch.where(active[:, None, None], new_latents.z_L, old_z_L).detach())
        return GTRMState(latents=new_latents, steps=steps, halted=halted, current_data=batch), outputs


    def forward(self, 
        state: GTRMState, 
        batch: Dict[str, torch.Tensor],
        prev_outputs: Optional[Dict[str, torch.Tensor]] = None,
    )-> Tuple[GTRMState, Dict[str, torch.Tensor]]:
        if self.training:
            return self._forward_train(state, batch)
        else:
            return self._forward_eval(state, batch, prev_outputs)
