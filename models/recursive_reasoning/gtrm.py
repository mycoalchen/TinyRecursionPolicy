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
from einops import rearrange

IGNORE_LABEL_ID = -100

@dataclass
class GaussianTRM_ACTV1InnerCarry:
    # z_H is Optional to support num_latents=1 mode
    z_H: Optional[torch.Tensor]
    z_L: torch.Tensor

@dataclass
class GaussianTRM_ACTV1Carry:
    inner_carry: GaussianTRM_ACTV1InnerCarry
    
    steps: torch.Tensor
    halted: torch.Tensor
    
    current_data: Dict[str, torch.Tensor]


class GaussianTRM_ACTV1Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
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

    # --- NEW CONFIG SWITCHES ---
    num_latents: int = 2 # 1 = Flat Recurrence, 2 = Hierarchical (Manager/Worker)
    
    # Options: "latent", "add", "concat"
    # Controls input to the Gaussian heads (mu/sigma)
    stochastic_head_input: str = "latent" 


class GaussianTRM_ACTV1Block(nn.Module):
    def __init__(self, config: GaussianTRM_ACTV1Config) -> None:
        super().__init__()

        self.config = config
        if self.config.mlp_t:
            self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
            self.mlp_t = SwiGLU(
                hidden_size=self.config.seq_len + self.puzzle_emb_len, # L
                expansion=config.expansion,
            )
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

class GaussianTRM_ACTV1ReasoningModule(nn.Module):
    def __init__(self, config: GaussianTRM_ACTV1Config, layers: List[GaussianTRM_ACTV1Block]):
        super().__init__()
        self.config = config
        self.layers = torch.nn.ModuleList(layers)
        
        # Calculate input dimension for Gaussian heads based on config
        self.head_input_dim = self.config.hidden_size
        if self.config.stochastic_head_input == "concat":
            self.head_input_dim = self.config.hidden_size * 2
            
        self.mu_head = CastedLinear(
            self.head_input_dim,
            self.config.hidden_size,
            bias=True,
        )
        # sigma in log-space to guarantee positive sigma
        self.log_sigma_head = CastedLinear(
            self.head_input_dim,
            self.config.hidden_size,
            bias=True,
        )
        # might help early stability to start with small sigma
        with torch.no_grad():
            self.log_sigma_head.bias.fill_(self.config.log_sigma_head_init_bias)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        # Injection logic for the recurrence path (always add)
        hidden_states = hidden_states + input_injection
        
        # Run Transformer/MLP Layers
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
            
        # Prepare inputs for Mu/Sigma heads based on config
        if self.config.stochastic_head_input == "latent":
            head_input = hidden_states
        elif self.config.stochastic_head_input == "add":
            head_input = hidden_states + input_injection
        elif self.config.stochastic_head_input == "concat":
            head_input = torch.cat([hidden_states, input_injection], dim=-1)
        else:
            raise ValueError(f"Unknown stochastic_head_input: {self.config.stochastic_head_input}")

        mu = self.mu_head(head_input)
        log_sigma = self.log_sigma_head(head_input)
        
        # Re-parameterization trick
        return mu + torch.exp(log_sigma) * torch.randn_like(log_sigma)


class GaussianTRM_ACTV1_Inner(nn.Module):
    def __init__(self, config: GaussianTRM_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O

        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head      = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head       = CastedLinear(self.config.hidden_size, 2, bias=True)

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
        self.L_level = GaussianTRM_ACTV1ReasoningModule(config=self.config, layers=[GaussianTRM_ACTV1Block(self.config) for _i in range(self.config.L_layers)])

        # Initial states
        
        # Always init L (Worker / Main State)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        
        # Conditionally init H (Manager State)
        if self.config.num_latents == 2:
            self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        else:
            # Placeholder to satisfy load_state_dict if architectures switch, 
            # though usually strictly separated by config.
            self.register_buffer("H_init", None)

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32))

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)

        # Position embeddings
        if self.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Scale
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        z_H = None
        if self.config.num_latents == 2:
            z_H = torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype)
            
        return GaussianTRM_ACTV1InnerCarry(
            z_H=z_H,
            z_L=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: GaussianTRM_ACTV1InnerCarry):
        z_H = None
        if self.config.num_latents == 2:
            z_H = torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H)
            
        return GaussianTRM_ACTV1InnerCarry(
            z_H=z_H,
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(self, carry: GaussianTRM_ACTV1InnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[GaussianTRM_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # Forward iterations
        z_H, z_L = carry.z_H, carry.z_L
        
        # --- BRANCHING LOGIC FOR 1 VS 2 LATENTS ---
        
        if self.config.num_latents == 2:
            # === HIERARCHICAL RECURRENCE (Original) ===
            # H_cycles-1 without grad
            with torch.no_grad():
                for _H_step in range(self.config.H_cycles-1):
                    for _L_step in range(self.config.L_cycles):
                        # L gets injected with Manager(H) + Input
                        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                    # H gets injected with Worker(L)
                    z_H = self.L_level(z_H, z_L, **seq_info)
            # 1 with grad
            for _L_step in range(self.config.L_cycles):
                z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
            z_H = self.L_level(z_H, z_L, **seq_info)
            
            readout_state = z_H

        else:
            # === FLAT RECURRENCE ===
            # We treat z_L as the single state 'z'. 
            # z_H is unused/None.
            total_steps = self.config.H_cycles * self.config.L_cycles
            
            with torch.no_grad():
                for _ in range(total_steps - 1):
                    # z_L gets injected with Input only
                    z_L = self.L_level(z_L, input_embeddings, **seq_info)
            
            # Final step with grad
            z_L = self.L_level(z_L, input_embeddings, **seq_info)
            
            readout_state = z_L
            z_H = None # ensure it stays None

        # LM Outputs
        new_carry = GaussianTRM_ACTV1InnerCarry(z_H=z_H.detach() if z_H is not None else None, z_L=z_L.detach())  # New carry no grad
        
        output = self.lm_head(readout_state)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(readout_state[:, 0]).to(torch.float32) # Q-head
        
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class GaussianTRM_ACTV1(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = GaussianTRM_ACTV1Config(**config_dict)
        self.inner = GaussianTRM_ACTV1_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]

        return GaussianTRM_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),  # Empty is expected, it will be reseted in first pass as all sequences are halted.
            
            steps=torch.zeros((batch_size, ), dtype=torch.int32),
            halted=torch.ones((batch_size, ), dtype=torch.bool),  # Default to halted
            
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
        
    def forward(self, carry: GaussianTRM_ACTV1Carry, batch: Dict[str, torch.Tensor]) -> Tuple[GaussianTRM_ACTV1Carry, Dict[str, torch.Tensor]]:

        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        
        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }

        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            
            halted = is_last_step

            # if training, and ACT is enabled
            if self.training and (self.config.halt_max_steps > 1):

                # Halt signal
                # NOTE: During evaluation, always use max steps, this is to guarantee the same halting steps inside a batch for batching purposes
                
                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                halted = halted & (new_steps >= min_halt_steps)

                if not self.config.no_ACT_continue:
                    # Compute target Q
                    # NOTE: No replay buffer and target networks for computing target Q-value.
                    # As batch_size is large, there're many parallel envs.
                    # Similar concept as PQN https://arxiv.org/abs/2407.04811
                    _, _, (next_q_halt_logits, next_q_continue_logits), _, _ = self.inner(new_inner_carry, new_current_data)
                    outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

        return GaussianTRM_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs