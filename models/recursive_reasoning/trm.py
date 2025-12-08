from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
from einops import repeat
import torch
import copy
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
import random
from models.common import trunc_normal_init_
from models.layers import rms_norm, LinearSwish, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding

IGNORE_LABEL_ID = -100

# State passed between inner loops


@dataclass
class TRMLatents:
    z_H: torch.Tensor
    z_L: torch.Tensor


# State passed betewen outer loops
@dataclass
class TRMState:
    latents: TRMLatents

    steps: torch.Tensor
    halted: torch.Tensor

    current_data: Dict[str, torch.Tensor]


class TinyRecursiveReasoningModel_ACTV1Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int

    H_layers: int  # ignored
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
    mlp_t: bool = False  # use mlp on L instead of transformer
    puzzle_emb_len: int = 16  # if non-zero, its specified to this value
    # No continue ACT loss, only use the sigmoid of the halt which makes much more sense
    no_ACT_continue: bool = True

    force_max_steps_at_eval: bool
    time_embeddings: bool


class TinyRecursiveReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()

        self.config = config
        if self.config.mlp_t:
            self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -
                                    self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
            self.mlp_t = SwiGLU(
                hidden_size=self.config.seq_len + self.puzzle_emb_len,  # L
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
        # B, L, D = hidden_states.shape
        # Post Norm
        if self.config.mlp_t:
            hidden_states = hidden_states.transpose(1, 2)
            out = self.mlp_t(hidden_states)
            hidden_states = rms_norm(
                hidden_states + out, variance_epsilon=self.norm_eps)
            hidden_states = hidden_states.transpose(1, 2)
        else:
            # Self Attention
            hidden_states = rms_norm(hidden_states + self.self_attn(
                cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        # Fully Connected
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(
            hidden_states + out, variance_epsilon=self.norm_eps)
        return hidden_states

# Wrapper for multiple Blocks; represented by f in the paper


class TinyRecursiveReasoningModel_ACTV1ReasoningModule(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config, layers: List[TinyRecursiveReasoningModel_ACTV1Block]):
        super().__init__()
        self.config = config
        self.layers = torch.nn.ModuleList(layers)
        
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

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, inner_step_num: torch.Tensor, act_step_num: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        if self.config.time_embeddings:
            hidden_states = hidden_states + self.inner_step_emb(inner_step_num)[:, None, :]
            hidden_states = hidden_states + self.act_step_emb(act_step_num)[:, None, :]
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


# Stacks multiple ReasoningModules, each with multiple Blocks, and wraps with I/O layers
# forward() runs one outer loop (H_cycles - 1 inner loops without grad + 1 inner loop with grad)
class TinyRecursiveReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O

        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(
            self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head = CastedLinear(
            self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -
                                self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len  # ceil div
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
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len,
                                             self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        else:
            pass

        # Reasoning Layers
        self.L_level = TinyRecursiveReasoningModel_ACTV1ReasoningModule(
            config=self.config,
            layers=[TinyRecursiveReasoningModel_ACTV1Block(self.config) for _i in range(self.config.L_layers)])

        # Initial states
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(
            self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(
            self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

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

            pad_count = self.puzzle_emb_len * \
                self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat(
                (puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)

        # Position embeddings
        if self.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * \
                (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Scale
        return self.embed_scale * embedding

    def initial_latents(self, batch_size: int):
        return TRMLatents(
            z_H = repeat(self.H_init, "D -> B L D", B=batch_size, L = self.config.seq_len + self.puzzle_emb_len),
            z_L = repeat(self.L_init, "D -> B L D", B=batch_size, L = self.config.seq_len + self.puzzle_emb_len)
        )

    def reset_halted_latents(self, reset_flag: torch.Tensor, latents: TRMLatents):
        return TRMLatents(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, latents.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, latents.z_L),
        )

    def forward(self, latents: TRMLatents, batch: Dict[str, torch.Tensor], act_step_num: torch.Tensor) -> Tuple[TRMLatents, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(
            batch["inputs"], batch["puzzle_identifiers"])

        # Forward iterations
        it = 0
        z_H, z_L = latents.z_H, latents.z_L
        # H_cycles-1 without grad
        with torch.no_grad():
            for _H_step in range(self.config.H_cycles-1):
                for _L_step in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + input_embeddings,
                        torch.ones_like(act_step_num) * _L_step, act_step_num, **seq_info)
                z_H = self.L_level(z_H, z_L, 
                    torch.ones_like(act_step_num) * self.config.L_cycles, act_step_num, **seq_info)
        # 1 with grad
        for _L_step in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings, 
                torch.ones_like(act_step_num) * _L_step, act_step_num, **seq_info)
        z_H = self.L_level(z_H, z_L, 
            torch.ones_like(act_step_num) * self.config.L_cycles, act_step_num, **seq_info)

        # LM Outputs
        new_latents = TRMLatents(
            z_H=z_H.detach(), z_L=z_L.detach())  # New latents no grad
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        # Q-head; uses the first puzzle_emb position
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
        return new_latents, output, (q_logits[..., 0], q_logits[..., 1])


# Wraps Inner
class TinyRecursiveReasoningModel_ACTV1(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TinyRecursiveReasoningModel_ACTV1Config(**config_dict)
        self.inner = TinyRecursiveReasoningModel_ACTV1_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_state_train(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]
        return TRMState(
            latents=self.inner.initial_latents(batch_size),
            steps=torch.zeros((batch_size, ), dtype=torch.int32),
            halted=torch.ones((batch_size, ), dtype=torch.bool),
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )

    def initial_state_eval(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]
        return TRMState(
            latents=self.inner.initial_latents(batch_size),
            steps=torch.zeros((batch_size, ), dtype=torch.int32),
            halted=torch.zeros((batch_size, ), dtype=torch.bool),
            current_data=batch
        )

    def _forward_train(self, 
        state: TRMState, 
        batch: Dict[str, torch.Tensor]
    )-> Tuple[TRMState, Dict[str, torch.Tensor]]:

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
        new_latents = TRMLatents(z_H=new_latents.z_H.detach(), z_L=new_latents.z_L.detach())
        return TRMState(latents=new_latents, steps=steps, halted=halted, current_data=data), outputs
    
    def _forward_eval(self, 
        state: TRMState, 
        batch: Dict[str, torch.Tensor],
        prev_outputs: Optional[Dict[str, torch.Tensor]]
    )-> Tuple[TRMState, Dict[str, torch.Tensor]]:

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
        new_latents = TRMLatents(
            z_H=torch.where(active[:, None, None], new_latents.z_H, old_z_H).detach(), 
            z_L=torch.where(active[:, None, None], new_latents.z_L, old_z_L).detach())
        return TRMState(latents=new_latents, steps=steps, halted=halted, current_data=batch), outputs


    def forward(self, 
        state: TRMState, 
        batch: Dict[str, torch.Tensor],
        prev_outputs: Optional[Dict[str, torch.Tensor]] = None,
    )-> Tuple[TRMState, Dict[str, torch.Tensor]]:
        if self.training:
            return self._forward_train(state, batch)
        else:
            return self._forward_eval(state, batch, prev_outputs)

    # def forward(
    #     self,
    #     carry: TRMState,
    #     batch: Dict[str, torch.Tensor],
    # ) -> Tuple[TRMState, Dict[str, torch.Tensor]]:

    #     if self.training:
    #         new_latents = self.inner.reset_halted_latents(
    #             carry.halted, carry.latents)
    #         new_steps = torch.where(carry.halted, 0, carry.steps)
    #         new_current_data = {
    #             k: torch.where(
    #                 carry.halted.view(
    #                     (-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v,
    #             )
    #             for k, v in carry.current_data.items()
    #         }
    #     else:
    #         new_latents = carry.latents
    #         new_steps = carry.steps
    #         new_current_data = batch

    #     new_latents_step, logits, (q_halt_logits, q_continue_logits) = self.inner(
    #         new_latents, new_current_data
    #     )

    #     outputs = {
    #         "logits": logits,
    #         "q_halt_logits": q_halt_logits,
    #         "q_continue_logits": q_continue_logits,
    #     }

    #     with torch.no_grad():
    #         if self.training:
    #             new_steps = new_steps + 1
    #             is_last_step = new_steps >= self.config.halt_max_steps

    #             if self.config.halt_max_steps > 1:
    #                 if self.config.no_ACT_continue:
    #                     halted = is_last_step | (q_halt_logits > 0)
    #                 else:
    #                     halted = is_last_step | (
    #                         q_halt_logits > q_continue_logits)
    #                 min_halt_steps = (
    #                     torch.rand_like(
    #                         q_halt_logits) < self.config.halt_exploration_prob
    #                 ) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
    #                 halted = halted & (new_steps >= min_halt_steps)

    #                 if not self.config.no_ACT_continue:
    #                     # Target Q for continue
    #                     _, _, (next_q_halt_logits, next_q_continue_logits) = self.inner(
    #                         new_latents_step, new_current_data
    #                     )
    #                     outputs["target_q_continue"] = torch.sigmoid(
    #                         torch.where(
    #                             is_last_step,
    #                             next_q_halt_logits,
    #                             torch.maximum(next_q_halt_logits,
    #                                           next_q_continue_logits),
    #                         )
    #                     )

    #         else:
    #             # Only increment steps for sequences that were not already halted
    #             new_steps = new_steps + (~carry.halted).to(new_steps.dtype)
    #             is_last_step = new_steps >= self.config.halt_max_steps

    #             if self.config.halt_max_steps > 1 and not self.config.force_max_steps_at_eval:
    #                 if self.config.no_ACT_continue:
    #                     halted = is_last_step | (q_halt_logits > 0)
    #                 else:
    #                     halted = is_last_step | (
    #                         q_halt_logits > q_continue_logits
    #                     )

    #             halted = carry.halted | halted

    #     if self.training:
    #         final_latents = TRMLatents(
    #             z_H=new_latents_step.z_H.detach(),
    #             z_L=new_latents_step.z_L.detach(),
    #         )
    #     else:
    #         # Eval:
    #         # - For sequences that were already halted at the previous step,
    #         #   keep their old z_H/z_L.
    #         # - For sequences that just halted now or are still active,
    #         #   use the newly computed z_H/z_L from this step.
    #         # uses *previous* halted
    #         keep_old_mask = carry.halted.view(-1, 1, 1)
    #         z_H = torch.where(keep_old_mask, carry.latents.z_H,
    #                           new_latents_step.z_H).detach()
    #         z_L = torch.where(keep_old_mask, carry.latents.z_L,
    #                           new_latents_step.z_L).detach()
    #         final_latents = TRMLatents(
    #             z_H=z_H, z_L=z_L
    #         )

    #     new_carry = TRMState(
    #         latents=final_latents,
    #         steps=new_steps,
    #         halted=halted,
    #         current_data=new_current_data,
    #     )

    #     return new_carry, outputs
