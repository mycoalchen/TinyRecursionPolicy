from typing import Iterable, Tuple, List, Dict, Optional
from dataclasses import dataclass
from jaxtyping import Array, Bool, Float, Int
import torch
import torch.nn as nn
from pydantic import BaseModel
from abc import ABC, abstractmethod
from einops import rearrange
import math

from models.layers import SwiGLU, rms_norm, CastedLinear, CastedEmbedding


@dataclass
class TRMCarry:
    z_H: Float[Array, "B L D"]
    z_L: Float[Array, "B L D"]
    steps: Int[Array, "B"]
    halted: Bool[Array, "B"]
    current_inputs: Int[Array, "B L"]
    current_labels: Int[Array, "B L"]
    current_puzzle_ids: Int[Array, "B"]


class TRMConfig(BaseModel):
    batch_size: int
    seq_len: int
    vocab_size: int

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    forward_dtype: str = "bfloat16"

    H_cycles: int
    L_cycles: int
    layers: int

    halt_max_steps: int
    halt_exploration_prob: float

    puzzle_emb_len: int = 16
    expansion: float = 4

    log_sigma_head_init_bias: float = -5.0

    q_head_input_detached: bool
    q_head_input_form: str
    H_deterministic_mode: str


class TRMBlock(ABC, nn.Module):
    def __init__(self, config: TRMConfig) -> None:
        super().__init__()
        self.config = config
        self.seq_mlp = SwiGLU(
            hidden_size=config.seq_len + config.puzzle_emb_len,
            expansion=config.expansion
        )
        self.hidden_dim_mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )

    def forward(self, hidden_states: Float[Array, "B L D"]) -> Float[Array, "B L D"]:
        hidden_states = rearrange(hidden_states, "B L D -> B D L")
        out = self.seq_mlp(hidden_states)
        hidden_states = rms_norm(
            hidden_states + out, variance_epsilon=self.config.norm_eps)
        hidden_states = rearrange(hidden_states, "B D L -> B L D")
        out = self.hidden_dim_mlp(hidden_states)
        hidden_states = rms_norm(
            hidden_states + out, variance_epsilon=self.norm_eps)
        return hidden_states


class TRMReasoningModule(ABC, nn.Module):
    def __init__(self, config: TRMConfig, layers: list[TRMBlock]) -> None:
        super().__init__()
        self.config = config
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: Float[Array, "B L D"], input_injection: Float[Array, "B L D"], **kwargs) -> Float[Array, "B L D"]:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class Base_TRM_Inner(ABC, nn.Module):
    @abstractmethod
    def __init__(self, config: TRMConfig) -> None:
        self.config = config
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale
        self.embed_tokens = CastedEmbedding(
            self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head = CastedLinear(
            self.config.hidden_size, self.config.vocab_size, bias=False)
        if config.q_head_input_form == "first puzzle emb":
            self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)
        elif config.q_head_input_form == "intermediate output":
            self.q_head = CastedLinear(
                self.config.seq_len * self.config.vocab_size, 2, bias=True)
        else:
            raise ValueError(
                f"Unknown q_head_input_form: {config.q_head_input_form}. Must be 'first puzzle emb' or 'intermediate output'")

    @abstractmethod
    def _input_embeddings(self, input: Float[Array, "B seq_len"], puzzle_ids: Int[Array, "B"]):
        ...

    @abstractmethod
    def forward():
        ...