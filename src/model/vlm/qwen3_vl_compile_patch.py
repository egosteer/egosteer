"""
Monkey-patch Qwen3VLVisionAttention.forward for torch.compile compatibility.

Problem:
  torch.compile on Qwen3 VL vision blocks fails because the original forward
  goes through transformers' attention_interface, which:
    1. Triggers a graph break via lazy_import_flash_attention.
    2. Passes max_seqlen as a 0-dim tensor; the flash-attention C++ op
       expects int / SymInt, not FakeTensor.

Fix:
  Bypass attention_interface entirely and call flash_attn_varlen_func directly,
  converting max_seqlen to a Python scalar via .item().

Requires env var: TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS=1
  (so .item() produces SymInt during dynamo tracing instead of a graph break)

References:
  - https://github.com/Dao-AILab/flash-attention/issues/1351
  - https://github.com/huggingface/transformers/pull/37206
"""

import torch
from flash_attn import flash_attn_varlen_func
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLVisionAttention,
    apply_rotary_pos_emb_vision,
)


def _patched_vision_attn_forward(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: torch.Tensor | None = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    **kwargs,
) -> torch.Tensor:
    seq_length = hidden_states.shape[0]

    # QKV projection: (seq_len, hidden) -> (seq_len, 3, heads, head_dim) -> unbind
    query_states, key_states, value_states = (
        self.qkv(hidden_states)
        .reshape(seq_length, 3, self.num_heads, -1)
        .permute(1, 0, 2, 3)
        .unbind(0)
    )
    # Each: (seq_len, heads, head_dim) — exactly what flash_attn_varlen_func expects

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb_vision(
        query_states, key_states, cos, sin
    )

    # .item() converts 0-dim tensor -> Python int (or SymInt under dynamo)
    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

    attn_output = flash_attn_varlen_func(
        query_states.contiguous(),
        key_states.contiguous(),
        value_states.contiguous(),
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        dropout_p=0.0 if not self.training else self.attention_dropout,
        softmax_scale=self.scaling,
        causal=False,
    )

    attn_output = attn_output.reshape(seq_length, -1)
    attn_output = self.proj(attn_output)
    return attn_output


def apply_patch():
    Qwen3VLVisionAttention.forward = _patched_vision_attn_forward
