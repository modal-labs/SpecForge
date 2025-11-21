from flash_attn import flash_attn_func
try:
    from flash_attn.cute.interface import _flash_attn_fwd as flash_attn_func_v4
except ImportError as e:
    print(f"[Warning] could not import flash_attn_func_v4: {e}")
    flash_attn_func_v4 = None
from torch import Tensor


def fa2(q: Tensor, k: Tensor, v: Tensor, softmax_scale: float, causal: bool) -> tuple[Tensor, Tensor]:
    attn_output, lse, _ = flash_attn_func(
        q, k, v, softmax_scale=softmax_scale, causal=causal, dropout_p=0.0, return_attn_probs=True
    )
    return attn_output, lse


def fa4(q: Tensor, k: Tensor, v: Tensor, softmax_scale: float, causal: bool) -> tuple[Tensor, Tensor]:
    return flash_attn_func_v4(q, k, v, softmax_scale=softmax_scale, causal=causal, return_lse=True)


def resolve_flash_attn_func(backend: str):
    if backend == "fa2":
        return fa2
    elif backend == "fa4":
        return fa4
    else:
        raise ValueError(f"Unknown flash attention backend: {backend}")
