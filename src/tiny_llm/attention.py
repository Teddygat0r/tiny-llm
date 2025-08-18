import mlx.core as mx
from .basics import softmax, linear
import math


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    scale_factor = 1 / math.sqrt(query.shape[-1]) if scale is None else scale

    axes = list(range(key.ndim))
    axes[-2], axes[-1] = axes[-1], axes[-2]
    weights = mx.matmul(query, key.transpose(axes)) * scale_factor

    if mask is not None:
        weights = mx.add(weights, mask)
    
    softmax_weights = softmax(weights, axis=-1)
    return mx.matmul(softmax_weights, value)



class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        # if mask is not None:
        #     mask = mx.expand_dims(mask, axis=0)
        n_dim = query.shape[0]
        
        query_expanded = linear(query, self.wq).reshape((n_dim, -1, self.num_heads, self.hidden_size // self.num_heads))
        key_expanded = linear(key, self.wk).reshape((n_dim, -1, self.num_heads, self.hidden_size // self.num_heads))
        value_expanded = linear(value, self.wv).reshape((n_dim, -1, self.num_heads, self.hidden_size // self.num_heads))

        del query
        del key
        del value

        axes = list(range(key_expanded.ndim))
        axes[-3], axes[-2] = axes[-2], axes[-3]

        query_expanded = query_expanded.transpose(axes)
        key_expanded = key_expanded.transpose(axes)
        value_expanded = value_expanded.transpose(axes)

        p_tokens = scaled_dot_product_attention_simple(query_expanded, key_expanded, value_expanded, None, mask)

        del query_expanded
        del key_expanded
        del value_expanded

        p_tokens = p_tokens.transpose(axes)
        return linear(p_tokens.reshape(n_dim, -1, self.hidden_size), self.wo)
        


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    mask = mx.tril(mx.ones((L, S)), k=(S - L))
    mask = mx.where(mask, mx.array(0), mx.array(-mx.inf)).astype(dtype)
    return mask


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    H_q, L, D = query.shape[-3:]
    H, S, D = key.shape[-3:]

    B = query.shape[:-3]

    n_repeats = H_q // H

    query = query.reshape(*query.shape[:-3], -1, H, n_repeats, L, D)
    key = key.reshape(*key.shape[:-3], -1, H, 1, S, D)
    value = value.reshape(*value.shape[:-3], -1, H, 1, S, D)
    
    scale = 1 / math.sqrt(query.shape[-1]) if scale is None else scale
    axes = list(range(key.ndim))
    axes[-2], axes[-1] = axes[-1], axes[-2]

    weights = mx.matmul(query, key.transpose(axes)) * scale
    if mask == "causal":
        mask = causal_mask(L, S, dtype=weights.dtype)
        weights += mask
    elif mask is not None:
        mask = mx.broadcast_to(mask, (*B, H_q, L, S))
        mask = mask.reshape(*B, 1, H, n_repeats, L, S)
        weights = mx.add(weights, mask)
    
    softmax_weights = softmax(weights, axis=-1)
    f = mx.matmul(softmax_weights, value)
    return f.reshape(*B, H_q, L, D)



def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
) -> mx.array:
    pass
