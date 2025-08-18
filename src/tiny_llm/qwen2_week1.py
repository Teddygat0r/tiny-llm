import mlx.core as mx

from tiny_llm_ref.attention import scaled_dot_product_attention_grouped
from tiny_llm_ref.basics import linear, silu
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear


class Qwen2MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.bq = bq
        self.bk = bk
        self.bv = bv
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.head_dim = hidden_size // num_heads
        self.scale = mx.rsqrt(self.head_dim)

        self.rope = RoPE(self.head_dim, max_seq_len, theta)

    def __call__(
        self,
        x: mx.array,
        offset: int,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        B, L, E = x.shape
        q = linear(x, self.wq, bias=self.bq).reshape(B, L, self.num_heads, self.head_dim)
        k = linear(x, self.wk, bias=self.bk).reshape(B, L, self.num_kv_heads, self.head_dim)
        v = linear(x, self.wv, bias=self.bv).reshape(B, L, self.num_kv_heads, self.head_dim)

        q = self.rope(q, offset=slice(0, L))
        k = self.rope(k, offset=slice(0, L))

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        x = scaled_dot_product_attention_grouped(q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32), scale=self.scale, mask=mask).astype(x.dtype)
        x = x.transpose(0, 2, 1, 3).reshape(B, L, self.hidden_size)
        return linear(x, self.wo)


class Qwen2MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

    def __call__(self, x: mx.array) -> mx.array:
        x_gate = linear(x, self.w_gate)
        up_projection = linear(x, self.w_up)
        silu_result = silu(x_gate) * up_projection

        return linear(silu_result, self.w_down)


class Qwen2TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.num_kv_heads = num_kv_heads
        self.intermediate_size = intermediate_size

        self.input_layernorm = RMSNorm(hidden_size, w_input_layernorm, rms_norm_eps)
        self.attention = Qwen2MultiHeadAttention(hidden_size, num_attention_heads, num_kv_heads, wq, wk, wv, wo, bq, bk, bv, max_seq_len, theta)
        self.post_attention_layernorm = RMSNorm(hidden_size, w_post_attention_layernorm, rms_norm_eps)
        self.mlp = Qwen2MLP(hidden_size, intermediate_size, w_gate, w_up, w_down)

    def __call__(
        self,
        x: mx.array,
        offset: int,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        residual = self.attention(self.input_layernorm(x), offset, mask)
        x += residual
        
        residual = self.mlp(self.post_attention_layernorm(x))
        return x + residual


class Qwen2ModelWeek1:
    def __init__(self, mlx_model: Any):
        self.num_hidden_layers = mlx_model.args.num_hidden_layers
        self.hidden_size = mlx_model.args.hidden_size
        self.vocab_size = mlx_model.args.vocab_size

        self.precision = mx.float16

        self.embeddings = Embedding(self.vocab_size, self.hidden_size, dequantize_linear(mlx_model.model.embed_tokens).astype(self.precision))
        self.inner_layers: list[Qwen2TransformerBlock] = []

        for i in range(self.num_hidden_layers):
            num_attention_heads = mlx_model.args.num_attention_heads
            num_kv_heads = mlx_model.args.num_key_value_heads
            hidden_size = self.hidden_size
            intermediate_size = mlx_model.args.intermediate_size
            rms_norm_eps = mlx_model.args.rms_norm_eps
            wq = dequantize_linear(mlx_model.model.layers[i].self_attn.q_proj).astype(self.precision)
            wk = dequantize_linear(mlx_model.model.layers[i].self_attn.k_proj).astype(self.precision)
            wv = dequantize_linear(mlx_model.model.layers[i].self_attn.v_proj).astype(self.precision)
            wo = dequantize_linear(mlx_model.model.layers[i].self_attn.o_proj).astype(self.precision)
            
            w_gate = dequantize_linear(mlx_model.model.layers[i].mlp.gate_proj).astype(self.precision)
            w_up = dequantize_linear(mlx_model.model.layers[i].mlp.up_proj).astype(self.precision)
            w_down = dequantize_linear(mlx_model.model.layers[i].mlp.down_proj).astype(self.precision)
            

            self.inner_layers.append(Qwen2TransformerBlock(
                num_attention_heads=num_attention_heads,
                num_kv_heads=num_kv_heads,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                rms_norm_eps=rms_norm_eps,
                wq = wq,
                wk = wk,
                wv = wv,
                wo = wo,
                bq = mlx_model.model.layers[i].self_attn.q_proj.bias.astype(self.precision),
                bk = mlx_model.model.layers[i].self_attn.k_proj.bias.astype(self.precision),
                bv = mlx_model.model.layers[i].self_attn.v_proj.bias.astype(self.precision),
                w_gate = w_gate,
                w_up=w_up,
                w_down = w_down,
                w_input_layernorm = mlx_model.model.layers[i].input_layernorm.weight.astype(self.precision),
                w_post_attention_layernorm=mlx_model.model.layers[i].post_attention_layernorm.weight.astype(self.precision),
                max_seq_len=mlx_model.args.max_position_embeddings,
                theta=mlx_model.args.rope_theta
            ))
        
        self.layernorm = RMSNorm(self.hidden_size, mlx_model.model.norm.weight.astype(self.precision), mlx_model.args.rms_norm_eps)
        self.mlx_model = mlx_model

        if not mlx_model.args.tie_word_embeddings:
            self.w_lm_head = dequantize_linear(mlx_model.lm_head)
        else:
            self.w_lm_head = None


    def __call__(
        self,
        inputs: mx.array,
        offset: int,
    ) -> mx.array:
        embeddings = self.embeddings(inputs)
        for transformer in self.inner_layers:
            embeddings = transformer(embeddings, offset, mask="causal")
        h = self.layernorm(embeddings)
        if self.w_lm_head is not None:
            return linear(h, self.w_lm_head)
        else:
            return self.embeddings.as_linear(h)
