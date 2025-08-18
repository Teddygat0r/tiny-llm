from functools import cache
import mlx.core as mx
import math


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        self.dims = dims
        self.seq_len = seq_len
        self.base = base
        self.traditional = traditional

        self.cosfreq, self.sinfreq = self._create_cos_sin_cache(seq_len, dims, base=base)

    def _create_cos_sin_cache(self, N, D, base = 10000, scale = 1.0, dtype=mx.float32):
        half_d = D // 2
        positions = mx.arange(0, N, dtype=dtype) * scale
        freqs = mx.exp(-mx.arange(0.0, half_d, dtype=dtype) * (math.log(base) / half_d))

        theta = mx.reshape(positions, (-1, 1)) * mx.reshape(freqs, (1, -1))

        return (mx.cos(theta), mx.sin(theta))
    
    def rope(self, x1: mx.array, x2: mx.array, costheta: mx.array, sintheta: mx.array):
        costheta = costheta.reshape(-1, x1.shape[1], 1, x1.shape[-1])
        sintheta = sintheta.reshape(-1, x1.shape[1], 1, x1.shape[-1])

        rx1 = mx.multiply(x1, costheta) - mx.multiply(x2, sintheta)
        rx2 = mx.multiply(x1, sintheta) + mx.multiply(x2, costheta)
        
        return rx1, rx2

    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        N, S, H, D = x.shape

        costheta = self.cosfreq[:S, :] if offset is None else self.cosfreq[offset, :]
        sintheta = self.sinfreq[:S, :] if offset is None else self.sinfreq[offset, :]
        
        if self.traditional:
            x = x.reshape(N, S, H, D // 2, 2)
            x1 = x[..., 0]
            x2 = x[..., 1]
            rx1, rx2 = self.rope(x1, x2, costheta, sintheta)
            rx = mx.stack([rx1, rx2], axis=-1)
        else:
            x1 = x[..., :D // 2]
            x2 = x[..., D // 2:]
            rx1, rx2 = self.rope(x1, x2, costheta, sintheta)
            rx = mx.concat([rx1, rx2], axis=-1)
        
        return rx.reshape(N, S, H, D)
        