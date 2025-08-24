from typing import Optional

import mlx.core as mx


class TinyKvCache:
    def update_and_fetch(
        self, key: mx.array, value: mx.array
    ) -> tuple[mx.array, mx.array, int]:
        pass


class BatchingKvCache(TinyKvCache):
    def __init__(self, max_active_requests: int, max_seq_len: int):
        pass

    def update_and_fetch(
        self, key: mx.array, value: mx.array
    ) -> tuple[mx.array, mx.array, int]:
        pass

    def add_request(self, prefilled: TinyKvCache, id: int):
        pass

    def remove_request(self, id: int):
        pass


class TinyKvFullCache(TinyKvCache):
    def __init__(self):
        self.key = None
        self.value = None
        self.offset = 0

    def update_and_fetch(
        self, key: mx.array, value: mx.array
    ) -> tuple[mx.array, mx.array, int]:
        if self.key is None:
            self.key = key
            self.value = value
            self.offset = key.shape[1]
        else:
            self.key = mx.concat([self.key, key], axis=1)
            self.value = mx.concat([self.value, value], axis=1)
            self.offset = self.key.shape[1]

        return self.key, self.value, self.offset


class TinyKvRotatingCache(TinyKvCache):
    def __init__(self, max_seq_len: int):
        pass

    def update_and_fetch(
        self, key: mx.array, value: mx.array, offset: int
    ) -> tuple[mx.array, mx.array]:
        pass
