import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper
from .qwen2_week1 import Qwen2ModelWeek1
from .qwen2_week2 import Qwen2ModelWeek2
from typing import Callable


def simple_generate(
    model: Qwen2ModelWeek1,
    tokenizer: TokenizerWrapper,
    prompt: str,
    sampler: Callable[[mx.array], mx.array] | None,
) -> str:
    def _step(y):
        output_logits = model(y[None], 0)
        logits = output_logits[:, -1, :]
        logprobs = logits - mx.logsumexp(
            logits, keepdims=True
        )

        if sampler:
            return sampler(logprobs)

        return mx.argmax(logits, axis=-1)

    tokens = mx.array(tokenizer.encode(prompt))
    detokenizer = tokenizer.detokenizer

    detokenizer.reset()
    print(tokenizer.eos_token_id)

    while tokens[-1].item() != tokenizer.eos_token_id:
        token = _step(tokens)
        mx.eval(token)
        tokens = mx.concat([tokens, token])
        detokenizer.add_token(token.item())

    print(detokenizer.last_segment, end="", flush=True)
    



def simple_generate_with_kv_cache(
    model: Qwen2ModelWeek2, tokenizer: TokenizerWrapper, prompt: str
) -> str:
    pass


def batch_generate(
    model: any,
    tokenizer: TokenizerWrapper,
    prompts: list[str],
    max_seq_len=512,
    batch_size=5,
    prefill_step=128,
):
    pass
