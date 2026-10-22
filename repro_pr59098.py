"""Minimal reproducer for vllm-project/vllm#59098: lost KV context on ROCm AITER FA.

A 12000-token prompt is prefilled in two chunks (max_num_batched_tokens=8192).
Its second chunk is scheduled in the same step as a short fresh prompt. On main,
sort_batch_req_ids orders that batch by scheduled tokens, so the fresh prompt
sorts ahead of the chunk, split_decodes_prefills_and_extends takes it as the
start of the prefills, and the chunk runs as a plain prefill: its attention
never sees the first 8192 tokens.

The long prompt runs twice on one engine, alone and then next to the short
prompt. Its first chunk is scheduled identically both times, so it is the
control; only the second chunk can differ.

    python repro_pr59098.py        # MODEL=<path or HF id> to override
"""

import os

os.environ.setdefault("VLLM_ROCM_USE_AITER", "1")
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")

import inspect
import random

import vllm
from vllm import LLM, SamplingParams

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-0.6B")
CHUNK, LONG, SHORT = 8192, 12000, 256
CODE = "48213"


def filler(tok, seed, n):
    rng = random.Random(seed)
    who = ["The archivist", "A courier", "The night clerk", "An engineer", "A pilot"]
    did = ["catalogued", "delivered", "inspected", "repaired", "misplaced"]
    what = ["a brass lantern", "the harbour map", "an old ledger", "a sealed letter"]
    where = ["in the east wing", "near the canal", "at the depot", "beside the mill"]
    text = " ".join(
        f"{rng.choice(who)} {rng.choice(did)} {rng.choice(what)} "
        f"{rng.choice(where)} on day {rng.randint(1, 999)}."
        for _ in range(n // 8)
    )
    ids = tok.encode(text, add_special_tokens=False)
    assert len(ids) >= n
    return ids[:n]


def build_prompts(tok):
    head = tok.encode(f"Remember this: the secret code is {CODE}.\n",
                      add_special_tokens=False)
    tail = tok.encode("\n\nQuestion: what is the secret code given at the very "
                      "beginning?\nAnswer: the secret code is",
                      add_special_tokens=False)
    long_ids = head + filler(tok, 0, LONG - len(head) - len(tail)) + tail
    return long_ids, filler(tok, 1, SHORT)


def main():
    from vllm.v1.worker.gpu.model_runner import sort_batch_req_ids

    fix_present = len(inspect.signature(sort_batch_req_ids).parameters) > 3

    llm = LLM(
        model=MODEL,
        attention_backend="ROCM_AITER_FA",
        max_num_batched_tokens=CHUNK,
        # Otherwise the second run would reuse the first run's KV.
        enable_prefix_caching=False,
        max_model_len=16384,
        gpu_memory_utilization=0.25,
        seed=0,
    )
    long_ids, short_ids = build_prompts(llm.get_tokenizer())
    long_prompt = {"prompt_token_ids": long_ids}
    short_prompt = {"prompt_token_ids": short_ids}

    params = SamplingParams(temperature=0, max_tokens=16, prompt_logprobs=0)
    alone = llm.generate([long_prompt], params, use_tqdm=False)[0]
    # The long prompt's first chunk takes the whole step, so the short prompt
    # is scheduled next to its second chunk.
    paired = llm.generate([long_prompt, short_prompt], params, use_tqdm=False)[0]

    def logprobs(out):
        return [None if d is None else d[long_ids[i]].logprob
                for i, d in enumerate(out.prompt_logprobs)]

    a, b = logprobs(alone), logprobs(paired)

    def drift(lo, hi):
        return sum(abs(a[i] - b[i]) for i in range(lo, hi)) / (hi - lo)

    # Entry i is predicted from position i - 1, so the first chunk (positions
    # 0..CHUNK-1) produces entries 1..CHUNK.
    first, second = drift(1, CHUNK + 1), drift(CHUNK + 1, LONG)

    print(f"vllm {vllm.__version__} | fix present: {fix_present} | model {MODEL}")
    print(f"long prompt {LONG} tokens, chunk {CHUNK} -> second chunk "
          f"{LONG - CHUNK} tokens; short prompt {SHORT} tokens")
    print(f"long prompt alone         -> {alone.outputs[0].text!r}")
    print(f"next to the short prompt  -> {paired.outputs[0].text!r}")
    print(f"secret code recovered     -> alone: {CODE in alone.outputs[0].text}, "
          f"paired: {CODE in paired.outputs[0].text}")
    print(f"mean |delta prompt logprob|, first chunk (control): {first:.4f}")
    print(f"mean |delta prompt logprob|, second chunk         : {second:.4f}")
    print("RESULT:", "second chunk LOST its context" if second > 0.1
          else "second chunk kept its context")


if __name__ == "__main__":
    main()
