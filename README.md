# Reproducer for vllm-project/vllm#59098

[vllm-project/vllm#59098](https://github.com/vllm-project/vllm/pull/59098) fixes a bug where a long prompt that is prefilled in chunks loses the context it already has in the KV cache. It happens with the V2 model runner on ROCm with the AITER FlashAttention backend (`ROCM_AITER_FA`), when a later chunk of a long prompt is scheduled in the same step as a new prompt that has fewer tokens to process.

This repository runs one small test on two Docker images that differ only by the PR's source change.

## Run it

You need Docker, `amd-smi` on the host, one AMD GPU that nothing else is using, and access to Docker Hub and Hugging Face. From a clone of this repository:

```bash
docker build -f Dockerfile.bug -t pr59098:bug .   # official nightly + read-only diagnostic
docker build -f Dockerfile.fix -t pr59098:fix .   # pr59098:bug + the PR's source diff
GPU=0 ./run_arm.sh main repro
GPU=0 ./run_arm.sh fix  repro
```

- `GPU` is the index of the GPU to use. The script refuses a GPU that already has more than 4 GB in use.
- Docker images are local to each machine, so build them on the machine that runs the test. The first build pulls the official vLLM ROCm image, which is large.
- The model, `Qwen/Qwen3-0.6B`, is read from your Hugging Face cache (`HF_HUB_CACHE`, else `$HF_HOME/hub`, else `~/.cache/huggingface/hub`) and downloaded into it if missing (about 1.5 GB).
- Each run prints a short summary and writes `main/` or `fix/` next to the script: the full container log (`repro.log`) and the GPU, image digest and model revision (`environment_repro.txt`).

## Result

Both runs used the same AMD Instinct MI325X, one after the other, and took about 2 minutes each. The full logs are in [`results/`](results).

| | main (`pr59098:bug`) | with the PR (`pr59098:fix`) |
|---|---|---|
| Diagnostic | 8192 cached tokens ignored | no context lost |
| Prompt logprob drift, first chunk (control) | 0.0000 | 0.0000 |
| Prompt logprob drift, second chunk | 0.3101 | 0.0000 |
| Output next to the short prompt | differs from the output alone | same as the output alone |

`GPU=0 ./run_arm.sh main repro`:

```
exit code: 0
(MainProcess pid=1) INFO 10-22 20:06:47 [startup.py:96] non-default args: {'max_model_len': 16384, 'enable_prefix_caching': False, 'gpu_memory_utilization': 0.25, 'max_num_batched_tokens': 8192, 'disable_log_stats': True, 'attention_backend': 'ROCM_AITER_FA', 'model': '/hfhub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca'}
[PR59098] mixed batch 1: CONTEXT LOST on 1 request(s), 8192 cached tokens ignored
    order as sorted (0 decode-length rows not shown):
    #0   new=256   cached=0      -> prefill
    #1   new=3808  cached=8192   -> prefill   <-- LOST 8192 cached tokens
vllm 0.32.1rc1.dev276+gdf6603f05 | fix present: False | model /hfhub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca
long prompt 12000 tokens, chunk 8192 -> second chunk 3808 tokens; short prompt 256 tokens
long prompt alone         -> ' 48213.\nAnswer:\nThe secret code given at the very'
next to the short prompt  -> ' 48213\nThe answer is: 4821'
secret code recovered     -> alone: True, paired: True
mean |delta prompt logprob|, first chunk (control): 0.0000
mean |delta prompt logprob|, second chunk         : 0.3101
RESULT: second chunk LOST its context
```

`GPU=0 ./run_arm.sh fix repro`:

```
exit code: 0
(MainProcess pid=1) INFO 10-22 20:09:30 [startup.py:96] non-default args: {'max_model_len': 16384, 'enable_prefix_caching': False, 'gpu_memory_utilization': 0.25, 'max_num_batched_tokens': 8192, 'disable_log_stats': True, 'attention_backend': 'ROCM_AITER_FA', 'model': '/hfhub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca'}
[PR59098] mixed batch 1: no context lost
    order as sorted (0 decode-length rows not shown):
    #0   new=3808  cached=8192   -> extend
    #1   new=256   cached=0      -> prefill
vllm 0.32.1rc1.dev276+gdf6603f05 | fix present: True | model /hfhub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca
long prompt 12000 tokens, chunk 8192 -> second chunk 3808 tokens; short prompt 256 tokens
long prompt alone         -> ' 48213.\nAnswer:\nThe secret code given at the very'
next to the short prompt  -> ' 48213.\nAnswer:\nThe secret code given at the very'
secret code recovered     -> alone: True, paired: True
mean |delta prompt logprob|, first chunk (control): 0.0000
mean |delta prompt logprob|, second chunk         : 0.0000
RESULT: second chunk kept its context
```

## How to read the output

**The test.** [`repro_pr59098.py`](repro_pr59098.py) builds a 12000-token prompt that starts with "Remember this: the secret code is 48213.", continues with filler, and ends by asking for the code, plus a 256-token prompt of other filler. With `max_num_batched_tokens=8192`, the long prompt is prefilled in two chunks: 8192 tokens, then 3808. The script runs the long prompt twice on one engine, greedy, with prompt logprobs:

1. alone;
2. together with the short prompt. The first chunk fills the whole first step, so the short prompt is scheduled in the second step, next to the long prompt's second chunk (3808 new tokens, 8192 already cached). That is the batch that triggers the bug.

**The `[PR59098]` lines** show that batch as the attention backend received it: the order it was sorted into, and whether each row ran as an extend (new tokens that attend to the cached ones) or a prefill (new tokens only). On main the short prompt is sorted first, so the splitter starts the prefill region at row 0 and the long prompt's chunk runs as a prefill: its 8192 cached tokens are never read. With the PR the chunk is sorted first and runs as an extend.

**The drift numbers** are the mean absolute change in the long prompt's prompt logprobs between the two runs. The first chunk is computed alone both times, so it must not change, and it doesn't: any change in the second chunk comes from the batch it ran in. The second chunk is unchanged when the cached tokens are read (0.0000 with the PR) and drifts when they are not (0.3101 on main).

**The generated text** changes on main too. The secret code is still recovered on both arms: only the second chunk was computed without context, and the decode steps that produce the digits read the first chunk's KV from the cache, which is intact. The drift is the measurement; the text change is the visible symptom.

## What the two images contain

- `pr59098:bug` is the official nightly `vllm/vllm-openai-rocm:nightly-rocm100-df6603f0583375f244d1ac6d36265462b7490392` (vLLM `0.32.1rc1.dev276+gdf6603f05`, digest `sha256:81899892d73777a4ae81e16f39e749169687f29249505eb4c56da83bd097d41c`), which still has the bug, with [`diag_pr59098.py`](diag_pr59098.py) appended to `vllm/v1/attention/backends/utils.py`. The diagnostic wraps the splitter, reads the batch and logs; it changes nothing.
- `pr59098:fix` is `pr59098:bug` plus one layer that applies [`fix_branch.patch`](fix_branch.patch), the PR's source diff at its head commit: `git diff 3d5f4d4cd5d8...bd2197772c00 -- vllm/`. It applies cleanly to the nightly.

Since the fix image is built on top of the bug image, the PR's change is the only difference between the two runs.

## Which attention backends are affected

The batch order comes from `sort_batch_req_ids` in the V2 model runner (`vllm/v1/worker/gpu/model_runner.py`). The wrong order only loses context where the batch is split with `split_decodes_prefills_and_extends`, which expects requests with cached tokens to come before new ones. In the tested nightly, two places call it:

- `ROCM_AITER_FA` (`vllm/v1/attention/backends/rocm_aiter_fa.py`), which this test runs.
- `FLASH_ATTN` with decode context parallelism: `split_dcp_context_queries` in `vllm/v1/worker/cp_utils.py`, called from `flash_attn.py` only when `dcp_world_size > 1`. This path was not run here.

## Optional: gsm8k

```bash
GPU=0 ./run_arm.sh main gsm8k
GPU=0 ./run_arm.sh fix  gsm8k
```

This serves the model with `vllm serve` (`--max-num-batched-tokens 512` and prefix caching off, so new prompts often share a step with later chunks of others) and runs lm-evaluation-harness gsm8k 5-shot against it. It also needs PyPI access to install `lm_eval`. It prints `exact_match` and the diagnostic's running total of batches that lost context.

## Files

| File | |
|---|---|
| `run_arm.sh` | Runs one arm on one GPU |
| `repro_pr59098.py` | The two-request test |
| `diag_pr59098.py` | Read-only diagnostic, appended to vLLM in both images |
| `Dockerfile.bug` | Official nightly plus the diagnostic |
| `Dockerfile.fix` | `pr59098:bug` plus `fix_branch.patch` |
| `fix_branch.patch` | The PR's source diff (vLLM code, Apache-2.0) |
| `results/` | Logs of the run shown above |
