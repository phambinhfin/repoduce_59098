#!/bin/bash
# Runs one arm on one free GPU. Identical for both arms except the image.
#   GPU=0 ./run_arm.sh main|fix repro|gsm8k
#
#   repro : repro_pr59098.py, the two-request reproducer
#   gsm8k : lm_eval gsm8k 5-shot against a server with chunked prefill forced
#           (--max-num-batched-tokens 512, prefix caching off), so fresh
#           prompts are routinely scheduled next to later chunks of others
set -uo pipefail
arm=${1:?usage: GPU=<n> run_arm.sh main|fix repro|gsm8k}
what=${2:?usage: GPU=<n> run_arm.sh main|fix repro|gsm8k}
GPU=${GPU:?set GPU=<index of a free GPU>}
HERE=$(cd "$(dirname "$0")" && pwd)
IMG=vllm/vllm-openai-rocm:nightly-rocm100-df6603f0583375f244d1ac6d36265462b7490392
TAG=pr59098:bug
[ "$arm" = fix ] && TAG=pr59098:fix
OUT=$HERE/$arm
# The runner's own Hugging Face cache, found the way huggingface_hub finds it.
# The whole hub is mounted, since weights can be symlinks into a hub-wide blobs/.
HUB=${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}
MODEL_REPO=$HUB/models--Qwen--Qwen3-0.6B
PORT=${PORT:-18459}

if ! docker image inspect "$TAG" >/dev/null 2>&1; then
  echo "$TAG is not in this machine's Docker. Build both images here first:" >&2
  echo "  cd $HERE && docker build -f Dockerfile.bug -t pr59098:bug . && docker build -f Dockerfile.fix -t pr59098:fix ." >&2
  exit 1
fi

if [ ! -f "$MODEL_REPO/refs/main" ]; then
  echo "Qwen/Qwen3-0.6B is not in $HUB; downloading it (about 1.5 GB)" >&2
  mkdir -p "$HUB"
  docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HUB_CACHE=/hfhub \
    -v "$HUB:/hfhub" --entrypoint hf "$TAG" download Qwen/Qwen3-0.6B >/dev/null || exit 1
fi
MODEL=/hfhub/models--Qwen--Qwen3-0.6B/snapshots/$(cat "$MODEL_REPO/refs/main")

used=$(amd-smi metric -m --csv -g "$GPU" 2>/dev/null | awk -F, 'NR==2 {print $3}')
if [ -z "$used" ]; then
  echo "amd-smi could not read GPU $GPU" >&2
  exit 1
fi
if [ "$used" -gt 4096 ]; then
  echo "GPU $GPU has ${used} MB in use -- someone else is on it" >&2
  exit 1
fi
mkdir -p "$OUT"
{
  echo "GPU $GPU: $(amd-smi static --asic -g "$GPU" | awk -F': ' '/MARKET_NAME/ {print $2}')"
  echo "image: $IMG ($(docker inspect -f '{{index .RepoDigests 0}}' "$IMG" | cut -d@ -f2))"
  echo "arm image: $TAG ($(docker inspect -f '{{.Id}}' "$TAG" | cut -c1-19))"
  echo "model: Qwen/Qwen3-0.6B @ $(cat "$MODEL_REPO/refs/main")"
  echo "date: $(date -u +%FT%TZ)"
} > "$OUT/environment_$what.txt"

# Numeric GIDs: the image's group file has no "render" entry.
GPU_FLAGS=(--device=/dev/kfd --device=/dev/dri --ipc=host --shm-size 16g
           --security-opt seccomp=unconfined --security-opt label=disable
           -e HIP_VISIBLE_DEVICES="$GPU")
for g in video render; do
  gid=$(getent group "$g" | cut -d: -f3)
  [ -n "$gid" ] && GPU_FLAGS+=(--group-add "$gid")
done
ENV_FLAGS=(-e VLLM_ROCM_USE_AITER=1 -e VLLM_USE_V2_MODEL_RUNNER=1)

if [ "$what" = repro ]; then
  docker run --rm --name "pr59098_repro_$$" "${GPU_FLAGS[@]}" "${ENV_FLAGS[@]}" \
    -e MODEL="$MODEL" -v "$HUB:/hfhub:ro" \
    -v "$HERE/repro_pr59098.py:/repro.py:ro" \
    --entrypoint python "$TAG" /repro.py > "$OUT/repro.log" 2>&1
  echo "exit code: $?"
  grep -E "attention backend|ROCM_AITER_FA" "$OUT/repro.log" | head -1
  grep -E "\[PR59098\]|order as sorted|new=[0-9]+ +cached=" "$OUT/repro.log" | head -8 \
    | sed -E 's/\r$//; s/^.*\[utils\.py:[0-9]+\] //'
  grep -E "^(vllm|long prompt|next to|secret code|mean|RESULT)" "$OUT/repro.log"
  exit 0
fi

srv=pr59098_srv_$$
cli=pr59098_cli_$$
trap 'docker rm -f "$srv" "$cli" >/dev/null 2>&1 || true' EXIT
if ss -ltn | awk '{print $4}' | grep -q ":$PORT\$"; then
  echo "port $PORT is taken; set PORT=" >&2
  exit 1
fi

docker run -d --name "$srv" --network host "${GPU_FLAGS[@]}" "${ENV_FLAGS[@]}" \
  -v "$HUB:/hfhub:ro" --entrypoint vllm "$TAG" \
  serve "$MODEL" --served-model-name Qwen/Qwen3-0.6B \
  --attention-backend ROCM_AITER_FA --gpu-memory-utilization 0.25 \
  --max-model-len 4096 --max-num-batched-tokens 512 --no-enable-prefix-caching \
  --seed 0 --port "$PORT" >/dev/null
docker run -d --name "$cli" --network host -e HF_HOME=/tmp/hf \
  -v "$HUB:/hfhub:ro" --entrypoint sleep "$IMG" infinity >/dev/null
docker exec "$cli" pip install -q "lm_eval[api]" > "$OUT/lm_eval_install.log" 2>&1
docker exec "$cli" pip show lm_eval | grep -i '^version' >> "$OUT/environment_$what.txt"

for _ in $(seq 1 180); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break
  if ! docker ps -q --filter "name=^${srv}\$" | grep -q .; then
    docker logs "$srv" > "$OUT/gsm8k_server.log" 2>&1
    echo "server exited; see $OUT/gsm8k_server.log" >&2
    exit 1
  fi
  sleep 5
done

docker exec "$cli" lm_eval --model local-completions --tasks gsm8k --num_fewshot 5 \
  --model_args "model=Qwen/Qwen3-0.6B,base_url=http://127.0.0.1:$PORT/v1/completions,num_concurrent=64,max_retries=3,tokenized_requests=False,tokenizer=$MODEL" \
  --batch_size 1 > "$OUT/gsm8k.log" 2>&1
echo "lm_eval exit code: $?"
sleep 15  # the diagnostic prints its running TOTAL every 10 s
docker logs "$srv" > "$OUT/gsm8k_server.log" 2>&1
grep -E "exact_match" "$OUT/gsm8k.log"
grep -E "attention backend|ROCM_AITER_FA" "$OUT/gsm8k_server.log" | head -1
grep "\[PR59098\] TOTAL" "$OUT/gsm8k_server.log" | tail -1
