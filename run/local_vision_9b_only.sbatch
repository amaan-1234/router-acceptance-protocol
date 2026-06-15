#!/bin/bash
#SBATCH --job-name=qwen35_9b_only
#SBATCH --partition=public
#SBATCH --qos=grp_yixuanh2
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/local_vision_%j.out
set -o pipefail
source ~/.bashrc 2>/dev/null || true
cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs

export VLLM_USE_FLASHINFER_SAMPLER=0
export LOCAL_API_KEY=EMPTY
PORT=8000
BASE_URL="http://localhost:${PORT}/v1"

declare -A MODELS=(
  ["Qwen/Qwen3.5-9B"]="qwen35-9b"
)
ORDER=("Qwen/Qwen3.5-9B")

wait_healthy() {
  for i in $(seq 1 240); do
    curl -sf "${BASE_URL}/models" >/dev/null 2>&1 && return 0
    sleep 2
  done
  return 1
}

for REPO in "${ORDER[@]}"; do
  NAME="${MODELS[$REPO]}"
  echo "=================================================================="
  echo "[$(date)] serving $REPO  as  $NAME (VISION)"
  echo "=================================================================="

  conda run -n vllm-srv --no-capture-output \
    vllm serve "$REPO" \
      --served-model-name "$NAME" \
      --port "$PORT" \
      --max-model-len 8192 \
      --gpu-memory-utilization 0.85 \
    > "logs/vllm_vision_${NAME}.out" 2>&1 &
  VLLM_PID=$!

  if ! wait_healthy; then
    echo "[ERROR] $NAME not healthy; see logs/vllm_vision_${NAME}.out"
    kill "$VLLM_PID" 2>/dev/null; sleep 30; continue
  fi
  echo "[$(date)] $NAME healthy -> running 450-item VISION sweep"

  conda run -n cac --no-capture-output \
    python -m run.run_vision_api \
      --model "$NAME" \
      --base-url "$BASE_URL" \
      --key-env LOCAL_API_KEY \
      --max-tokens 4096

  echo "[$(date)] $NAME vision sweep done -> shutting server down"
  kill "$VLLM_PID" 2>/dev/null
  for i in $(seq 1 30); do kill -0 "$VLLM_PID" 2>/dev/null || break; sleep 2; done
  sleep 30
done

echo "[$(date)] local vision points (4B, 9B) complete"
wc -l outputs/vision_cifar_qwen35_4b_sub450.jsonl \
      outputs/vision_cifar_qwen35_9b_sub450.jsonl 2>/dev/null
