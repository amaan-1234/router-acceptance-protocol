#!/bin/bash
#SBATCH --job-name=qwen35_sweep_local
#SBATCH --output=logs/local_sweep_%j.out
#SBATCH --error=logs/local_sweep_%j.out
#SBATCH --partition=general
#SBATCH --qos=grp_yixuanh2
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#
# Local half of the ChaosNLI capacity curve: the 4B and 9B points ASU does not host.
# (2B is already complete at 450.) Serves each model on the A100 via vLLM from the
# vllm-srv env, then runs the EXISTING run_ensemble_api.py from the cac env against
# localhost -- same parser, caching, output tag (_sub450) as the ASU points, so
# eval_sweep.py picks them up automatically.
#
# Two envs in one job: vllm serve needs vllm-srv; the runner needs cac. `conda run`
# gives each command its own env without global activation.
#
# Submit from repo root:   sbatch run/local_sweep.sbatch
set -o pipefail

source ~/.bashrc 2>/dev/null || true
cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p logs

# the env var that stops the FlashInfer JIT (nvcc) crash; inherited by `conda run`
export VLLM_USE_FLASHINFER_SAMPLER=0
export LOCAL_API_KEY=EMPTY
PORT=8000
BASE_URL="http://localhost:${PORT}/v1"

# HF repo -> served-name (drives output filename: sweep_chaosnli_<name>_sub450.jsonl)
declare -A MODELS=(
  ["Qwen/Qwen3.5-4B"]="qwen35-4b"
  ["Qwen/Qwen3.5-9B"]="qwen35-9b"
)
ORDER=("Qwen/Qwen3.5-9B")

wait_healthy() {   # poll /v1/models until ready, up to ~8 min (9B compile is slower)
  for i in $(seq 1 240); do
    if curl -sf "${BASE_URL}/models" >/dev/null 2>&1; then return 0; fi
    sleep 2
  done
  return 1
}

for REPO in "${ORDER[@]}"; do
  NAME="${MODELS[$REPO]}"
  echo "=================================================================="
  echo "[$(date)] serving $REPO  as  $NAME"
  echo "=================================================================="

  conda run -n vllm-srv --no-capture-output \
    vllm serve "$REPO" \
      --served-model-name "$NAME" \
      --port "$PORT" \
      --max-model-len 8192 \
      --gpu-memory-utilization 0.80 \
    > "logs/vllm_${NAME}.out" 2>&1 &
  VLLM_PID=$!

  if ! wait_healthy; then
    echo "[ERROR] $NAME did not become healthy; see logs/vllm_${NAME}.out"
    kill "$VLLM_PID" 2>/dev/null; sleep 5; continue
  fi
  echo "[$(date)] $NAME healthy -> running 450-item sweep"

  conda run -n cac --no-capture-output \
    python -m run.run_ensemble_api \
      --model "$NAME" \
      --base-url "$BASE_URL" \
      --key-env LOCAL_API_KEY \
      --max-tokens 2048

  echo "[$(date)] $NAME sweep done -> shutting server down"
  kill "$VLLM_PID" 2>/dev/null
  for i in $(seq 1 30); do
    kill -0 "$VLLM_PID" 2>/dev/null || break
    sleep 2
  done
  sleep 30
done

echo "[$(date)] all local sweep points complete"
wc -l outputs/sweep_chaosnli_qwen35_4b_sub450.jsonl \
      outputs/sweep_chaosnli_qwen35_9b_sub450.jsonl 2>/dev/null
