#!/bin/bash
#SBATCH --job-name=temp_abl
#SBATCH --partition=public
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/temp_abl_%j.log

cd /scratch/akalemul/cac
export PYTHONPATH=/scratch/akalemul/cac
export LOCAL_API_KEY=EMPTY
export HF_HOME=/scratch/akalemul/hf_cache
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
module load cuda/12.9 2>/dev/null || true

VLLM=/home/akalemul/.conda/envs/vllm-srv/bin/vllm
PYBIN=/home/akalemul/.conda/envs/vllm-srv/bin/python
MODEL_DIR=/scratch/akalemul/hf_cache/hub/models--Qwen--Qwen2.5-3B-Instruct
MODEL_PATH=$(ls -d $MODEL_DIR/snapshots/*/ 2>/dev/null | head -1)
PORT=8120

echo "Starting vLLM for qwen25-3b on $PORT..."
$VLLM serve $MODEL_PATH --served-model-name qwen25-3b --port $PORT \
    --max-model-len 8192 --dtype bfloat16 --enforce-eager \
    --gpu-memory-utilization 0.92 --max-num-seqs 16 > logs/vllm_tempabl_${SLURM_JOB_ID}.log 2>&1 &
VLLM_PID=$!
for i in $(seq 1 90); do sleep 10; curl -s http://localhost:$PORT/v1/models >/dev/null 2>&1 && echo "ready ${i}0s" && break
  kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM died"; tail -40 logs/vllm_tempabl_${SLURM_JOB_ID}.log; exit 1; }; done

for T in 0.0 0.7 1.0; do
  TAG="t$(echo $T | tr -d '.')"
  echo "=== temperature $T -> tag $TAG ==="
  $PYBIN run/run_ensemble_api.py --model qwen25-3b \
      --base-url http://localhost:$PORT/v1 --key-env LOCAL_API_KEY \
      --temperature $T --out-tag $TAG
done
kill $VLLM_PID
