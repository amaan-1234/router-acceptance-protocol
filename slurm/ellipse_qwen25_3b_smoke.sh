#!/bin/bash
#SBATCH --job-name=ell_q25_smoke
#SBATCH --partition=public
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=logs/ellipse_q25_smoke_%j.log
cd /scratch/akalemul/cac
export PYTHONPATH=/scratch/akalemul/cac
export LOCAL_API_KEY=EMPTY
export HF_HOME=/scratch/akalemul/hf_cache
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
module load cuda/12.9 2>/dev/null || true
export CUDA_HOME=${CUDA_HOME:-$(dirname $(dirname $(which nvcc 2>/dev/null) 2>/dev/null) 2>/dev/null)}
VLLM=/home/akalemul/.conda/envs/vllm-srv/bin/vllm
PYBIN=/home/akalemul/.conda/envs/vllm-srv/bin/python
MODEL_PATH=/scratch/akalemul/hf_cache/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1
PORT=8110
echo "Checking GPU free memory..."
for i in $(seq 1 30); do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    echo "  free=${FREE} MiB"
    [ "$FREE" -gt 40000 ] && echo "GPU has room (${FREE} MiB free)" && break
    [ "$i" -eq 30 ] && echo "GPU never freed up (${FREE} MiB) — exiting" && exit 1
    sleep 20
done
echo "Starting vLLM for Qwen2.5-3B on port $PORT..."
$VLLM serve $MODEL_PATH \
    --served-model-name qwen25-3b \
    --port $PORT \
    --max-model-len 4096 \
    --dtype bfloat16 \
    --enforce-eager \
    --gpu-memory-utilization 0.85 --max-num-seqs 32 > logs/vllm_ell_q25_${SLURM_JOB_ID}.log 2>&1 &
VLLM_PID=$!
echo "Waiting for vLLM to come up..."
for i in $(seq 1 90); do
    sleep 10
    if curl -s http://localhost:$PORT/v1/models > /dev/null 2>&1; then echo "vLLM ready after ${i}0s"; break; fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then echo "vLLM died"; tail -40 logs/vllm_ell_q25_${SLURM_JOB_ID}.log; exit 1; fi
done
$PYBIN run/run_ellipse_sweep.py \
    --model qwen25-3b \
    --base-url http://localhost:$PORT/v1 \
    --key-env LOCAL_API_KEY \
    --max-tokens 2048 \
    --out-tag smoke \
    --max-tokens 512 \
    --limit 25
echo "Smoke done. Killing vLLM."
kill $VLLM_PID
