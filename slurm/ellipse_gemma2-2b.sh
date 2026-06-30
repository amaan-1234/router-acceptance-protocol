#!/bin/bash
#SBATCH --job-name=ell_gemma2
#SBATCH --partition=public
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/ell_gemma2_%j.log

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
MODEL_DIR=/scratch/akalemul/hf_cache/hub/models--google--gemma-2-2b-it
MODEL_PATH=$(ls -d $MODEL_DIR/snapshots/*/ 2>/dev/null | head -1)
PORT=8112
[ -z "$MODEL_PATH" ] && echo "no snapshot under $MODEL_DIR" && exit 1

echo "Checking GPU free memory..."
for i in $(seq 1 30); do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    echo "  free=${FREE} MiB"
    [ "$FREE" -gt 30000 ] && echo "GPU has room (${FREE} MiB free)" && break
    [ "$i" -eq 30 ] && echo "GPU never freed up (${FREE} MiB) — exiting" && exit 1
    sleep 20
done

echo "Starting vLLM for gemma2-2b on port $PORT..."
$VLLM serve $MODEL_PATH \
    --served-model-name gemma2-2b \
    --port $PORT \
    --max-model-len 8192 \
    --dtype bfloat16 \
    --enforce-eager \
    --gpu-memory-utilization 0.92 --max-num-seqs 16  > logs/vllm_gemma2-2b_${SLURM_JOB_ID}.log 2>&1 &
VLLM_PID=$!

echo "Waiting for vLLM to come up..."
for i in $(seq 1 90); do
    sleep 10
    curl -s http://localhost:$PORT/v1/models > /dev/null 2>&1 && echo "vLLM ready after ${i}0s" && break
    if ! kill -0 $VLLM_PID 2>/dev/null; then echo "vLLM died — see logs/vllm_gemma2-2b_${SLURM_JOB_ID}.log"; tail -40 logs/vllm_gemma2-2b_${SLURM_JOB_ID}.log; exit 1; fi
done

$PYBIN -m run.run_ellipse_graders \
    --model gemma2-2b \
    --base-url http://localhost:$PORT/v1 \
    --key-env LOCAL_API_KEY \
    --out-tag sub450

echo "Run done. Killing vLLM."
kill $VLLM_PID
