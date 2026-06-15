#!/bin/bash
#SBATCH --job-name=cac-2b-10k
#SBATCH --partition=public
#SBATCH --account=grp_yixuanh2
#SBATCH --qos=public
#SBATCH --time=1-00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=anonymized@example.com
#SBATCH --output=logs/%x-%j.out
set -uo pipefail

module load mamba/latest
# Initialize the conda/mamba shell hook (the step that silently no-ops in batch):
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cac
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export HF_HOME=/scratch/$USER/cac/.hf_cache
export HF_HUB_OFFLINE=1

echo "== env check =="
which python
python -c "import numpy, torch, vllm; print('numpy',numpy.__version__,'torch',torch.__version__,'cuda',torch.cuda.is_available(),'vllm',vllm.__version__)" || { echo "ENV BROKEN — aborting"; exit 1; }

echo "== 2B ensemble (active profile), N=10000 =="
python -m run.run_vision_inference --profile active --n 10000 --batch-size 64 --skip-if-exists

echo "== aggregate =="
python -m run.compute_mta
python -m run.build_results
python -m run.run_pipeline --real
python -m run.report
echo "DONE -> outputs/results.jsonl, outputs/figures/pipeline_real.png"
