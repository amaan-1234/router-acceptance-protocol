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
set -euo pipefail
module load mamba/latest
source activate cac
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export HF_HOME=/scratch/$USER/cac/.hf_cache
export HF_HUB_OFFLINE=1

echo "== 2B ensemble (active profile), N=10000, both models serially =="
python -m run.run_vision_inference --profile active --n 10000 --batch-size 64 --skip-if-exists

echo "== aggregate: MTA -> results -> pipeline -> report =="
python -m run.compute_mta
python -m run.build_results
python -m run.run_pipeline --real
python -m run.report
echo "DONE -> outputs/results.jsonl, outputs/figures/pipeline_real.png"
