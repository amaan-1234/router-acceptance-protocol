#!/bin/bash
cd /scratch/akalemul/cac
while true; do
  F="outputs/vision_cifar_qwen35_27b_sub450.jsonl"
  N=$(wc -l < "$F" 2>/dev/null || echo 0)
  RUNNING=$(squeue -u $USER -h -n vision_27b -o %i | head -1)
  if [ "$N" -ge 450 ]; then
    echo "$(date +%H:%M) 27b: DONE ($N)"; break
  elif [ -z "$RUNNING" ]; then
    echo "$(date +%H:%M) 27b: $N/450, not running -> resubmitting"
    sbatch slurm/vision_27b_1gpu.sh
  else
    echo "$(date +%H:%M) 27b: $N/450, running ($RUNNING)"
  fi
  sleep 600
done
