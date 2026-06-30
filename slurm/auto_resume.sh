#!/bin/bash
# Resubmits 4B/9B vision jobs until each output hits 450. Resume skips done idx.
cd /scratch/akalemul/cac
while true; do
  for M in 27b; do
    F="outputs/vision_cifar_qwen35_${M}_sub450.jsonl"
    N=$(wc -l < "$F" 2>/dev/null || echo 0)
    RUNNING=$(squeue -u $USER -h -n vision_${M} -o %i | head -1)
    if [ "$N" -ge 450 ]; then
      echo "$(date +%H:%M) ${M}: DONE ($N)"
    elif [ -z "$RUNNING" ]; then
      echo "$(date +%H:%M) ${M}: $N/450, not running -> resubmitting"
      sbatch slurm/vision_${M}.sh
    else
      echo "$(date +%H:%M) ${M}: $N/450, running ($RUNNING)"
    fi
  done
  N4=$(wc -l < outputs/vision_cifar_qwen35_4b_sub450.jsonl 2>/dev/null || echo 0)
  N9=$(wc -l < outputs/vision_cifar_qwen35_9b_sub450.jsonl 2>/dev/null || echo 0)
  [ "$N4" -ge 450 ] && [ "$N9" -ge 450 ] && echo "BOTH DONE" && break
  sleep 600
done
