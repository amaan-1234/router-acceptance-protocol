#!/bin/bash
cd /scratch/akalemul/cac
JOBS=("ell_q25:qwen25_3b:slurm/ellipse_qwen25-3b.sh"
      "ell_mistral:mistral_7b:slurm/ellipse_mistral-7b.sh"
      "ell_gemma2:gemma2_2b:slurm/ellipse_gemma2-2b.sh"
      "ell_phi35:phi35_mini:slurm/ellipse_phi35-mini.sh")
while true; do
  alldone=1
  for e in "${JOBS[@]}"; do
    IFS=: read -r job safe script <<< "$e"
    F="outputs/ellipse_graders_${safe}_sub450.jsonl"
    N=$(wc -l < "$F" 2>/dev/null || echo 0)
    RUNNING=$(squeue -u $USER -h -n "$job" -o %i | head -1)
    if [ "$N" -ge 450 ]; then echo "$(date +%H:%M) $job: DONE ($N)"
    elif [ -z "$RUNNING" ]; then echo "$(date +%H:%M) $job: $N/450, resubmitting"; sbatch "$script"; alldone=0
    else echo "$(date +%H:%M) $job: $N/450, running ($RUNNING)"; alldone=0; fi
  done
  [ "$alldone" -eq 1 ] && echo "$(date +%H:%M) ALL ELLIPSE LOCAL DONE" && break
  sleep 600
done
