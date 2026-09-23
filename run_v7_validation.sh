#!/bin/bash
# Instrument check: does the generalised criterion call a bounded joint a
# translation?  Eight seeds, one architecture, dark condition only -- this
# validates the estimator, it does not carry a claim.
cd "$(dirname "$0")"
export PYTHONPATH=src
export OMP_NUM_THREADS=1
for s in 0 1 2 3 4 5 6 7; do
  ( python3 src/train_arm_visual.py --model gru --conds dark --seeds $s \
      --steps 1600 --variant v7 --outdir runs/arm_v7 \
      >> "logs/arm_v7_gru_dark_s$s.log" 2>&1 ) &
done
wait
echo "=== v7 training done: $(ls runs/arm_v7/*.json | wc -l) ==="
python3 src/arm_family.py --indir runs/arm_v7 --output runs/arm_family_v7.json \
  --variant v7 --conds dark --models gru > logs/family_v7.log 2>&1
echo "V7_VALIDATION_DONE"
