#!/bin/bash
# Architecture axis: does the criterion give the same verdicts on a stochastic
# latent-variable world model as on the deterministic recurrent one?
# v6 joints wrap (expect rotation); v7 joints have stops (expect translation).
#
# The validity gate runs between training and analysis and is not optional.
# Two free-nats floors in a row silently switched the KL term off, leaving the
# prior untrained while the reconstruction loss looked fine; the prior is what
# the model rolls on during a blackout, so the family verdicts read off those
# checkpoints were meaningless.  No loss statistic caught it.  Comparing the
# blackout rollout against frame persistence does.
cd "$(dirname "$0")"
export PYTHONPATH=src
export OMP_NUM_THREADS=1
for variant in v6 v7; do
  for s in 0 1 2 3 4 5 6 7; do
    ( python3 src/train_rssm.py --variant "$variant" --conds dark --seeds $s \
        --steps 1600 --outdir "runs/rssm_$variant" \
        >> "logs/rssm_${variant}_s$s.log" 2>&1 ) &
  done
  wait
  echo "=== $variant trained: $(ls runs/rssm_$variant/*.json | wc -l) ==="
done

for variant in v6 v7; do
  python3 src/blackout_validity.py --indir "runs/rssm_$variant" \
    --variant "$variant" --pattern "arm_rssm_dark_s*.pt" \
    --output "runs/rssm_validity_$variant.json" \
    > "logs/rssm_validity_$variant.log" 2>&1
  V=$(python3 -c "import json;print(json.load(open('runs/rssm_validity_$variant.json'))['verdict'])")
  echo "=== $variant validity: $V ==="
  if [ "$V" = "FAIL" ]; then
    echo "REFUSING to analyse $variant: no model beats frame persistence "\
"during a blackout. There is no prior worth probing. See "\
"runs/rssm_validity_$variant.json"
    continue
  fi
  python3 src/arm_family.py --indir "runs/rssm_$variant" \
    --output "runs/rssm_family_$variant.json" --variant "$variant" \
    --conds dark --models rssm > "logs/rssm_family_$variant.log" 2>&1
  echo "=== $variant analysed ==="
done
echo "RSSM_DONE"
