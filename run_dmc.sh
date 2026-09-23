#!/bin/bash
# Domain axis: point the criterion at a world model trained from pixels on a
# standard control-suite domain, with torque actions and a physics engine.
#
# The validity gate between training and analysis is pre-registered in
# PREREG_DMC.md as the first thing that would invalidate the measurement: if
# the model cannot predict the arm during a blackout there is no latent worth
# probing.  It is checked before any family is fitted, not after.
cd "$(dirname "$0")"
export PYTHONPATH=src
export OMP_NUM_THREADS=2
for s in 0 1 2 3; do
  ( python3 src/train_rssm_dmc.py --buffer runs/dmc/reacher_easy.npz \
      --conds dark --seeds $s --steps 1000 --outdir runs/dmc_reacher \
      >> "logs/dmc_s$s.log" 2>&1 ) &
done
wait
echo "=== dmc trained: $(ls runs/dmc_reacher/*.json | wc -l) ==="

python3 src/blackout_validity.py --indir runs/dmc_reacher \
  --buffer runs/dmc/reacher_easy.npz --pattern "dmc_rssm_dark_s*.pt" \
  --output runs/dmc_validity_reacher.json > logs/dmc_validity.log 2>&1
V=$(python3 -c "import json;print(json.load(open('runs/dmc_validity_reacher.json'))['verdict'])")
echo "=== dmc validity: $V ==="
if [ "$V" = "FAIL" ]; then
  echo "REFUSING to analyse: no model beats frame persistence during a "\
"blackout. See runs/dmc_validity_reacher.json"
  exit 1
fi

python3 src/dmc_family.py --indir runs/dmc_reacher \
  --buffer runs/dmc/reacher_easy.npz --output runs/dmc_family_reacher.json \
  > logs/dmc_family.log 2>&1
echo "DMC_DONE"
