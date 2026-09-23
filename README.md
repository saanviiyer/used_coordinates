# Used coordinates

This project asks which latent coordinates a predictive world model's dynamics use, and what decides which coordinates the model builds.

The project is independent of `neurobotwm` and of the NEmo workshop manuscript. It was split off on 28 August 2026. No file in the source repository was moved or changed. `MIGRATION_MANIFEST.md` has the details.

## Claims

**Decodability and use are different properties.** A probe can read a variable out of a world model's latent state even when the model's own transition dynamics do not treat that variable as a coordinate. The two properties come apart often, and the gap can be measured.

**The observation decides which coordinates get built.** A predictive world model acquires an action-conjugate coordinate for a degree of freedom only when the observation stream makes that degree of freedom separable. Extra actuators do not give extra coordinates.

## Relation to the parent projects

`robowomo-attractors` and `neurobotwm` ask whether partial observability is sufficient to produce a persistent heading coordinate, and they find that it is. That is a claim about one variable in one setting. Those projects built a criterion to make the heading claim safe. The criterion turns out to be more general, and it supports a claim about world models in general. This project owns the criterion and the causal account. The parent projects keep the heading result.

## The criterion

Discovery uses only actions and the frozen model's own predictions. No pose labels, supervised decoder or other ground truth enter before scoring.

A candidate coordinate is a plane in latent space. The test asks whether rotating the state inside that plane imitates an extra action. The model's own next-observation predictions are the judge. A coordinate that passes is one the dynamics use. A coordinate that is only decodable may fail.

The two questions give different answers. In the parent project's shuffled-action control, heading decodes to within 20 to 24 degrees, but the conjugate plane sits 69 to 83 degrees away. The variable is present, held in a decaying trace of the last visible observation, and the dynamics do not act on it.

## The manipulation

Three environments differ only in how the observation couples to the two actuators. Dynamics, action space, image size, sensor noise, architectures, training budget and the estimator are the same in all three.

| environment | what the image depends on | coordinates built |
|---|---|---|
| serial arm | q1 and q1+q2, proximal dominant | one, at (1,0) |
| parity arm | q1+q2 only | one, at (1,1) |
| decoupled | q1 and q2 separately | two, at (1,0) and (0,1) |

The design uses eight seeds, two architectures, matched-variance nulls, and lit and shuffled-action controls. The coordinate follows what the image makes available. A serial chain cannot be made distal-dominant. This is a fact about kinematics and has nothing to do with learning. The ratio of image change per joint falls to 1.00 and no further.

`INSTRUMENT.md` describes an extension of the test beyond planar rotations, and its validation. `PREREG_DMC.md` is a pre-registration for the DeepMind Control Suite reacher task, written before any world model was trained on that buffer.

## Claim boundary

Established in this setting: the dissociation between decoding and use, and the causal dependence of which coordinates appear on observation separability.

Not established, and not to be written as if it were:

- Anything beyond 16x32 synthetic RGB, small convolutional GRUs and two degrees of freedom.
- Anything about non-circular coordinates. The criterion as implemented tests for a planar rotation, which is one group action among many.
- Any comparison with the probing, disentanglement or activation-patching literature. Until those baselines are run, "probes overstate what is used" remains a hypothesis.
- Anything about real robots or real video.

Language rule: write "action-conjugate coordinate". Do not write "the model learned a grid cell" or "the model represents joint angle". Say what was measured.

## Reproduce

Python 3 with `torch`, `numpy`, `scipy`, `scikit-learn`, `matplotlib` and `pytest`.

```bash
PYTHONPATH=src python3 -m pytest -q
PYTHONPATH=src python3 src/arm_env_diag.py --output runs/arm_env_diagnostic_all.json
PYTHONPATH=src python3 src/arm_torus.py --indir runs/arm_v6 \
  --output runs/arm_torus_search_v6.json --mode search --variant v6 \
  --batch 128 --n-null 3
PYTHONPATH=src python3 src/summarize_d1_followup.py
```

`arm_torus.py` needs the trained checkpoints in `runs/arm_v6`, which are not in this repository. The `run_*.sh` scripts train them. `install_dmc.sh` installs `dm_control` and MuJoCo for the control-suite runs.

Long runs detach with `python3 src/detach.py <log> <cmd>`, which does a real double fork with `os.setsid`. Confirm that the process has ppid 1 and a `??` TTY. Re-parenting to init alone is not enough. That mistake cost a 96-model sweep on 27 August 2026.

## Layout

    src/       environments, models, the estimator, analysis
    tests/     structural invariants for the environments and the estimator
    runs/      result JSON files for the checkpoint sweeps
    paper/     manuscript draft
    figs/      generated figures

## Data and exclusions

The repository leaves out the 236 model checkpoints (`*.pt`), logs, and the generated control-suite replay buffer (`runs/dmc/reacher_easy.npz`, 8.5 MB). `src/collect_dmc.py` regenerates the buffer.
