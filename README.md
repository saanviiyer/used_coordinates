# Used coordinates

Which latent coordinates does a predictive world model's *dynamics* actually
use, and what decides which ones get built?

This project is independent of `neurobotwm` and of the NEmo manuscript. It was
separated on 28 August 2026. No file in the source repository was moved or
modified; see `MIGRATION_MANIFEST.md`.

## The claim

**Decodability is not use.** That a variable can be read out of a world model's
latent state says nothing about whether the model's own transition dynamics
treat it as a coordinate. The two come apart, measurably and often.

**What gets built is set by the observation, not by the action space.** A
predictive world model acquires an action-conjugate coordinate for a degree of
freedom when the observation stream makes that degree of freedom separable, and
not otherwise. Extra actuators do not buy extra coordinates.

## Why this is not the parent project's claim

`robowomo-attractors` and `neurobotwm` ask whether partial observability is
sufficient to produce a persistent heading coordinate, and answer yes. That is
a claim about one variable in one setting. The criterion those projects
developed to make the claim safe turns out to be the more general object, and
it supports a claim about world models rather than about heading. This project
owns the criterion and the causal account; the parent projects keep the heading
result.

## The criterion

Discovery uses actions and the frozen model's own predictions. No pose labels,
no supervised decoder, no ground truth of any kind enters before scoring.

A candidate coordinate is a plane in latent space. The question asked of it is
whether **rotating the state inside that plane imitates taking an extra
action**, judged against the model's own next-observation predictions. A
coordinate that satisfies this is one the dynamics use. A coordinate that is
merely decodable need not.

The two questions dissociate. In the parent project's shuffled-action control,
heading decodes at 20 to 24 degrees while the conjugate plane sits 69 to 83
degrees away: the variable is present, held in a decaying trace of the last
visible observation, and the dynamics do not act on it.

## The manipulation

Three environments differing only in how the observation couples to the two
actuators. Dynamics, action space, image size, sensor noise, architectures,
training budget and the estimator are identical across all three.

| environment | what the image depends on | coordinates built |
|---|---|---|
| serial arm | q1 and q1+q2, proximal dominant | one, at (1,0) |
| parity arm | q1+q2 only | one, at (1,1) |
| decoupled | q1 and q2 separately | two, at (1,0) and (0,1) |

Eight seeds, two architectures, matched variance nulls, lit and shuffled-action
controls. The coordinate follows what the image affords. A serial chain cannot
be made distal-dominant, which is a fact about kinematics rather than about
learning: the ratio of image change per joint falls to exactly 1.00 and no
further.

## Claim boundary

What is established: the dissociation, and the causal dependence of which
coordinates appear on observation separability, in the setting below.

What is **not** established, and must not be written as if it were:

- Anything beyond 16x32 synthetic RGB, small convolutional GRUs, and two
  degrees of freedom.
- Anything about non-circular coordinates. The criterion as implemented tests
  for a planar rotation, which is one group action among many.
- Any comparison against the probing, disentanglement or activation-patching
  literature. Until those baselines are run, "probes overstate what is used" is
  a hypothesis, not a result.
- Anything about real robots or real video.

Language rule: write "action-conjugate coordinate", never "the model learned a
grid cell" or "the model represents joint angle". Say what was measured.

## Layout

    src/       environments, models, the estimator, analysis
    tests/     structural invariants for the environments and the estimator
    runs/      192 checkpoints across four variants, plus result JSON
    paper/     manuscript, once there is one
    figs/      generated figures

## Reproduce

```bash
PYTHONPATH=src python3 -m pytest -q
PYTHONPATH=src python3 src/arm_env_diag.py --output runs/arm_env_diagnostic_all.json
PYTHONPATH=src python3 src/arm_torus.py --indir runs/arm_v6 \
  --output runs/arm_torus_search_v6.json --mode search --variant v6 \
  --batch 128 --n-null 3
PYTHONPATH=src python3 src/summarize_d1_followup.py
```

Long runs detach with `python3 src/detach.py <log> <cmd>`, which does the real
double fork with `os.setsid`. Confirm ppid 1 **and** a `??` TTY. Re-parenting
to init alone is not enough; that mistake cost this line of work a 96-model
sweep on 27 August 2026.
