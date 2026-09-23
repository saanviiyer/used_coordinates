# Migration manifest

Date: 28 August 2026.

`used_coordinates` is an independent working project. No file was removed from
or modified in `neurobotwm`, `robowomo-attractors`, `fiete_grid_robotics` or
any other sibling. Everything here was copied, and the source repositories are
unchanged.

## What was migrated, and what deliberately was not

Migrated: the dependency closure of the conjugacy estimator and the arm
experiments, traced automatically from seven entry points and copied whole, so
this project has no cross-repository imports.

**Not migrated, on purpose.** The modular grid-code work is a different paper:
the algebraic readout whose cost is independent of precision, goal vectors
computed from phase differences with no map, and the measured ambiguity ranges
that set a code's usable range. That is a robotics and algorithms contribution
and belongs at a robotics venue, not at ICML. It stays in `neurobotwm`. Also not
migrated: the NEmo manuscript, the symbolic guard, the displacement composition
result, and everything about heading in the parent projects.

## Included evidence

- 23 source modules, the closure of `arm_torus`, `train_arm_visual`,
  `arm_env_diag`, `arm_torus_calib`, `summarize_d1_followup`,
  `analyze_arm_zeroshot` and `detach`.
- 192 trained checkpoints: variants v3, v4, v5 and v6, each two architectures
  by three conditions by eight seeds, with training metadata.
- Eight-seed conjugacy results for all four variants, in three estimator modes
  for v3, with matched-variance nulls and untrained floors.
- Environment diagnostics: per-joint image sensitivity and linear decodability
  for all four renderers.
- Instrument calibration: horizon and probe sweeps for two checkpoints.
- 13 tests: estimator equivalence against the reference path, deflation and
  orthonormality invariants, gain-grid bracketing, a uniformity check against a
  synthetic uniform circle, and five structural properties of the environments.

## Reproduction gates

1. The 13 migrated tests pass in this project against local code only.
2. The estimator's cached path reproduces the reference implementation to
   within 1e-6 for both architectures.
3. Re-running the v4 search analysis from local code and local checkpoints
   reproduces the migrated result with no differences: across eight seeds and two planes,
   the maximum absolute difference in conjugacy residual is 0.000e+00, the
   maximum absolute difference in held-out concentration is 0.000e+00, and the
   best integer basis agrees in 16 of 16 cases. Exactness is expected rather
   than surprising: state collection, the matched-null draw and the plane
   search all run from fixed seeds under `torch.no_grad`, so the analysis is
   deterministic given a checkpoint. Verified 28 August 2026, output in
   `runs/verify_v4.json`.

## Artifact hashes

    arm_torus_search.json              d93eb247b469a64049f9ceaa6f23e26784692c4c0c5618ecd9e8959c94fe02b5
    arm_torus_search_v4.json           adcf9434c4cdbde988cde31c16331e67b71a3feac9f469fd171fcff04645a6d1
    arm_torus_search_v5.json           fe89ca3ed44376941476fe4846873af6526992b2054a24ae960527bce45a041a
    arm_torus_search_v6.json           7c4050a5b05b8aaf99cde6ae564ac0accdb5c2f9d1bf5e8dfcfbdda2aa8e1fbf
    D1_followup_report.json            4d6f12ab84b75abbefd8e3be62e4ceac426d28413f81fc3512a69037db4acad4
    arm_env_diagnostic_all.json        1b8aafc08598ca913963153d5b919b708133e2dca86fd3cd4726d5093cb5ef3b

## Provenance of the result being built on

The experiments were designed and run between 26 and 28 August 2026 under
`neurobotwm/PLAN_FIETE_D145.md`, which fixed every gate before the
corresponding sweep ran and records four corrections made along the way: a
single-module candidate generator that failed its gate and was replaced by a
pooled one with the failing variant retained; a gain grid widened twice; a
sampler that placed goals outside the arena and penalised the comparator; and a
colour-keyed loss term that weighted one pointer 4.11 times the other. Those
corrections travel with the result and should be reported with it.
