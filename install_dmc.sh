#!/bin/bash
# labmaze needs bazel and only serves the locomotion mazes, not the control
# suite. Install everything else, then dm_control without dependency
# resolution, and verify the suite imports.
set -x
python3 -m pip install --quiet mujoco absl-py dm-env dm-tree glfw lxml \
  pyopengl pyparsing requests scipy tqdm protobuf
python3 -m pip install --quiet --no-deps dm_control
python3 - <<'PY'
try:
    from dm_control import suite
    import mujoco
    print("MUJOCO_OK", mujoco.__version__)
    print("SUITE_OK", len(suite.ALL_TASKS), "tasks")
    env = suite.load("reacher", "easy")
    ts = env.reset()
    print("RESET_OK action_spec", env.action_spec().shape)
    px = env.physics.render(64, 64, camera_id=0)
    print("RENDER_OK", px.shape, px.dtype)
except Exception as e:
    print("DMC_FAIL", type(e).__name__, e)
PY
