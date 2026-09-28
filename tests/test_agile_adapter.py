"""AgileVelocityHeight (torchscript + our obs/history/decoding) must match NVIDIA's LEAPP ONNX export step by step.

  docker run --rm -v $PWD:/w -w /w --entrypoint sh g1-stairs:latest -c \
      "pip install -q onnxruntime==1.20.1 && python tests/test_agile_adapter.py"
"""
import math
import sys

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, ".")
from skills.adapters import G1_JOINTS_LAB, AgileVelocityHeight, RobotState  # noqa: E402

D = "skills/imported/agile/"
sess = ort.InferenceSession(D + "Velocity-Height-G1-History-v0.onnx")
agile = AgileVelocityHeight(D + "unitree_g1_velocity_height_history_torchscript.pt")
rng = np.random.default_rng(0)


def quat_xyzw_to_gravity(q):
    x, y, z, w = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return R.T @ np.array([0.0, 0.0, -1.0])


hist = {i.name: np.zeros(i.shape, np.float32)
        for i in sess.get_inputs() if i.name.startswith("h_policy") or i.name == "last_action_in"}
worst = 0.0
for step in range(8):
    cmd = np.array([[rng.uniform(-0.5, 1.0), rng.uniform(-0.3, 0.3), rng.uniform(-0.5, 0.5), rng.uniform(0.5, 0.72)]], np.float32)
    ang = rng.normal(0, 0.3, (1, 3)).astype(np.float32)
    axis = rng.normal(size=3); axis /= np.linalg.norm(axis); a = rng.uniform(0, 0.3)
    quat = np.array([[*(axis * math.sin(a / 2)), math.cos(a / 2)]], np.float32)
    q = (rng.normal(0, 0.3, (1, 29))).astype(np.float32)
    qd = rng.normal(0, 2.0, (1, 29)).astype(np.float32)
    feeds = {"base_velocity": cmd, "robot_root_ang_vel_b": ang, "robot_root_quat_w": quat,
             "robot_joint_pos": q, "robot_joint_vel": qd, **hist}
    if step == 0:
        # Isaac Lab (training) fills an empty history with the first frame; the export starts from zeros. Seed the
        # export the same way: take the newest slot after one push and tile it (actions stay zero).
        first = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, feeds)))
        for k in hist:
            if k.startswith("h_policy") and "actions" not in k:
                hist[k] = np.repeat(first[k.replace("_in", "_out")][-1:], 5, axis=0)
        feeds.update(hist)
    outs = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, feeds)))
    for k in list(hist):
        hist[k] = outs[k.replace("_in", "_out")]
    st = RobotState(G1_JOINTS_LAB, torch.tensor(q), torch.tensor(qd), torch.tensor(ang),
                    torch.tensor(quat_xyzw_to_gravity(quat[0]), dtype=torch.float32)[None], torch.zeros(1), torch.zeros(1))
    mine = agile.act(st, torch.tensor(cmd)).numpy()
    err = np.abs(mine - outs["joint_pos"]).max()
    print(f"step {step}: max |target diff| = {err:.2e}")
    worst = max(worst, err)
assert np.allclose(outs["joint_pos_kp_gains"][0], agile.kp.numpy()) and np.allclose(outs["joint_pos_kd_gains"][0], agile.kd.numpy())
print("PASS" if worst < 1e-3 else f"FAIL worst={worst}")
sys.exit(0 if worst < 1e-3 else 1)
