"""Retarget MediaPipe arm/torso motion (tools/extract_pose.py output) onto the G1 upper body -> joint-angle reference.

  docker run --rm -v $PWD:/w -w /w --entrypoint python g1-stairs:latest tools/retarget_g1.py \
      reference/bully_pose.npz reference/bully_g1.npz

Arms: for every frame, the dancer's upper-arm and forearm directions are expressed in their torso frame; a damped
least-squares IK finds G1 shoulder pitch/roll/yaw + elbow so the robot's segments point the same way (MuJoCo FK of the
G1 model, torso frame). Direction matching, not positions, so body proportions don't matter. Waist: the shoulder line's
twist and tilt relative to the hip line, scaled down. The clip is resampled to 50 Hz, lightly smoothed, and closed into
a loop with a short eased transition from the last pose back to the first.

Output keys: joints (names, dance.JOINTS order), q (T, 16) at dt, dt, period, source_dt.
"""
import sys

import mujoco
import numpy as np

sys.path.insert(0, ".")
from g1_strut.dance import JOINTS  # noqa: E402

MJCF = "/opt/TienKung-Lab/legged_lab/assets/unitree/g1/mjcf/g1_29dof_rev_1_0_daf.xml"
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP = 11, 12, 13, 14, 15, 16, 23, 24
DT = 0.02            # output rate = policy control rate
SMOOTH_SIGMA = 1.0   # source frames
LOOP_BLEND = 0.5     # s, last pose -> first pose
WAIST_SCALE = 0.6
WAIST_LIM = {"waist_yaw_joint": 0.6, "waist_roll_joint": 0.25}

src, out = sys.argv[1], sys.argv[2]
d = np.load(src)
W, src_dt = d["world"], float(d["dt"])
assert np.isfinite(W).all(), "pose missing in some frames"


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


# --- dancer torso frame (x forward, y left, z up). MediaPipe world: x image-right, y down, z away from camera.
y_b = unit(W[:, L_SH] - W[:, R_SH])
spine = (W[:, L_SH] + W[:, R_SH]) / 2 - (W[:, L_HIP] + W[:, R_HIP]) / 2
z_b = unit(spine - np.sum(spine * y_b, -1, keepdims=True) * y_b)
x_b = np.cross(y_b, z_b)
R_b = np.stack([x_b, y_b, z_b], axis=1)  # rows: body axes in world -> v_body = R_b @ v_world


def in_body(v):
    return np.einsum("tij,tj->ti", R_b, v)


seg = {side: (unit(in_body(W[:, e] - W[:, s])), unit(in_body(W[:, w] - W[:, e])))
       for side, s, e, w in (("left", L_SH, L_EL, L_WR), ("right", R_SH, R_EL, R_WR))}
fwd_hands = np.mean([in_body(W[:, w] - W[:, s])[:, 0] for s, w in ((L_SH, L_WR), (R_SH, R_WR))], axis=0)
print(f"sanity: mean forward reach of wrists {fwd_hands.mean():+.2f} m (hands mostly in front of the body -> > 0)")

# --- waist: shoulder-line twist and spine tilt relative to the hips (hip frame, camera-vertical up)
y_h = unit(W[:, L_HIP] - W[:, R_HIP])
up_cam = np.array([0.0, -1.0, 0.0])
z_h = unit(up_cam - np.outer(y_h @ up_cam, np.ones(3)) * y_h)
x_h = np.cross(y_h, z_h)
sh_line = W[:, L_SH] - W[:, R_SH]
waist_yaw = np.arctan2(np.sum(sh_line * -x_h, -1), np.sum(sh_line * y_h, -1))  # + = right shoulder forward... see sign check
waist_roll = np.arctan2(np.sum(spine * y_h, -1), np.sum(spine * z_h, -1))    # + = leaning toward the left

# --- G1 IK on directions
m = mujoco.MjModel.from_xml_path(MJCF)
dd = mujoco.MjData(m)
torso = m.body("torso_link").id


def qadr(name):
    return m.jnt_qposadr[m.joint(name).id]


def fk_dirs(side, q4):
    dd.qpos[:] = 0
    dd.qpos[0:7] = [0, 0, 0.793, 1, 0, 0, 0]
    for j, v in zip(("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow"), q4):
        dd.qpos[qadr(f"{side}_{j}_joint")] = v
    mujoco.mj_kinematics(m, dd)
    Rt = dd.xmat[torso].reshape(3, 3).T
    p = {b: Rt @ dd.xpos[m.body(f"{side}_{b}_link").id] for b in ("shoulder_pitch", "elbow", "wrist_roll")}
    return unit(p["elbow"] - p["shoulder_pitch"]), unit(p["wrist_roll"] - p["elbow"])


def limits(side):
    return np.array([m.jnt_range[m.joint(f"{side}_{j}_joint").id] for j in
                     ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")]) * 0.95


def solve_from(side, u, f, q_init, q_prev, iters=40, lam=1e-2, reg=2e-2):
    """Damped least squares on the two segment directions, with a small pull toward the previous frame."""
    lo, hi = limits(side).T
    q = q_init.copy()
    for _ in range(iters):
        ru, rf = fk_dirs(side, q)
        r = np.concatenate([ru - u, rf - f, np.sqrt(reg) * (q - q_prev)])
        J = np.zeros((len(r), 4))
        for k in range(4):
            dq = np.zeros(4); dq[k] = 1e-4
            ru2, rf2 = fk_dirs(side, q + dq)
            J[:6, k] = (np.concatenate([ru2 - ru, rf2 - rf])) / 1e-4
        J[6:] = np.sqrt(reg) * np.eye(4)
        step = np.linalg.solve(J.T @ J + lam * np.eye(4), -J.T @ r)
        q = np.clip(q + step, lo, hi)
        if np.linalg.norm(step) < 1e-5:
            break
    ru, rf = fk_dirs(side, q)
    return q, np.degrees(np.arccos(np.clip([ru @ u, rf @ f], -1, 1)))


def seeds(side):
    """Coarse grid over the arm workspace; warm-starting from the last frame alone gets stuck in local minima."""
    sgn = 1 if side == "left" else -1
    for p in (-2.4, -1.2, 0.0, 0.8):
        for r in (0.0, 0.9):
            for y in (-1.2, 0.0, 1.2):
                for e in (-0.5, 1.0):
                    yield np.array([p, sgn * r, sgn * y, e])


def solve(side, u, f, q_prev):
    best = None
    for q0 in [q_prev, *seeds(side)]:
        q, e = solve_from(side, u, f, q0, q_prev, iters=25)
        cost = np.radians(e).sum() + 0.05 * np.linalg.norm(q - q_prev)  # ties -> the continuous branch
        if best is None or cost < best[0]:
            best = (cost, q, e)
    q, e = solve_from(side, u, f, best[1], q_prev)  # polish
    return q, e


T = len(W)
q_src = np.zeros((T, len(JOINTS)))
col = {n: i for i, n in enumerate(JOINTS)}
errs = []
for side in ("left", "right"):
    q = np.array([0.3, 0.2 if side == "left" else -0.2, 0.0, 0.9])  # G1 default arm pose
    for t in range(T):
        q, e = solve(side, seg[side][0][t], seg[side][1][t], q)
        errs.append(e)
        for j, v in zip(("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow"), q):
            q_src[t, col[f"{side}_{j}_joint"]] = v
errs = np.array(errs)
bad = np.where(errs.max(1) > 20)[0]
print("frames with >20 deg error (side:frame):", [("L" if b < T else "R") + str(b % T) for b in bad])
print(f"IK direction error: upper arm median {np.median(errs[:, 0]):.1f} deg (p90 {np.percentile(errs[:, 0], 90):.1f}), "
      f"forearm median {np.median(errs[:, 1]):.1f} deg (p90 {np.percentile(errs[:, 1], 90):.1f})")
for n, v in (("waist_yaw_joint", waist_yaw), ("waist_roll_joint", waist_roll)):
    v = v - np.median(v)  # a constant twist/tilt is the camera angle, not the dance
    q_src[:, col[n]] = np.clip(WAIST_SCALE * v, -WAIST_LIM[n], WAIST_LIM[n])

# --- smooth (Gaussian, edge-padded), close the loop, resample to DT with cubic Hermite (Catmull-Rom)
k = np.arange(-3, 4)
g = np.exp(-0.5 * (k / SMOOTH_SIGMA) ** 2); g /= g.sum()
pad = np.concatenate([q_src[:1].repeat(3, 0), q_src, q_src[-1:].repeat(3, 0)])
q_s = np.stack([np.convolve(pad[:, j], g, mode="valid") for j in range(q_src.shape[1])], 1)
n_blend = int(round(LOOP_BLEND / src_dt))
s = 0.5 - 0.5 * np.cos(np.pi * np.arange(1, n_blend + 1) / (n_blend + 1))
q_loop = np.concatenate([q_s, (1 - s)[:, None] * q_s[-1] + s[:, None] * q_s[0]])  # then wraps to q_s[0]
period = len(q_loop) * src_dt
tq = np.arange(0, period, DT) / src_dt
i0 = np.floor(tq).astype(int)
u = (tq - i0)[:, None]
P = [q_loop[(i0 + o) % len(q_loop)] for o in (-1, 0, 1, 2)]
q = 0.5 * (2 * P[1] + (-P[0] + P[2]) * u + (2 * P[0] - 5 * P[1] + 4 * P[2] - P[3]) * u**2
           + (-P[0] + 3 * P[1] - 3 * P[2] + P[3]) * u**3)
vel = np.abs(np.diff(np.concatenate([q, q[:1]]), axis=0)) / DT
print(f"period {period:.2f} s, {len(q)} samples @ {DT*1000:.0f} ms; peak joint speed {vel.max():.1f} rad/s "
      f"({JOINTS[int(vel.max(0).argmax())]}), G1 arm limit 37 rad/s")
print("range per joint:", {n: (round(float(q[:, i].min()), 2), round(float(q[:, i].max()), 2))
                            for i, n in enumerate(JOINTS) if np.ptp(q[:, i]) > 0.05})
np.savez(out, joints=np.array(JOINTS), q=q.astype(np.float32), dt=DT, period=period, source_dt=src_dt,
         source_frames=T)
