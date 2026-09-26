"""Hand-keyframed "Spider-Man 3 strut" upper-body dance for the G1 29-DoF, locked to the DWAQ gait clock.

Moves: alternating finger guns (with a "pew" recoil), swagger torso twist toward the pointing arm, and the other
arm swinging loose behind. One gun per step: the right-hand gun peaks as the left foot lands (contralateral, like a
normal arm swing), the left-hand gun half a cycle later. The second half of the cycle is the mirror of the first.

The clock is the left-leg gait phase the policy already observes (period 0.8 s, 0 = left stance begins), so the
reference needs no new observations. Waist pitch is deliberately not in the reference: the policy keeps it for
leaning into the stairs.

Pure numpy; `table()` returns a (N_SAMPLES, len(JOINTS)) lookup over phase in [0, 1) for torch/numpy indexing.
"""

import numpy as np

JOINTS = [
    "waist_yaw_joint", "waist_roll_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
ARM = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]
# joints whose sign flips under a left/right mirror (rotations about x or z)
_MIRROR_NEG = {"shoulder_roll", "shoulder_yaw", "wrist_roll", "wrist_yaw"}

# Right-arm poses (left-arm values come from mirroring). Angles in rad; G1: -pitch raises the arm forward,
# -roll abducts the right arm; elbow 0 = forearm bent 90 deg forward, ~+1.5 = straight (checked with MuJoCo FK).
GUN = dict(shoulder_pitch=-1.30, shoulder_roll=-0.30, shoulder_yaw=0.10, elbow=1.35,
           wrist_roll=-0.80, wrist_pitch=0.0, wrist_yaw=0.0)
PEW = dict(shoulder_pitch=-1.55, shoulder_roll=-0.30, shoulder_yaw=0.10, elbow=0.95,   # recoil: kick up and bend
           wrist_roll=-0.80, wrist_pitch=-0.30, wrist_yaw=0.0)
SWING_BACK = dict(shoulder_pitch=0.55, shoulder_roll=-0.15, shoulder_yaw=0.0, elbow=1.20,
                  wrist_roll=0.0, wrist_pitch=0.0, wrist_yaw=0.0)
RELAXED = dict(shoulder_pitch=0.25, shoulder_roll=-0.20, shoulder_yaw=0.0, elbow=0.90,  # ~ G1 default arm pose
               wrist_roll=0.0, wrist_pitch=0.0, wrist_yaw=0.0)

TWIST = 0.30  # waist yaw toward the pointing arm (+ brings the right shoulder forward)
LEAN = 0.06   # waist roll: slight shoulder dip toward the pointing side

# First half-cycle keyframes (phase, right-arm pose, left-arm pose as a RIGHT-arm pose to be mirrored, waist yaw, roll)
_HALF = [
    (0.00, GUN, SWING_BACK, TWIST, -LEAN),
    (0.12, PEW, SWING_BACK, TWIST, -LEAN),
    (0.25, RELAXED, RELAXED, 0.0, 0.0),
]
N_SAMPLES = 400


def mirror_arm(pose):
    return {k: (-v if k in _MIRROR_NEG else v) for k, v in pose.items()}


def vector(right, left_as_right, yaw, roll):
    """Pose vector in JOINTS order; the left arm is given as a right-arm pose and mirrored."""
    left = mirror_arm(left_as_right)
    return np.array([yaw, roll] + [left[k] for k in ARM] + [right[k] for k in ARM])


def keyframes():
    """(phases, poses) for the full cycle; the second half mirrors the first."""
    phases, poses = [], []
    for ph, right, left, yaw, roll in _HALF:
        phases.append(ph)
        poses.append(vector(right, left, yaw, roll))
    for ph, right, left, yaw, roll in _HALF:  # mirror: left arm does what the right arm did
        phases.append(ph + 0.5)
        poses.append(vector(left, right, -yaw, -roll))
    return np.array(phases), np.array(poses)


def table(n=N_SAMPLES):
    """Periodic Catmull-Rom interpolation of the keyframes on a uniform phase grid."""
    kp, kv = keyframes()
    k = len(kp)
    out = np.zeros((n, kv.shape[1]))
    for i, ph in enumerate(np.arange(n) / n):
        j = np.searchsorted(kp, ph, side="right") - 1
        p0, p1 = kp[j], (kp[(j + 1) % k] + (1.0 if j + 1 == k else 0.0))
        s = (ph - p0) / (p1 - p0)
        a, b, c, d = kv[(j - 1) % k], kv[j], kv[(j + 1) % k], kv[(j + 2) % k]
        out[i] = 0.5 * ((2 * b) + (-a + c) * s + (2 * a - 5 * b + 4 * c - d) * s**2 + (-a + 3 * b - 3 * c + d) * s**3)
    return out


def at_phase(phase, tab=None):
    """Reference joint targets at a left-leg phase in [0, 1) (scalar or array), nearest-sample lookup."""
    tab = table() if tab is None else tab
    idx = (np.asarray(phase) * len(tab)).astype(int) % len(tab)
    return tab[idx]
