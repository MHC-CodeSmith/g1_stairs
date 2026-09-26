"""Arm choreography after Unitree's "G1 can learn any dance" clip (Feb 2025), as a phase-locked reference.

8 beats, one pose per footstep (phrase = 4 gait cycles = 3.2 s), taken from the clip's arm moves:
  0 vogue: right hand up beside the face, left hand at the chest      1 mirror
  2 arms crossed in front of the chest (right high, left low)          3 low groove, hands at the hips
  4 right arm pointing out to the side, left hand at the chest         5 mirror
  6 right hand over the head, left forearm across the face             7 low groove
Each pose holds briefly on the beat and eases (cosine, no overshoot) into the next. The torso twists/leans with the
arms; waist pitch is left to the policy for the stairs, legs are not part of the reference.

The clock is a "dance phase" in [0, 1) over the phrase, aligned with the gait clock (beat k starts at gait phase
0 or 0.5); the policy observes it as sin/cos (see tasks.G1DanceEnv).
"""

import numpy as np

from g1_strut.dance import JOINTS, vector  # noqa: F401  (JOINTS re-exported for the reward/env)

GAIT_PERIOD = 0.8
BEATS = 8
PERIOD = BEATS * GAIT_PERIOD / 2  # one beat per footstep
HOLD = 0.4  # fraction of a beat spent holding the pose
N_SAMPLES = 800

# Right-arm poses (left arm = mirror). G1: -pitch raises forward, -roll abducts, -yaw turns a bent forearm up,
# +yaw turns it inward across the body; elbow 0 = 90 deg bend, ~1.5 straight, <0 tighter (checked with MuJoCo FK).
Z = dict(wrist_roll=0.0, wrist_pitch=0.0, wrist_yaw=0.0)
VOGUE = dict(shoulder_pitch=-0.70, shoulder_roll=-0.50, shoulder_yaw=-1.25, elbow=-0.35, **{**Z, "wrist_pitch": -0.3})
CHEST = dict(shoulder_pitch=-0.50, shoulder_roll=-0.10, shoulder_yaw=0.90, elbow=-0.30, **Z)
CROSS_HI = dict(shoulder_pitch=-0.75, shoulder_roll=-0.05, shoulder_yaw=1.00, elbow=-0.35, **Z)
CROSS_LO = dict(shoulder_pitch=-0.35, shoulder_roll=-0.10, shoulder_yaw=0.95, elbow=-0.25, **Z)
GROOVE = dict(shoulder_pitch=-0.10, shoulder_roll=-0.40, shoulder_yaw=0.20, elbow=0.45, **{**Z, "wrist_pitch": 0.3})
POINT = dict(shoulder_pitch=-0.70, shoulder_roll=-0.85, shoulder_yaw=0.00, elbow=1.35, **Z)
OVERHEAD = dict(shoulder_pitch=-2.40, shoulder_roll=-0.75, shoulder_yaw=0.00, elbow=-0.30, **Z)
FACE_ACROSS = dict(shoulder_pitch=-1.30, shoulder_roll=-0.10, shoulder_yaw=0.90, elbow=-0.55, **Z)

# (right arm, left arm given as a right-arm pose, waist yaw, waist roll) per beat
CHOREO = [
    (VOGUE, CHEST, 0.15, 0.00),
    (CHEST, VOGUE, -0.15, 0.00),
    (CROSS_HI, CROSS_LO, 0.00, 0.00),
    (GROOVE, GROOVE, 0.25, -0.05),
    (POINT, CHEST, -0.25, -0.06),
    (CHEST, POINT, 0.25, 0.06),
    (OVERHEAD, FACE_ACROSS, 0.10, 0.10),
    (GROOVE, GROOVE, -0.25, 0.05),
]


def beat_poses():
    return np.array([vector(r, l, yaw, roll) for r, l, yaw, roll in CHOREO])


def table(n=N_SAMPLES):
    """(n, len(JOINTS)) reference over dance phase [0, 1): hold each beat's pose, then cosine-ease to the next."""
    poses = beat_poses()
    out = np.zeros((n, poses.shape[1]))
    for i in range(n):
        x = i / n * BEATS
        k = int(x)
        f = x - k
        s = 0.0 if f < HOLD else 0.5 - 0.5 * np.cos(np.pi * (f - HOLD) / (1 - HOLD))
        out[i] = (1 - s) * poses[k] + s * poses[(k + 1) % BEATS]
    return out


def at_phase(phase, tab=None):
    tab = table() if tab is None else tab
    idx = (np.asarray(phase) * len(tab)).astype(int) % len(tab)
    return tab[idx]
