"""Upper-body dance played straight from a retargeted clip (tools/extract_pose.py + tools/retarget_g1.py).

Reference: reference/bully_g1.npz by default (override with G1_DANCE_REF), joint angles for dance.JOINTS sampled at
the 50 Hz control rate over one loop of the clip.
"""
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_PATH = os.environ.get("G1_DANCE_REF", os.path.join(REPO, "reference", "bully_g1.npz"))


def load(path=REF_PATH):
    r = np.load(path)
    return r["q"].astype(np.float32), float(r["dt"]), float(r["period"]), [str(j) for j in r["joints"]]
