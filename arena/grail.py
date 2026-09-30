"""NVlabs/GRAIL terrain tracker (`terrain_release`) in the arena.

GRAIL fine-tunes SONIC's universal-token tracker on generated stair/curb/slope motions and gives its encoder an 11 x 11
height map. Rebuilt from GRAIL's `imports/SONIC` code and the released `config.yaml` / `last.pt`:

- Encoder input, per each of 10 future reference frames (0.1 s apart, the first is the current one), 192 values:
  58 reference joint values, 6 root-orientation values, and the 128-d embedding of the height map (broadcast over
  time). The 58 quirk is reproduced: `command_multi_future` concatenates all 10 frames of joint positions and then all
  10 of joint velocities, and the non-flat view reshapes that vector to (10, 58). The reference joint velocity is the
  forward difference of the 50 Hz reference. Anchor term: the first two columns of R(robot pelvis)^-1 R(reference
  root), row-major.
- Height map: rays from the pelvis in the yaw frame toward an 11 x 11 grid (+-0.75 m, 0.15 m) one metre below,
  clamped to the ground plane; the value is the hit height relative to the pelvis. Here they are cast with MuJoCo.
- Conv2d projector (conv 1-16-32, ReLU, 2x2 max-pool after each, 128 -> 128 -> 128), MLP 1920 -> 2048 -> 1024 -> 512 ->
  512 -> 64 (SiLU), the 64 values as 2 tokens of 32 quantised by FSQ (32 levels per dimension).
- Decoder MLP 1093 -> 2048 -> 2048 -> 1024 -> 1024 -> 512 -> 512 -> 29 on [tokens (64), proprioception (1029)]:
  10-frame histories of base angular velocity, joint position - default, joint velocity, last action (raw) and
  gravity direction, then the object terms (static stair object: zero position deltas, identity orientation deltas,
  object pose in the pelvis frame). Action -> default + scale * action, as in SONIC.
- The arena has no stair mesh that matches the released assets (see docs/ARENA_REPORT.md), so `terrain_from_reference`
  builds a height field from the reference's own foot contacts.
"""
from __future__ import annotations

import os
import pickle

import mujoco
import numpy as np
import torch
import torch.nn as nn

from arena.policies import MJ29, ArenaPolicy
from arena.sonic import (ACTION_SCALE, DEFAULT, ISAACLAB_TO_MUJOCO, KD, KP, MUJOCO_TO_ISAACLAB, Motion, qconj, qmat,
                         qmul)
from arena.world import TP

CKPT = os.path.join(TP, "hf/GRAIL/terrain_release/last.pt")
N_FUT, FUT_STEP, N_HM = 10, 5, 11              # future frames, 0.1 s at 50 Hz, height-map side


def _load_state_dict(path=CKPT):
    """last.pt pickles GRAIL training classes; only the tensors are needed."""
    class Any:
        def __init__(self, *a, **k):
            pass

        def __setstate__(self, s):
            self.__dict__.update(s if isinstance(s, dict) else {"state": s})

    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            try:
                return super().find_class(mod, name)
            except Exception:
                return type(name, (Any,), {})

    class PM:
        Unpickler = U
        Pickler = pickle.Pickler
        load = staticmethod(pickle.load)
        loads = staticmethod(pickle.loads)

    return torch.load(path, map_location="cpu", weights_only=False, pickle_module=PM)["policy_state_dict"]


def _mlp(dims, act):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


def fsq(z, levels=32, eps=1e-3):
    """vector_quantize_pytorch.FSQ (all dimensions with `levels` levels), returns the code scaled to [-1, 1]."""
    lv = torch.tensor(float(levels))
    half_l = (lv - 1) * (1 + eps) / 2
    offset = 0.5 if levels % 2 == 0 else 0.0
    shift = torch.atanh(torch.tensor(offset) / half_l)
    bounded = torch.tanh(z + shift) * half_l - offset
    return torch.round(bounded) / (levels // 2)


class GrailNet(nn.Module):
    def __init__(self, sd):
        super().__init__()
        self.enc = _mlp([1920, 2048, 1024, 512, 512, 64], nn.SiLU)
        self.dec = _mlp([1093, 2048, 2048, 1024, 1024, 512, 512, 29], nn.SiLU)
        self.conv = nn.Sequential(nn.Conv2d(1, 16, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
                                  nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.head = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128))
        pre = {"enc": "actor_module.encoders.g1.module.", "dec": "actor_module.decoders.g1_dyn.module.",
               "conv": "actor_module.encoder_input_projectors.g1.height_map_z_flat.conv_layers.",
               "head": "actor_module.encoder_input_projectors.g1.height_map_z_flat.mlp_head."}
        for name, p in pre.items():
            getattr(self, name).load_state_dict({k[len(p):]: v for k, v in sd.items() if k.startswith(p)})
        self.eval()

    @torch.no_grad()
    def forward(self, cmd, anchor, height, proprio):
        """cmd (10, 58), anchor (10, 6), height (11, 11) [x, y] heights, proprio (1029,) -> action (29,)."""
        hm = self.head(self.conv(height.reshape(1, 1, N_HM, N_HM)).reshape(1, -1))              # (1, 128)
        x = torch.cat([cmd, anchor, hm.expand(N_FUT, -1)], dim=-1).reshape(1, -1)               # (1, 1920)
        tok = fsq(self.enc(x).reshape(1, 2, 32)).reshape(1, -1)                                 # (1, 64)
        return self.dec(torch.cat([tok, proprio[None]], dim=-1))[0]


def _six(q):
    m = qmat(q)
    return m[:, :2].reshape(-1)


def _yaw(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class GrailTerrain(ArenaPolicy):
    name = "grail_terrain"
    uses = "reference motion + height map"
    joints = MJ29
    kp, kd, default = KP, KD, DEFAULT

    def __init__(self, clip):
        self.clip = clip
        self.ref = Motion.from_clip(clip)
        self.net = GrailNet(_load_state_dict())
        self.obj = clip.obj                       # dict(pos (3,), quat_wxyz (4,)) of the stair object, or None
        self.arena = None
        r = np.linspace(-0.75, 0.75, N_HM)
        gx, gy = np.meshgrid(r, r, indexing="ij")
        d = np.stack([gx, gy, -np.ones_like(gx)], -1).reshape(-1, 3)
        self.dirs = d / np.linalg.norm(d, axis=1, keepdims=True)

    def bind(self, arena):
        self.arena = arena

    def _height_map(self, st):
        a = self.arena
        yaw = _yaw(st.base_quat)
        c, s = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        start = np.asarray(st.base_pos, float)
        groups = np.zeros(6, dtype=np.uint8)
        groups[3] = 1                              # arena.world.TERRAIN_GROUP
        out = np.zeros(len(self.dirs))
        gid = np.zeros(1, dtype=np.int32)
        for i, d in enumerate(self.dirs):
            w = Rz @ d
            dist = mujoco.mj_ray(a.model, a.data, start, w, groups, 1, -1, gid)
            hit = start + (dist if dist >= 0 else 5.0) * w
            den = max(start[2] - hit[2], 1e-8)
            sc = min(start[2] / den, 1.0)          # clamp to the z >= 0 ground, as in GRAIL
            hit = start + (hit - start) * sc
            out[i] = (Rz.T @ (hit - start))[2]
        return out.reshape(N_HM, N_HM)

    def reset(self, st):
        self.last = np.zeros(29)
        self.k = 0
        self.hist = None

    def _proprio(self, st):
        q, qd = st.get(st.q, MJ29), st.get(st.qd, MJ29)
        frame = {"w": st.ang_vel_b.copy(), "q": (q - DEFAULT)[MUJOCO_TO_ISAACLAB], "qd": qd[MUJOCO_TO_ISAACLAB],
                 "a": self.last.copy(), "g": st.gravity_b.copy()}
        self.hist = [frame] * 10 if self.hist is None else self.hist[1:] + [frame]
        h = lambda key: np.concatenate([f[key] for f in self.hist])            # noqa: E731  oldest first
        if self.obj is not None:
            Ra = qmat(st.base_quat)
            pos_b = Ra.T @ (self.obj["pos"] - st.base_pos)
            ori6 = _six(qmul(qconj(st.base_quat), self.obj["quat"]))
        else:
            pos_b, ori6 = np.zeros(3), np.array([1.0, 0, 0, 1, 0, 0])
        return np.concatenate([h("w"), h("q"), h("qd"), h("a"), h("g"), np.zeros(30), np.tile([1.0, 0, 0, 1, 0, 0], 10),
                               pos_b, ori6])

    def _reference(self, st):
        idx = np.minimum(self.k + FUT_STEP * np.arange(N_FUT), self.ref.T - 1)
        pos = self.ref.jpos[idx].reshape(-1)
        vel = self.ref.jvel[idx].reshape(-1)
        cmd = np.concatenate([pos, vel]).reshape(N_FUT, 58)                   # GRAIL's (10, 58) view of [pos; vel]
        anchor = np.stack([_six(qmul(qconj(st.base_quat), self.ref.quat[i])) for i in idx])
        return cmd, anchor

    def obs(self, st, cmd=None):
        c, a = self._reference(st)
        return c, a, self._height_map(st), self._proprio(st)

    def act(self, st, cmd=None):
        c, a, hm, pr = self.obs(st)
        f = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32)          # noqa: E731
        act = self.net(f(c), f(a), f(hm), f(pr)).numpy().astype(float)
        self.last = act.copy()
        self.k += 1
        return self.default + act[ISAACLAB_TO_MUJOCO] * ACTION_SCALE


# ---------------------------------------------------------------------------------------------------- terrain
SOLE = 0.0                        # ankle-roll link origin to sole, measured in terrain_from_reference


def terrain_from_reference(clip, out_path, width=2.4, res=0.02, margin=1.5, riser_shift=0.0):
    """Height field (npz) that puts the clip's feet on the ground: sole height of the stance feet as a function of
    progress along the clip's horizontal direction of travel, constant across the width."""
    from arena.world import Arena
    a = Arena("flat")
    m, d = a.model, a.data
    bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)           # noqa: E731
    feet = [bid("left_ankle_roll_link"), bid("right_ankle_roll_link")]
    # sole offset: settle standing on flat ground
    a.reset(0.80, {})
    a.set_targets(MJ29, DEFAULT, KP, KD)
    a.step_physics(500)
    sole = float(np.mean([d.xpos[b][2] for b in feet]))
    qadr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in MJ29]
    T = len(clip.dof)
    P = np.zeros((T, 2, 3))
    for t in range(T):
        d.qpos[:3] = clip.root_pos[t]
        x, y, z, w = clip.root_rot[t]
        d.qpos[3:7] = [w, x, y, z]
        d.qpos[qadr] = clip.dof[t]
        mujoco.mj_kinematics(m, d)
        for k, b in enumerate(feet):
            P[t, k] = d.xpos[b]
    v = np.linalg.norm(np.diff(P, axis=0) * clip.fps, axis=2)
    stance = np.vstack([v[:1], v]) < 0.12
    xy0, xy1 = clip.root_pos[0, :2], clip.root_pos[-1, :2]
    dirn = (xy1 - xy0) / max(np.linalg.norm(xy1 - xy0), 1e-6)
    # stance segments (>= 6 frames) -> one (progress, height) sample per footfall
    segs = []
    for k in range(2):
        t = 0
        while t < T:
            if stance[t, k]:
                u = t
                while u + 1 < T and stance[u + 1, k]:
                    u += 1
                if u - t + 1 >= 6:
                    segs.append((float(np.median((P[t:u + 1, k, :2] - xy0) @ dirn)),
                                 float(np.median(P[t:u + 1, k, 2])) - sole))
                t = u + 1
            else:
                t += 1
    segs.sort()
    S = np.array([x[0] for x in segs])
    H = np.array([x[1] for x in segs])
    # footfalls at (nearly) the same height belong to one step; risers halfway between consecutive steps
    groups, cur = [], [0]
    for i in range(1, len(H)):
        if abs(H[i] - np.mean(H[cur])) < 0.045:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)
    lvl = np.array([max(np.mean(H[g]), 0.0) for g in groups])
    lvl[lvl < 0.03] = 0.0
    sa = np.array([S[g[-1]] for g in groups])            # progress of the last footfall on each step
    sb = np.array([S[g[0]] for g in groups])             # ... and of the first
    drops = -np.diff(lvl)
    rise = float(np.median(drops[np.abs(drops) > 0.05])) if (np.abs(drops) > 0.05).any() else 0.12
    lv, rs = [lvl[0]], []
    for i in range(len(groups) - 1):
        n_st = max(int(round(abs(drops[i]) / rise)), 1)             # a jump of 2 rises hides a step nobody stood on
        lo, hi = sa[i] + 0.11, sb[i + 1] - 0.11                     # higher foot's toe .. lower foot's heel
        if hi < lo:
            lo = hi = (sa[i] + sb[i + 1]) / 2
        for j in range(1, n_st + 1):
            rs.append(lo + (hi - lo) * j / n_st if n_st > 1 else (lo + hi) / 2)
            lv.append(lvl[i] + (lvl[i + 1] - lvl[i]) * j / n_st)
    lvl, riser = np.array(lv), np.array(rs) + riser_shift
    n = int((np.linalg.norm(xy1 - xy0) + 2 * margin) / res)
    s_grid = -margin + res * np.arange(n)
    h = lvl[np.searchsorted(riser, s_grid)]
    info = {"levels": lvl.tolist(), "risers": riser.tolist()}
    nl = int(width / res)
    perp = np.array([-dirn[1], dirn[0]])
    # world-aligned grid covering the corridor
    ext = np.linalg.norm(xy1 - xy0) + 2 * margin + width
    cx = (xy0 + xy1) / 2
    nx = ny = int(ext / res)
    xs = cx[0] - ext / 2 + res * np.arange(nx)
    ys = cx[1] - ext / 2 + res * np.arange(ny)
    X, Y = np.meshgrid(xs, ys)                    # rows = y, cols = x
    rel = np.stack([X - xy0[0], Y - xy0[1]], -1)
    s = rel @ dirn
    l = rel @ perp
    inside = (np.abs(l) < width / 2) & (s > -margin) & (s < np.linalg.norm(xy1 - xy0) + margin)
    idx = np.clip(((s + margin) / res).astype(int), 0, n - 1)
    G = np.where(inside, h[idx], 0.0)
    np.savez_compressed(out_path, h=G, cx=cx, ext=ext, res=res, sole=sole, dirn=dirn, levels=lvl, risers=riser)
    return out_path


# ---------------------------------------------------------------------------------------------- GRAIL stair clips
STAIR_DIR = os.path.join(TP, "hf/GRAIL/data/stair_p1")


def stair_clip_names():
    return sorted(f[:-4] for f in os.listdir(os.path.join(STAIR_DIR, "robot")) if f.endswith(".pkl"))


def load_stair_clip(name):
    import joblib

    from arena.trackers import Clip
    R = next(iter(joblib.load(os.path.join(STAIR_DIR, "robot", name + ".pkl")).values()))
    O = next(iter(joblib.load(os.path.join(STAIR_DIR, "objects", name + ".pkl")).values()))
    qx, qy, qz, qw = O["root_quat"][0, 0]
    obj = {"pos": np.array(O["root_pos"][0, 0], float), "quat": np.array([qw, qx, qy, qz], float)}
    short = ("up_down" if "updown" in name else "down") + "_" + name.split("with_")[1].split("_")[0] + "steps"
    return Clip(R["fps"], R["root_trans_offset"], R["root_rot"], R["dof"], short, obj)


def track_on_stairs(policy, name, video=None, cam=None, riser_shift=0.0):
    """Play a GRAIL stair clip through `policy` on the reference-derived stair terrain."""
    from arena.track import track
    clip = load_stair_clip(name).anchored()
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output/grail_terrain")
    os.makedirs(out, exist_ok=True)
    path = terrain_from_reference(clip, os.path.join(out, f"{clip.name}_{policy}_{os.getpid()}.npz"),
                                  riser_shift=riser_shift)
    return track(policy, clip, video, terrain=f"hfield:{path}", lookahead=0.2, cam=cam)
