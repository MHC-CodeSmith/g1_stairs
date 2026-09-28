"""Adapter equivalence: run each repo's own control code on its own robot model, and at every control step compare the
observation and action it computes with what our adapter computes from the same MjData.

  docker run --rm -v $PWD:/workspace/g1_stairs g1-arena -m arena.check_adapters

The repo code is used as-is where it can be imported (playground's OnnxController, GR00T's GearWbcController,
g1_walk's observation builder and gain table); unitree_rl_gym's loop only exists inside `__main__`, so its 15 lines are
copied. Each check drives the robot with the repo's own actions, so the comparison runs along the repo's trajectory.
"""
from __future__ import annotations

import ast
import os
import sys
import types

import mujoco
import numpy as np

from arena import policies as P
from arena.world import TP, Command, joint_index, state_from


LAYOUTS = {}
WORST = {}


def track(name, mine, ref, layout):
    """Per-term max |diff| for the report."""
    off = 0
    for term, n in layout:
        e = float(np.abs(np.asarray(mine[off:off + n], float) - np.asarray(ref[off:off + n], float)).max())
        WORST.setdefault(name, {})[term] = max(WORST.get(name, {}).get(term, 0.0), e)
        off += n


def report(name, obs_err, act_err, steps, fell):
    if name in WORST:
        bad = {k: f"{v:.1e}" for k, v in WORST[name].items() if v > 1e-5}
        if bad:
            print(f"    per-term obs diff > 1e-5: {bad}")
    ok = obs_err < 1e-4 and act_err < 1e-4
    print(f"[{name}] {steps} control steps  max|obs diff| {obs_err:.2e}  max|action diff| {act_err:.2e}  "
          f"robot {'fell' if fell else 'stayed up'}  -> {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def _class_from_file(path, cls_name, ns):
    src = open(path).read()
    tree = ast.parse(src)
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), ns)
    return ns[cls_name]


# ------------------------------------------------------------------------------------------------ rl_gym
def check_rl_gym(seconds=4.0):
    import torch
    root = os.path.join(TP, "unitree_rl_gym")
    m = mujoco.MjModel.from_xml_path(os.path.join(root, "resources/robots/g1_description/scene.xml"))
    d = mujoco.MjData(m)
    m.opt.timestep = 0.002
    policy = torch.jit.load(os.path.join(root, "deploy/pre_train/g1/motion.pt"))
    kps = np.array([100, 100, 100, 150, 40, 40] * 2, np.float32)
    kds = np.array([2, 2, 2, 4, 2, 2] * 2, np.float32)
    default = np.array([-0.1, 0, 0, 0.3, -0.2, 0] * 2, np.float32)
    cmd = np.array([0.5, 0, 0.2], np.float32)
    action, target, obs, counter = np.zeros(12, np.float32), default.copy(), np.zeros(47, np.float32), 0
    names, qadr, vadr = joint_index(m)
    ad = P.RlGym()
    ad.reset(None)
    pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    oe = ae = 0.0
    n = 0
    for _ in range(int(seconds / 0.002)):          # deploy_mujoco.py loop, verbatim logic
        d.ctrl[:] = (target - d.qpos[7:]) * kps + (0 - d.qvel[6:]) * kds
        mujoco.mj_step(m, d)
        counter += 1
        if counter % 10 == 0:
            qj, dqj, quat, omega = d.qpos[7:], d.qvel[6:], d.qpos[3:7], d.qvel[3:6]
            qw, qx, qy, qz = quat
            grav = np.array([2 * (-qz * qx + qw * qy), -2 * (qz * qy + qw * qx), 1 - 2 * (qw * qw + qz * qz)])
            phase = counter * 0.002 % 0.8 / 0.8
            obs[:3] = omega * 0.25
            obs[3:6] = grav
            obs[6:9] = cmd * np.array([2.0, 2.0, 0.25])
            obs[9:21] = qj - default
            obs[21:33] = dqj * 0.05
            obs[33:45] = action
            obs[45:47] = [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)]
            st = state_from(m, d, pelvis, -1, qadr, vadr, names, counter * 0.002)
            c = Command(*cmd)
            mine = ad.obs(st, c)
            track("unitree_rl_gym", mine, obs, [("ang", 3), ("grav", 3), ("cmd", 3), ("q", 12), ("qd", 12), ("act", 12), ("phase", 2)])
            action = policy(torch.from_numpy(obs).unsqueeze(0)).detach().numpy().squeeze()
            my_target = ad.act(st, c)
            target = action * 0.25 + default
            oe, ae, n = max(oe, np.abs(mine - obs).max()), max(ae, np.abs(my_target - target).max()), n + 1
    return report("unitree_rl_gym", oe, ae, n, d.qpos[2] < 0.4)


# --------------------------------------------------------------------------------------------- playground
def check_playground(seconds=4.0):
    import onnxruntime as rt
    ns = {"np": np, "rt": rt, "mujoco": mujoco}

    class Gamepad:  # fixed command instead of a joystick
        def __init__(self, **kw):
            pass

        def get_command(self):
            return np.array([0.5, 0.0, 0.2])
    ns["Gamepad"] = Gamepad
    Ctl = _class_from_file(os.path.join(TP, "mujoco_playground/mujoco_playground/experimental/sim2sim/play_g1_joystick.py"),
                           "OnnxController", ns)
    m = P._playground_model()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 1)
    m.opt.timestep = 0.002
    ctl = Ctl(policy_path=os.path.join(TP, "mujoco_playground/mujoco_playground/experimental/sim2sim/onnx/g1_policy.onnx"),
              default_angles=np.array(m.keyframe("knees_bent").qpos[7:]), ctrl_dt=0.02, n_substeps=10,
              action_scale=0.5, vel_scale_x=1.5, vel_scale_y=0.8, vel_scale_rot=2 * np.pi)
    ad = P.PlaygroundPolicy()
    ad.reset(None)
    names, qadr, vadr = joint_index(m)
    pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    box = {"oe": 0.0, "ae": 0.0, "n": 0}
    orig_get_obs = ctl.get_obs
    c = Command(0.5, 0.0, 0.2)

    def get_obs(model, data):                 # called by their get_control inside mj_step
        ref = orig_get_obs(model, data)
        st = state_from(m, data, pelvis, torso, qadr, vadr, names, 0.0)
        mine = ad.obs(st, c)
        track("mujoco_playground", mine, ref, [("linvel", 3), ("gyro", 3), ("grav", 3), ("cmd", 3), ("q", 29), ("qd", 29), ("act", 29), ("phase", 4)])
        box["target"] = ad.act(st, c)
        box["oe"] = max(box["oe"], np.abs(mine - ref).max())
        return ref
    ctl.get_obs = get_obs
    mujoco.set_mjcb_control(ctl.get_control)
    for _ in range(int(seconds / 0.002)):
        n_before = ctl._counter
        mujoco.mj_step(m, d)
        if "target" in box:
            box["ae"] = max(box["ae"], np.abs(box.pop("target") - d.ctrl).max())
            box["n"] += 1
    mujoco.set_mjcb_control(None)
    oe, ae, n = box["oe"], box["ae"], box["n"]
    return report("mujoco_playground", oe, ae, n, d.qpos[2] < 0.4)


# ---------------------------------------------------------------------------------------------- GR00T WBC
def check_gr00t(seconds=4.0):
    import onnxruntime as ort
    import torch
    import yaml
    sys.modules.setdefault("pynput", types.ModuleType("pynput"))
    root = os.path.join(TP, "GR00T-WholeBodyControl/decoupled_wbc/sim2mujoco")
    cfg_dir = os.path.join(root, "resources/robots/g1")
    ns = {"np": np, "mujoco": mujoco, "ort": ort, "torch": torch, "yaml": yaml, "os": os, "collections": __import__("collections"),
          "threading": __import__("threading"), "time": __import__("time"), "CONFIG_PATH": cfg_dir}
    Ctl = _class_from_file(os.path.join(root, "scripts/run_mujoco_gear_wbc.py"), "GearWbcController", ns)
    Ctl.keyboard_listener = lambda self, a, b: None

    def load_onnx_policy(self, path):     # upstream returns a CUDA tensor; same model, CPU
        model = ort.InferenceSession(path)
        return lambda x: torch.tensor(model.run(None, {model.get_inputs()[0].name: x.cpu().numpy()})[0])
    Ctl.load_onnx_policy = load_onnx_policy
    cfgf = os.path.join(cfg_dir, "g1_gear_wbc.yaml")
    txt = open(cfgf).read()
    if "GR00T-WholeBodyControl-Balance" not in txt:       # the yaml still names the pre-release files
        txt = txt.replace('policy/ft92.onnx', 'policy/GR00T-WholeBodyControl-Balance.onnx').replace(
            'policy/ft109.onnx', 'policy/GR00T-WholeBodyControl-Walk.onnx')
    tmp = os.path.join("/tmp", "g1_gear_wbc.yaml")
    open(tmp, "w").write(txt)
    orig_load = Ctl.load_config

    def load_config(self, path):
        return orig_load(self, tmp)
    Ctl.load_config = load_config
    ctl = Ctl(cfg_dir)
    ctl.control_dict["loco_cmd"] = np.array([0.5, 0.0, 0.2], np.float32)
    m, d = ctl.model, ctl.data
    ad = P.Gr00tWbc()
    ad.reset(None)
    names, qadr, vadr = joint_index(m)
    pelvis, torso = ctl.base_index, ctl.torso_index
    cfg = ctl.config
    oe = ae = 0.0
    n = 0
    for _ in range(int(seconds / cfg["simulation_dt"])):   # GearWbcController.run without the viewer
        leg_tau = ctl.pd_control(ctl.target_dof_pos, d.qpos[7:22], cfg["kps"], 0, d.qvel[6:21], cfg["kds"])
        d.ctrl[:15] = leg_tau
        d.ctrl[15:] = ctl.pd_control(np.zeros(14), d.qpos[22:36], np.full(14, 100.0), 0, d.qvel[21:35], np.full(14, 0.5))
        mujoco.mj_step(m, d)
        ctl.counter += 1
        if ctl.counter % cfg["control_decimation"] == 0:
            single, _ = ctl.compute_observation(d, cfg, ctl.action, ctl.control_dict, ctl.n_joints)
            ctl.obs_history.append(single)
            for i, h in enumerate(ctl.obs_history):
                ctl.obs[i * 86:(i + 1) * 86] = h
            st = state_from(m, d, pelvis, torso, qadr, vadr, names, 0.0)
            c = Command(0.5, 0.0, 0.2, height=cfg["height_cmd"])
            pol = ctl.policy if np.linalg.norm(ctl.control_dict["loco_cmd"]) <= 0.05 else ctl.walk_policy
            ctl.action = pol(torch.from_numpy(ctl.obs).unsqueeze(0)).numpy().squeeze()
            ctl.target_dof_pos = ctl.action * cfg["action_scale"] + cfg["default_angles"]
            my_target = ad.act(st, c)           # builds the same history internally
            track("gr00t_wbc", ad.hist[-1], ctl.obs[-86:], [("cmd", 7), ("ang", 3), ("grav", 3), ("q", 29), ("qd", 29), ("act", 15)])
            oe = max(oe, np.abs(np.concatenate(list(ad.hist)) - ctl.obs).max())
            ae = max(ae, np.abs(my_target - ctl.target_dof_pos).max())
            n += 1
    return report("gr00t_wbc", oe, ae, n, d.qpos[2] < 0.4)


# ------------------------------------------------------------------------------------------ g1_walk (37)
def check_g1walk(variant="robust", seconds=4.0):
    root = os.path.join(TP, "g1_walk_isaaclab_mujoco/mujoco")
    sys.path.insert(0, root)
    from mujoco_eval import run_grid as rg
    from mujoco_eval.policy import TorchActorPolicy, build_policy_observation, load_metadata
    from mujoco_eval.terrains import configure_runtime_terrain, write_scene_xml
    meta = load_metadata(os.path.join(root, "policies/isaac_metadata.json"))
    jn = meta["joint_names"]
    default = np.asarray(meta["default_joint_pos"], float)
    scene = write_scene_xml(os.path.join(root, "assets/unitree_g1_37dof_mujoco/g1_37dof_policy_aligned.xml"),
                            "/tmp/g1walk_scene/scene_plane.xml", "plane", 1.0)
    m = mujoco.MjModel.from_xml_path(str(scene))
    m.opt.timestep = 0.001
    configure_runtime_terrain(m, "plane", 1.0, 0)
    d = mujoco.MjData(m)
    qpos_adr, qvel_adr, joint_ids, _ = rg.joint_addresses(m, jn, False)
    rg.configure_policy_armature(m, jn, joint_ids)
    act_ids = rg.actuator_ids(m, jn, joint_ids, False)
    kp, kd = rg.pd_gains(jn)
    d.qpos[2] = 0.74
    d.qpos[qpos_adr] = default
    mujoco.mj_forward(m, d)
    f = {"baseline": "baseline/policy_actor.pt", "robust": "robust/policy_actor_robust.pt"}[variant]
    policy = TorchActorPolicy(os.path.join(root, "policies", f))
    last = np.zeros(37)
    target = default.copy()
    ad = P.G1Walk37(variant)
    ad.reset(None)
    mj_names = [m.joint(j).name for j in joint_ids]
    assert mj_names == ad.joints, (mj_names, ad.joints)
    names, qadr, vadr = joint_index(m, ad.joints)
    pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    cmd = np.array([0.5, 0.0, 0.2])
    oe = ae = 0.0
    n = 0
    for step in range(int(seconds / 0.001)):       # run_grid.run_once loop (pd control mode)
        jp, jv = rg.read_joint_state(d, qpos_adr, qvel_adr, default)
        if step % 20 == 0:
            quat, lin, ang, _ = rg.read_base_state(m, d, "pelvis")
            obs = build_policy_observation(quat, lin, ang, cmd, jp, jv, default, last)
            action = policy.act(obs)
            st = state_from(m, d, pelvis, -1, qadr, vadr, names, 0.0)
            from arena.world import quat_rotate_inverse
            st.lin_vel_b, st.ang_vel_b = quat_rotate_inverse(quat, lin), quat_rotate_inverse(quat, ang)
            st.gravity_b = quat_rotate_inverse(quat, np.array([0, 0, -1.0]))   # their read_base_state quantities
            c = Command(*cmd)
            mine = ad.obs(st, c)
            track(f"g1_walk37_{variant}", mine, obs, [("lin", 3), ("ang", 3), ("grav", 3), ("cmd", 3), ("q", 37), ("qd", 37), ("act", 37)])
            my_target = ad.act(st, c)
            last = action
            target = default + 0.5 * action
            oe, ae, n = max(oe, np.abs(mine - obs).max()), max(ae, np.abs(my_target - target).max()), n + 1
        ctrl = kp * (target - jp) - kd * jv
        lo, hi = m.actuator_ctrlrange[act_ids, 0], m.actuator_ctrlrange[act_ids, 1]
        d.ctrl[act_ids] = np.clip(ctrl, lo, hi) if np.any(m.actuator_ctrllimited[act_ids]) else ctrl
        mujoco.mj_step(m, d)
    assert np.allclose(ad.kp, kp) and np.allclose(ad.kd, kd)
    return report(f"g1_walk37_{variant}", oe, ae, n, d.qpos[2] < 0.4)


# ---------------------------------------------------------------------------------------------- holosoma
def check_holosoma(variant="fastsac", seconds=4.0):
    """holosoma runs sim and policy as separate processes (DDS), so its own LocomotionPolicy observation code is
    called directly (hardware/network modules stubbed) on the arena's states, and its actions drive the robot."""
    import importlib
    src = os.path.join(TP, "holosoma/src/holosoma_inference")
    sys.path.insert(0, src)
    for mod in ("netifaces", "sshkeyboard", "pinocchio", "holosoma_inference.sdk", "holosoma_inference.utils.wandb"):   # hardware / IO / WBT-only
        stub = types.ModuleType(mod)
        for attr in ("create_interface", "load_checkpoint", "listen_keyboard", "stop_listening"):
            setattr(stub, attr, None)
        sys.modules.setdefault(mod, stub)
    L = importlib.import_module("holosoma_inference.policies.locomotion")
    obs_cfg = importlib.import_module("holosoma_inference.config.config_values.observation").loco_g1_29dof
    rob_cfg = importlib.import_module("holosoma_inference.config.config_values.robot").g1_29dof

    ref = object.__new__(L.LocomotionPolicy)
    ref.config = types.SimpleNamespace(task=types.SimpleNamespace(
        debug=types.SimpleNamespace(force_zero_angular_velocity=False, force_upright_imu=False, force_zero_action=False)))
    ref.num_dofs = 29
    ref.default_dof_angles = np.array(rob_cfg.default_dof_angles)[None]
    ref.obs_config = obs_cfg
    ref.obs_dims, ref.obs_dict, ref.obs_scales = obs_cfg.obs_dims, obs_cfg.obs_dict, obs_cfg.obs_scales
    ref.history_length_dict = obs_cfg.history_length_dict
    ref._initialize_history_state()
    ref.last_policy_action = np.zeros((1, 29))
    ref.lin_vel_command = np.array([[0.5, 0.0]])
    ref.ang_vel_command = np.array([[0.2]])
    ref.stand_command = np.array([[1]])
    ref.phase = np.array([[0.0, np.pi]])
    ref.phase_dt = 2 * np.pi / (50 * 1.0)
    ref.is_standing = False
    ref.use_phase = True

    from arena.world import Arena
    arena = Arena("flat")
    ad = L_ad = P.Holosoma(variant)
    init = dict(zip(ad.joints, ad.default))
    arena.reset(0.80, init)
    ad.reset(arena.state())
    arena.set_targets(ad.joints, ad.default, ad.kp, ad.kd)
    c = Command(0.5, 0.0, 0.2)
    oe = ae = 0.0
    n = 0
    for k in range(int(seconds / 0.002)):
        if k % 10 == 0:
            st = arena.state()
            d = arena.data
            rsd = np.concatenate([st.base_pos, st.base_quat, st.get(st.q, ad.joints), st.lin_vel_w, st.ang_vel_b,
                                  st.get(st.qd, ad.joints)])[None]
            ref.update_phase_time()                       # run loop order: phase update, then policy_action
            obs_ref = ref.prepare_obs_for_rl(rsd)["actor_obs"][0]
            a_ref = np.clip(ad.sess.run(None, {ad.inp: obs_ref[None]})[0][0], -100, 100)
            ref.last_policy_action = a_ref[None].copy()
            tgt = ad.act(st, c)
            mine = np.concatenate([ad._last_obs])
            track(f"holosoma_{variant}", mine, obs_ref, [("actions", 29), ("ang", 3), ("cmd_ang", 1), ("cmd_lin", 2),
                                                          ("cos", 2), ("q", 29), ("qd", 29), ("grav", 3), ("sin", 2)])
            oe = max(oe, np.abs(mine - obs_ref).max())
            ae = max(ae, np.abs(tgt - (a_ref * 0.25 + ad.default)).max())
            n += 1
            arena.set_targets(ad.joints, tgt, ad.kp, ad.kd)
        arena.step_physics(1)
    return report(f"holosoma_{variant}", oe, ae, n, arena.fallen())


# ------------------------------------------------------------------------------------------------ trackers
class _Stop(BaseException):
    """Ends a repo's endless sim loop (their loops catch Exception)."""


class _FakeViewer:
    def __init__(self, *a, **k):
        self.cam = types.SimpleNamespace(distance=0.0, lookat=None)
        self.opt = types.SimpleNamespace(flags={})
        self.user_scn = types.SimpleNamespace(ngeom=0)

    def render(self):
        pass

    def sync(self):
        pass

    def read_pixels(self):
        return np.zeros((8, 8, 3), np.uint8)

    def close(self):
        pass


def _tracker_state(m, d, t):
    """State with the IMU values these repos read: MuJoCo sensordata, which mj_step computes before integrating, so
    it lags qpos/qvel by one physics step."""
    names, qadr, vadr = joint_index(m)
    pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    st = state_from(m, d, pelvis, torso, qadr, vadr, names, t)
    st.base_quat = d.sensor("orientation").data.copy()
    st.ang_vel_b = d.sensor("angular-velocity").data.copy()
    return st


def check_gmt(seconds=4.0, motion="walk_stand.pkl"):
    """GMT's own sim2sim.py HumanoidEnv.run() (viewer stubbed) on its own g1.xml; its policy call is wrapped to
    compare every observation and action with the adapter's, computed from the same MjData."""
    import torch
    from arena.trackers import GMT_ROOT, Clip, Gmt
    root = GMT_ROOT
    sys.modules["mujoco_viewer"] = types.SimpleNamespace(MujocoViewer=_FakeViewer)
    sys.path.insert(0, root)
    ns = {"__name__": "gmt_sim2sim"}             # module name for torch.jit.script
    path = os.path.join(root, "sim2sim.py")
    exec(compile(open(path).read().replace('if __name__ == "__main__":', "if False:"), path, "exec"), ns)
    cwd = os.getcwd()
    os.chdir(root)
    try:
        env = ns["HumanoidEnv"]("assets/pretrained_checkpoints/pretrained.pt", os.path.join("assets/motions", motion),
                                device="cpu", record_video=True)
    finally:
        os.chdir(cwd)
    env.sim_duration = seconds
    ad = Gmt(Clip.from_gmt_pkl(os.path.join(root, "assets/motions", motion)))
    ad.reset(None)
    net, res = env.policy_jit, {"oe": 0.0, "ae": 0.0, "n": 0}
    layout = [("mimic", 600), ("ang", 3), ("rp", 2), ("q", 23), ("qd", 23), ("last", 23), ("hist", 1480)]

    def wrapped(obs_tensor):
        st = _tracker_state(env.model, env.data, 0.0)
        mine, _ = ad.obs(st)
        ref = obs_tensor[0].numpy()
        track("gmt", mine, ref, layout)
        a_ref = net(obs_tensor)
        tgt = ad.act(st)
        res["oe"] = max(res["oe"], float(np.abs(mine - ref).max()))
        res["ae"] = max(res["ae"], float(np.abs(tgt - (np.clip(a_ref[0].numpy(), -10, 10) * 0.5 + env.default_dof_pos)).max()))
        res["n"] += 1
        return a_ref

    env.policy_jit = wrapped
    import tempfile
    os.chdir(tempfile.mkdtemp())                   # run() writes its video under ./mujoco_videos
    try:
        env.run()
    finally:
        os.chdir(cwd)
    q = env.data.qpos[3:7]
    fell = (1 - 2 * (q[1] ** 2 + q[2] ** 2)) < 0.5
    return report("gmt", res["oe"], res["ae"], res["n"], fell)


def check_twist(seconds=4.0, motion="walk_stand.pkl"):
    """TWIST's own low-level sim loop (RealTimePolicyController.run, viewer stubbed) on its g1_sim2sim_with_wrist_roll.xml.
    Its reference comes over Redis from the high-level server's build_mimic_obs; a fake Redis serves the same function
    (moved from CUDA to CPU) on a GMT example clip. Observations and actions are compared at every policy call."""
    import torch
    from arena.trackers import GMT_ROOT, TWIST_ROOT, Clip, Twist, _motion_lib
    root = os.path.join(TWIST_ROOT, "deploy_real")
    sys.path.insert(0, root)
    clip = Clip.from_gmt_pkl(os.path.join(GMT_ROOT, "assets/motions", motion))
    ad = Twist(clip)
    ad.reset(None)
    ml = _motion_lib(os.path.join(TWIST_ROOT, "pose"), "pose.utils.motion_lib_pkl", clip)
    hl = os.path.join(root, "server_high_level_motion_lib.py")
    src = open(hl).read().replace('torch.device("cuda")', 'torch.device("cpu")')
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_mimic_obs")
    hns = {"torch": torch, "np": np}
    exec("from data_utils.rot_utils import euler_from_quaternion, quat_rotate_inverse, quat_rotate_inverse_torch", hns)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), hl, "exec"), hns)
    steps = torch.tensor([1], dtype=torch.int)
    state = {"t": 0}

    class FakeRedis:
        def __init__(self, *a, **k):
            pass

        def set(self, *a, **k):
            pass

        def get(self, key):
            if key == "action_mimic_g1":        # high-level server: one mimic frame per control step
                import json
                mo = hns["build_mimic_obs"](ml, state["t"], 0.02, steps)[0]
                return json.dumps(np.asarray(mo).tolist())
            return None

    ll = os.path.join(root, "server_low_level_g1_sim.py")
    lsrc = open(ll).read().replace('if __name__ == "__main__":', "if False:")
    sys.modules.setdefault("redis", types.SimpleNamespace(Redis=FakeRedis))
    sys.modules.setdefault("rich", types.SimpleNamespace(print=lambda *a, **k: None))   # console colouring only
    ns = {"__name__": "twist_low_level_sim"}
    exec(compile(lsrc, ll, "exec"), ns)
    ns["redis"] = types.SimpleNamespace(Redis=FakeRedis)
    ns["mjv"] = types.SimpleNamespace(launch_passive=lambda *a, **k: _FakeViewer())
    ns["draw_root_velocity"] = lambda *a, **k: 0
    ns["time"] = types.SimpleNamespace(time=lambda: 0.0, sleep=lambda s: None)
    ctl = ns["RealTimePolicyController"](os.path.join(TWIST_ROOT, "assets/g1/g1_sim2sim_with_wrist_roll.xml"),
                                         os.path.join(TWIST_ROOT, "assets/twist_general_motion_tracker.pt"), device="cpu")
    net, res = ctl.policy, {"oe": 0.0, "ae": 0.0, "n": 0}
    n_calls = int(seconds / 0.02)
    layout = [("mimic", 31), ("ang", 3), ("rp", 2), ("q", 23), ("qd", 23), ("last", 23), ("hist", 1050)]

    def wrapped(obs_tensor):
        if res["n"] >= n_calls:
            raise _Stop()
        st = _tracker_state(ctl.model, ctl.data, 0.0)
        mine, _ = ad.obs(st)
        ref = obs_tensor[0].numpy()
        track("twist", mine, ref, layout)
        a_ref = net(obs_tensor)
        tgt = ad.act(st)
        ref_tgt = np.clip(a_ref[0].numpy(), -10, 10) * 0.5 + ctl.default_dof_pos
        res["oe"] = max(res["oe"], float(np.abs(mine - ref).max()))
        res["ae"] = max(res["ae"], float(np.abs(tgt[:23] - ref_tgt).max()))
        res["n"] += 1
        state["t"] += 1
        return a_ref

    ctl.policy = wrapped
    try:
        ctl.run()
    except _Stop:
        pass
    q = ctl.data.qpos[3:7]
    fell = (1 - 2 * (q[1] ** 2 + q[2] ** 2)) < 0.5
    return report("twist", res["oe"], res["ae"], res["n"], fell)


if __name__ == "__main__":
    results = []
    for fn in (check_rl_gym, check_playground, check_gr00t, lambda: check_g1walk("baseline"), lambda: check_g1walk("robust"),
               lambda: check_holosoma("fastsac"), lambda: check_holosoma("ppo"), check_gmt, check_twist):
        try:
            results.append(fn())
        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append(False)
    print("ALL PASS" if all(results) else "SOME FAILED")
