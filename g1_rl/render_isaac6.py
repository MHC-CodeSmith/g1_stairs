"""Render an Isaac Lab rollout (.npz from rollout_isaac.py) with Isaac Sim 6.0's RTX renderer -> PNG frames.

  /isaac-sim/python.sh g1_rl/render_isaac6.py --rollout output/rollout.npz --frames output/frames_rollout

Kinematic replay: the same G1 USD the policy was trained with is referenced, every link is posed from the recorded
world transforms (the USD's links are all direct children of its root prim), and physics never runs. The staircase
is the exact terrain mesh from the rollout. A camera tracks the pelvis from the side.
"""
import argparse
import os

from isaacsim import SimulationApp

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--rollout", required=True)
parser.add_argument("--frames", required=True, help="output directory for frame_%%05d.png")
parser.add_argument("--robot_usd", default="G1DWAQ_Lab/TienKung-Lab/legged_lab/assets/unitree/g1/g1.usd")
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--every", type=int, default=2, help="render every N-th control step (50 Hz / 2 = 25 fps)")
parser.add_argument("--start", type=int, default=0, help="first control step to render")
parser.add_argument("--max_frames", type=int, default=0, help="stop after N frames (0 = all)")
parser.add_argument("--cam_offset", type=float, nargs=3, default=[0.8, -3.6, 0.55])
args = parser.parse_args()

app = SimulationApp({"headless": True, "width": args.width, "height": args.height})

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, UsdGeom, UsdLux, Vt  # noqa: E402

data = np.load(args.rollout, allow_pickle=False)
names = [str(n) for n in data["body_names"]]
pos, quat = data["body_pos"], data["body_quat"]  # (T, B, 3), (T, B, 4) w,x,y,z

ctx = omni.usd.get_context()
ctx.new_stage()
stage = ctx.get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)

terrain = UsdGeom.Mesh.Define(stage, "/World/terrain")
faces = data["terrain_faces"]
terrain.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(data["terrain_vertices"].astype(np.float32)))
terrain.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
terrain.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.reshape(-1).astype(np.int32)))
terrain.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
terrain.CreateDisplayColorAttr([Gf.Vec3f(0.22, 0.24, 0.28)])

UsdLux.DomeLight.Define(stage, "/World/sky").CreateIntensityAttr(220.0)
sun = UsdLux.DistantLight.Define(stage, "/World/sun")
sun.CreateIntensityAttr(1100.0)
sun.CreateAngleAttr(1.0)
UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-35.0, 25.0, 0.0))

robot = stage.DefinePrim("/World/G1", "Xform")
robot.GetReferences().AddReference(os.path.abspath(args.robot_usd))
ops = []
for n in names:
    prim = stage.GetPrimAtPath(f"/World/G1/{n}")
    if not prim.IsValid():
        raise RuntimeError(f"body {n} not found under /World/G1")
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    ops.append((xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble), xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)))

cam = UsdGeom.Camera.Define(stage, "/World/cam")
cam.CreateFocalLengthAttr(24.0)
cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 200.0))
cam_op = UsdGeom.Xformable(cam).AddTransformOp(UsdGeom.XformOp.PrecisionDouble)

rp = rep.create.render_product(str(cam.GetPath()), (args.width, args.height))
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach([rp])

pelvis = names.index("pelvis")
os.makedirs(args.frames, exist_ok=True)
smooth = None
for i in range(10):  # let materials/shaders settle
    app.update()
k = 0
for t in range(args.start, len(pos), args.every):
    for (t_op, o_op), p, q in zip(ops, pos[t], quat[t]):
        t_op.Set(Gf.Vec3d(*map(float, p)))
        o_op.Set(Gf.Quatd(float(q[0]), Gf.Vec3d(float(q[1]), float(q[2]), float(q[3]))))
    target = pos[t, pelvis].astype(float)
    smooth = target if smooth is None else 0.7 * smooth + 0.3 * target  # damp gait bob in the camera
    eye = smooth + np.array(args.cam_offset)
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*(smooth - [0, 0, 0.1])), Gf.Vec3d(0, 0, 1))
    cam_op.Set(view.GetInverse())
    rep.orchestrator.step(rt_subframes=8, delta_time=0.0)  # converge each pose: no ghosting from the last one
    img = rgb.get_data()
    if img is not None and img.size:
        Image.fromarray(np.asarray(img)[..., :3]).save(os.path.join(args.frames, f"frame_{k:05d}.png"))
        k += 1
    if args.max_frames and k >= args.max_frames:
        break
print(f"[render] wrote {k} frames to {args.frames}", flush=True)
os._exit(0)
