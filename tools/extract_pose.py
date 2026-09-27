"""Extract per-frame 3D body landmarks (MediaPipe Pose, heavy model) from a dance clip (GIF/MP4) -> .npz + overlay.

  docker run --rm -v $PWD:/w -w /w g1-pose python tools/extract_pose.py reference/bully_maguire.gif reference/bully_pose.npz

Stores world landmarks (metres, hip-centred), image landmarks and visibility for the 33 MediaPipe keypoints, plus
the frame period. The overlay sheet (<out>.png) is for checking that the tracker locked on the right person.
"""
import sys

import cv2
import imageio.v2 as iio
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

src, out = sys.argv[1], sys.argv[2]
reader = iio.get_reader(src)
dt = (reader.get_meta_data().get("duration") or 100) / 1000.0
frames = [np.ascontiguousarray(f[..., :3]) for f in reader]

opts = PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path="/models/pose_landmarker_heavy.task"),
                             running_mode=RunningMode.VIDEO, num_poses=1,
                             min_pose_detection_confidence=0.3, min_tracking_confidence=0.3)
world, image, vis = [], [], []
with PoseLandmarker.create_from_options(opts) as lm:
    for i, f in enumerate(frames):
        big = cv2.resize(f, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        r = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=big), int(i * dt * 1000))
        if r.pose_world_landmarks:
            world.append([[p.x, p.y, p.z] for p in r.pose_world_landmarks[0]])
            image.append([[p.x, p.y] for p in r.pose_landmarks[0]])
            vis.append([p.visibility for p in r.pose_landmarks[0]])
        else:
            world.append(np.full((33, 3), np.nan)); image.append(np.full((33, 2), np.nan)); vis.append(np.zeros(33))
world, image, vis = np.array(world), np.array(image), np.array(vis)
print(f"{len(frames)} frames @ {dt*1000:.0f} ms, detected in {int(np.isfinite(world[:, 0, 0]).sum())}")
np.savez(out, world=world, image=image, visibility=vis, dt=dt)

# overlay: shoulders-elbows-wrists (L green, R red) + hips, on 20 frames
BONES = [(11, 13, (0, 220, 0)), (13, 15, (0, 220, 0)), (12, 14, (230, 0, 0)), (14, 16, (230, 0, 0)),
         (11, 12, (255, 255, 0)), (23, 24, (255, 255, 0)), (11, 23, (255, 255, 0)), (12, 24, (255, 255, 0))]
tiles = []
for i in np.linspace(0, len(frames) - 1, 20).astype(int):
    f = frames[i].copy(); h, w = f.shape[:2]
    if np.isfinite(image[i, 0, 0]):
        for a, b, c in BONES:
            pa, pb = (image[i, a] * [w, h]).astype(int), (image[i, b] * [w, h]).astype(int)
            cv2.line(f, tuple(pa), tuple(pb), c, 3)
    cv2.putText(f, str(i), (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    tiles.append(f[::2, ::2])
iio.imwrite(out.replace(".npz", ".png"), np.concatenate([np.concatenate(tiles[r*5:(r+1)*5], 1) for r in range(4)], 0))
