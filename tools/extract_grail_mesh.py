"""Extract the stair meshes of GRAIL's released clips from their USD files (needs usd-core, so run it in a throwaway
container) and print their bounds; writes third_party/hf/GRAIL/meshes/<clip>.npz. Used for the report's statement that
the released stair meshes (about 1.2 x 1.25 x 2.0 m) cannot be placed to match the reference motions.

  docker run --rm -v $PWD:/w python:3.11-slim sh -c "pip install -q usd-core numpy && python /w/tools/extract_grail_mesh.py"
"""
import glob
import os
import urllib.request

import numpy as np
from pxr import Usd, UsdGeom

ROOT = "/w/third_party/hf/GRAIL"
URL = ("https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL/resolve/main/"
       "data/stair_p1/object_usd/")
os.makedirs(f"{ROOT}/meshes", exist_ok=True)
for r in sorted(glob.glob(f"{ROOT}/data/stair_p1/robot/*.pkl")):
    n = os.path.basename(r)[:-4]
    usd = f"/tmp/{n}.usd"
    urllib.request.urlretrieve(URL + n + ".usd", usd)
    st = Usd.Stage.Open(usd)
    cache, V, F = UsdGeom.XformCache(), [], []
    for p in st.Traverse():
        if p.IsA(UsdGeom.Mesh):
            m = UsdGeom.Mesh(p)
            pts = np.array(m.GetPointsAttr().Get(), float)
            M = np.array(cache.GetLocalToWorldTransform(p)).T
            pts = pts @ M[:3, :3].T + M[:3, 3]
            cnt, idx = np.array(m.GetFaceVertexCountsAttr().Get()), np.array(m.GetFaceVertexIndicesAttr().Get())
            o, tri = 0, []
            for c in cnt:
                tri += [[idx[o], idx[o + k], idx[o + k + 1]] for k in range(1, c - 1)]
                o += c
            F.append(np.array(tri) + sum(len(v) for v in V))
            V.append(pts)
    V, F = np.concatenate(V), np.concatenate(F)
    print(n[-50:], "up", UsdGeom.GetStageUpAxis(st), "bounds", V.min(0).round(2), V.max(0).round(2))
    np.savez_compressed(f"{ROOT}/meshes/{n}.npz", v=V, f=F)
