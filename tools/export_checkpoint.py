"""Export a training checkpoint for release: policy weights + provenance, without optimizer state.

  docker run --rm -v $PWD:/w -w /w --entrypoint python g1-stairs:latest tools/export_checkpoint.py \
      logs/g1_body/<run>/model_1499.pt checkpoints/g1_body.pt --task g1_body --parent <warm-start checkpoint> \
      --extras height_cmd

The output loads everywhere a training checkpoint does for inference or warm starts (`model_state_dict` key); it
cannot resume training with the same optimizer state.
"""
import argparse
import hashlib
import os

import torch

p = argparse.ArgumentParser()
p.add_argument("src")
p.add_argument("dst")
p.add_argument("--task", required=True)
p.add_argument("--parent", default=None, help="checkpoint this run was warm-started from")
p.add_argument("--extras", nargs="*", default=[], help="observation extras after the upstream 100 (DwaqPolicy)")
p.add_argument("--note", default="")
a = p.parse_args()

src = torch.load(a.src, map_location="cpu", weights_only=False)
out = {
    "model_state_dict": src["model_state_dict"],
    "iter": src.get("iter"),
    "meta": {"task": a.task, "source_run": os.path.relpath(a.src), "parent": a.parent, "extras": a.extras,
             "num_obs": int(src["model_state_dict"]["decoder.4.weight"].shape[0]),
             "note": a.note},
}
torch.save(out, a.dst)
digest = hashlib.sha256(open(a.dst, "rb").read()).hexdigest()[:16]
print(f"{a.dst}: iter {out['iter']}, {out['meta']['num_obs']} obs, {os.path.getsize(a.dst) / 1e6:.1f} MB, sha256 {digest}")
