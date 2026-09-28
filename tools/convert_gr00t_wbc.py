"""Rebuild GR00T-WholeBodyControl's Walk/Balance ONNX policies as TorchScript for the Isaac Lab skill layer.

  docker run --rm -v $PWD:/workspace/g1_stairs g1-arena tools/convert_gr00t_wbc.py

Graph (both files): estimator MLP(516 -> 256 -> 256 -> 35, ELU) on the 6-frame history; its output splits into a
3-d velocity estimate and a 32-d latent that is L2-normalized (eps 1e-12); actor MLP(121 -> 512 -> 256 -> 256 -> 15,
ELU) on [current 86-d frame, velocity, latent]. Weights are copied from the ONNX initializers and the result is
checked against onnxruntime. Output goes to third_party/converted/ (gitignored: NVIDIA Open Model License weights).
"""
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import numpy_helper

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arena.world import TP  # noqa: E402

SRC = os.path.join(TP, "GR00T-WholeBodyControl/decoupled_wbc/sim2mujoco/resources/robots/g1/policy")
OUT = os.path.join(TP, "converted/gr00t_wbc")


class Gr00tWbcNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        E, L = torch.nn.ELU, torch.nn.Linear
        self.estimator = torch.nn.Sequential(L(516, 256), E(), L(256, 256), E(), L(256, 35))
        self.actor = torch.nn.Sequential(L(121, 512), E(), L(512, 256), E(), L(256, 256), E(), L(256, 15))

    def forward(self, x):
        e = self.estimator(x)
        z = e[:, 3:]
        z = z / torch.linalg.vector_norm(z, dim=-1, keepdim=True).clamp_min(1e-12)
        return self.actor(torch.cat([x[:, -86:], e[:, :3], z], dim=1))


def convert(name):
    path = os.path.join(SRC, f"GR00T-WholeBodyControl-{name}.onnx")
    w = {i.name: torch.from_numpy(numpy_helper.to_array(i).copy()) for i in onnx.load(path).graph.initializer}
    net = Gr00tWbcNet()
    net.load_state_dict(w)
    net.eval()
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    x = np.random.default_rng(0).normal(size=(256, 516)).astype(np.float32)
    ref = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    with torch.no_grad():
        out = net(torch.from_numpy(x)).numpy()
    err = float(np.abs(out - ref).max())
    assert err < 1e-4, err
    os.makedirs(OUT, exist_ok=True)
    torch.jit.script(net).save(os.path.join(OUT, f"{name.lower()}.pt"))
    print(f"{name}: max |torch - onnx| = {err:.2e} -> {OUT}/{name.lower()}.pt")


if __name__ == "__main__":
    for n in ("Walk", "Balance"):
        convert(n)
