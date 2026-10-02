"""How long each policy takes per control step (ms), and how big each model is (params or ONNX bytes).

CPU timing, same arena used for the benchmark (no GPU): isolates model cost, not render cost.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from arena.policies import MJ29, REGISTRY, make  # noqa: E402
from arena.world import Arena, Command  # noqa: E402

N_WARMUP = 10
N_TIMED = 100


def model_size(pol):
    """Best-effort: total torch params, or ONNX file size, across whatever nets the adapter holds."""
    import torch

    seen, total_params, onnx_bytes = set(), 0, 0
    for attr in vars(pol).values():
        if isinstance(attr, torch.nn.Module) and id(attr) not in seen:
            seen.add(id(attr))
            total_params += sum(p.numel() for p in attr.parameters())
        # ORT InferenceSession tuples stored as (session, name) in several adapters
        if isinstance(attr, tuple) and attr and type(attr[0]).__name__ == "InferenceSession":
            try:
                onnx_bytes += os.path.getsize(attr[0]._model_path)
            except Exception:
                pass
    return total_params, onnx_bytes


def time_policy(name):
    arena = Arena("flat")
    pol = make(name)
    if hasattr(pol, "bind"):
        pol.bind(arena)
    arena.reset(0.80, {})
    st = arena.state()
    pol.reset(st)
    cmd = Command(vx=0.5)

    for _ in range(N_WARMUP):
        pol.act(st, cmd)
    t0 = time.perf_counter()
    for _ in range(N_TIMED):
        pol.act(st, cmd)
    dt_ms = (time.perf_counter() - t0) / N_TIMED * 1000
    params, onnx_b = model_size(pol)
    return dt_ms, params, onnx_b


if __name__ == "__main__":
    print(f"{'policy':<22} {'ms/step (CPU)':>14} {'params':>12} {'onnx (MB)':>10}")
    rows = []
    for name in REGISTRY:
        try:
            dt_ms, params, onnx_b = time_policy(name)
            rows.append((name, dt_ms, params, onnx_b))
            print(f"{name:<22} {dt_ms:>14.3f} {params:>12,} {onnx_b/1e6:>10.2f}")
        except Exception as e:
            print(f"{name:<22} ERROR: {e!r}")

    # Trackers need a clip; use the tiniest one available (sonic's sample walk) for a fair, short timing.
    print()
    from arena.track import clips, load_clip
    from arena.policies import TRACKERS, make_tracker

    clip = load_clip("sonic_walk")
    for name in TRACKERS:
        if name == "grail_terrain":
            continue  # needs GRAIL weights/clip plumbing; skip for this quick pass
        try:
            arena = Arena("flat")
            pol = make_tracker(name, clip)
            if hasattr(pol, "bind"):
                pol.bind(arena)
            arena.reset(0.80, {})
            st = arena.state()
            pol.reset(st)
            for _ in range(N_WARMUP):
                pol.act(st, None)
            t0 = time.perf_counter()
            for _ in range(N_TIMED):
                pol.act(st, None)
            dt_ms = (time.perf_counter() - t0) / N_TIMED * 1000
            params, onnx_b = model_size(pol)
            print(f"{name:<22} {dt_ms:>14.3f} {params:>12,} {onnx_b/1e6:>10.2f}")
        except Exception as e:
            print(f"{name:<22} ERROR: {e!r}")
