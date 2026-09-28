# G1 stairs: combining pretrained Unitree G1 policies

Tools for taking pretrained whole-body policies for the **Unitree G1 (29 DoF)** from different repositories, running
them on one robot model, and training a single controller from several of them. Training runs in
**Isaac Sim 5.1 / Isaac Lab 2.3**, policies can be checked in **MuJoCo**, and videos are rendered with
**Isaac Sim 6.0**. Everything runs in Docker.

What is here:

- **Stair climbing out of the box**: the pretrained blind stair policy of
  [G1DWAQ_Lab](https://github.com/liuyufei-nubot/G1DWAQ_Lab) (DreamWaQ, deployed on a real G1 by its authors), with a
  MuJoCo demo and a robustness sweep.
- **A skill layer** (`skills/`): pretrained policies from other repositories run behind one interface, each tested
  against its original. Currently G1DWAQ_Lab's stair policy and NVIDIA
  [WBC-AGILE](https://github.com/nvidia-isaac/WBC-AGILE)'s walk-and-crouch policy.
- **Multi-teacher distillation** (`g1_body` task): one lower-body controller trained with PPO plus supervision from
  whichever teacher is valid in each situation, while the upper body stays free for another skill.

Everything here is simulation only. Nothing in this repository has been run on a real robot.

## Status

| Piece | State |
|---|---|
| G1DWAQ_Lab stair policy in MuJoCo | works: 10 steps up + down, 12/12 crossings at 0.9 m/s over 12 start poses |
| Skill adapters | DWAQ adapter reproduces the trained policy exactly; AGILE adapter matches NVIDIA's ONNX export to 1.7e-4 |
| AGILE in our Isaac Lab env | walks within 0.03 m/s of the command, no falls with the arms clear of the thighs; crouches only to ~0.62 m pelvis height (asked 0.50) |
| `g1_body` distillation | code runs; no released checkpoint. A first run (with an earlier teacher set) kept stair climbing but did not learn AGILE's crouching |

## Quick start

Requirements: Linux, Docker with the NVIDIA Container Toolkit, an RTX GPU (developed on an RTX A2000 12 GB with driver
595). Pulling `nvcr.io/nvidia/isaac-sim` may need `docker login nvcr.io`.

```bash
git clone --recursive https://github.com/MHC-CodeSmith/g1_stairs.git
cd g1_stairs
docker compose build record                  # MuJoCo demo image (~1.5 GB)
docker compose build train                   # Isaac Sim 5.1 + Isaac Lab 2.3.2 + TienKung-Lab (~18 GB)
docker pull nvcr.io/nvidia/isaac-sim:6.0.0   # only for rendering Isaac videos (~21 GB)
```

GPU assignments in `docker-compose.yaml` (`NVIDIA_VISIBLE_DEVICES`) match the machine this was built on. Change them
if you have a single GPU.

**MuJoCo demo**: the upstream stair policy walks 10 steps up, across a platform and down, driven by an autopilot.

```bash
docker compose run --rm record               # -> output/g1_stairs.mp4
xhost +local: ; docker compose up viewer     # live MuJoCo window
```

**Check the imported skills** (one check per process):

```bash
docker run --rm -v $PWD:/w -w /w --entrypoint sh g1-stairs:latest -c \
    "pip install -q onnxruntime==1.20.1 && python tests/test_agile_adapter.py"          # AGILE adapter vs ONNX
docker compose run --rm train skills/validate.py --check dwaq_equivalence              # DWAQ adapter vs policy
docker compose run --rm train skills/validate.py --check agile_native                  # AGILE walking/crouching
docker compose run --rm train skills/validate.py --check agile_remap                   # AGILE on our PD gains
```

**Train the combined controller** (headless, 4096 robots, about 6 s per iteration on an RTX A2000):

```bash
docker compose run --rm train g1_rl/train.py --headless --task g1_body --num_envs 4096 --max_iterations 1500 \
    --init_checkpoint /workspace/TienKung-Lab/logs/g1_dwaq/2026-01-16_00-46-00/model_9999.pt
```

Logs and checkpoints go to `logs/<task>/<date>_<run>/`; `tools/export_checkpoint.py` strips the optimizer state and
records provenance for release.

**Isaac Sim video** (physics in Isaac Sim 5.1, rendering in Isaac Sim 6.0):

```bash
g1_rl/make_isaac_video.sh G1DWAQ_Lab/TienKung-Lab/logs/g1_dwaq/2026-01-16_00-46-00/model_9999.pt upstream --task g1_dwaq
g1_rl/make_isaac_video.sh logs/g1_body/<run>/model_1499.pt body --task g1_body --upper random   # default, hold, random
```

Isaac Sim 5.1's RTX renderer crashed on driver 595 here (the stock NVIDIA image too), so `rollout_isaac.py` records
every link's pose from a headless 5.1 run and `render_isaac6.py` replays them in Isaac Sim 6.0.

## How it works

**Base policy.** DreamWaQ ("DWAQ"): an encoder reads the last 5 frames of the robot's own sensor readings and estimates
its velocity plus a 16-number terrain summary; the actor never sees the steps. The critic, used only in training, also
gets terrain heights and foot contacts. The policy runs at 50 Hz, physics at 200 Hz, and outputs joint targets for PD
motors.

**Skill layer** (`skills/`). `adapters.py` rebuilds each imported policy's observation (term order, joint order,
defaults, scales, history) and action decoding, so every skill takes the same robot state and returns joint targets plus
the PD gains it was trained with. `gain_equivalent_target` converts a target between gain sets, so a skill trained with
other motor gains can still drive, or label, our robot. `registry.yaml` lists every skill with its source, license,
joints, command and what it is good at. `validate.py` holds the go/no-go checks each imported skill must pass before
it is used as a teacher.

**Distillation** (`g1_rl/body.py`, `g1_rl/distill.py`). The student drives the legs and waist pitch from a command
(vx, vy, yaw rate, pelvis height); a motion library drives the upper body (default pose, fixed holds, random reaches)
so the student learns to balance under whatever an upper-body skill does. For each robot the teacher is NVIDIA AGILE
when standing on a flat tile with a height command, and the G1DWAQ_Lab stair policy everywhere else. The loss is PPO
plus a per-joint imitation term toward the active teacher, labelled on the student's own states and faded from weight
1.0 to 0.05 so the task reward takes over. `g1_rl/warmstart.py` maps a checkpoint onto a new observation layout (the
height command starts with zero weights). The approach follows HANDOFF
([arXiv 2606.06493](https://arxiv.org/abs/2606.06493)), in a much smaller form.

## Known limitations

- The first `g1_body` run did not learn to crouch: only about 4.5% of robots were in the crouch context and the
  imitation weight faded too early. Planned: balanced context sampling and a fixed imitation weight per teacher until
  the student matches it.
- All policies are blind: step height is felt, not seen.
- Only locomotion skills so far; no manipulation policy is imported yet.
- The MuJoCo runner supports the upstream observation only (no height command).
- Not tested on hardware.

## Repository layout

| Path | Contents |
|---|---|
| `skills/` | skill adapters, registry, Isaac Lab runtime, validation checks, imported AGILE policy |
| `g1_rl/` | `g1_body` task, distillation, training, warm start, rollout and render scripts |
| `tools/` | checkpoint export |
| `tests/` | AGILE adapter test against NVIDIA's ONNX export |
| `run_stairs.py`, `eval_sweep.py`, `scenes/` | MuJoCo demo, robustness sweep, staircase scene |
| `docker/`, `docker-compose.yaml` | images: MuJoCo demo, Isaac Sim 5.1 training, Isaac Sim 6.0 render |
| `G1DWAQ_Lab/` | upstream training code and pretrained policy (git submodule) |

## License and credits

Code in this repository: Apache-2.0 (`LICENSE`). Policies fine-tuned from the G1DWAQ_Lab model fall under its
BSD-3-Clause license; the imported AGILE files keep NVIDIA's license. Details: `THIRD_PARTY_NOTICES.md`.

Built on [G1DWAQ_Lab](https://github.com/liuyufei-nubot/G1DWAQ_Lab), [TienKung-Lab](https://github.com/Open-X-Humanoid/TienKung-Lab),
[Legged Lab](https://github.com/Hellod035/LeggedLab), [RSL-RL](https://github.com/leggedrobotics/rsl_rl),
[Isaac Lab](https://github.com/isaac-sim/IsaacLab), [WBC-AGILE](https://github.com/nvidia-isaac/WBC-AGILE) and
[MuJoCo](https://github.com/google-deepmind/mujoco).
