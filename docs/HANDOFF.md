# Handoff notes (2026-09-28)

## Goal
One efficient G1 "foundation" controller built from many pretrained policies, not trained from scratch. The user wants
the G1 29-DoF **with Dex3 hands**, as many repos combined as possible, and a path to language/vision (VLA) on top.

## Decided
- **Dance work is dropped** (removed in `1ab1f67`; old runs still in local `logs/`).
- **Architecture** (matches DiT4DiT, GR00T N1.7 + SONIC, WholeBodyVLA, LATENT; see research below):
  - L0 robot / sim.
  - L1 **our foundation controller**, 50 Hz, two interfaces: commands (vx, vy, wz, pelvis height, torso rpy + upper-body
    joint targets) and latent motion tokens.
  - L2 motion generators (SONIC planner, FRoM-W1 H-GPT text→motion).
  - L3 VLA / VAM, 3-10 Hz (GR00T N1.7, DiT4DiT), cloud GPU.
  - L4 LLM task planner over `skills/registry.yaml`.
- **L1 recipe = LATENT's**: distill the scorecard winners per context into a conditional VAE (state-dependent prior;
  decoder = controller, latent = motion tokens). Wrists and hands are excluded from the latent and commanded directly.
  A command front-end maps velocity/height commands to latents. No RT-2-style discretized action tokens (nobody uses
  them for G1 whole body).
- **Hardware split**: 2x RTX A2000 12 GB here. Low-level policies, the arena and Isaac training run here. GR00T N1.7
  (inference 16 GB+, fine-tune 40 GB+), DiT4DiT training (8+ GPUs) and H-GPT fp16 (8B) need the cloud VM.
- Commits as **MHC-CodeSmith** (see memory). Repo: https://github.com/MHC-CodeSmith/g1_stairs (public, Apache-2.0).

## Done
- `skills/`: adapters for G1DWAQ (DWAQ), NVIDIA WBC-AGILE and GR00T WBC (`Gr00tWbc`, TorchScript from
  `tools/convert_gr00t_wbc.py`), each verified against its original (`tests/test_{agile,gr00t}_adapter.py`,
  `skills/validate.py`). In our Isaac env AGILE bottoms out at ~0.62 m; GR00T WBC crouches to 0.47 m (0.024 m error).
- `arena/` (MuJoCo 3.6, image `g1-arena`, CPU): G1 29-DoF + Dex3 (43 motors), flat / stairs / rough.
  - Locomotion adapters (`arena/policies.py` `REGISTRY`): unitree_rl_gym, mujoco_playground, g1_walk37_{baseline,robust},
    gr00t_wbc, holosoma_{fastsac,ppo}, g1dwaq_stairs, agile_vel_height, sonic (planner-driven), g1_body (ours).
  - Trackers (`TRACKERS`, `arena/trackers.py`, `arena/sonic.py`): sonic_tracking, gmt, twist; `arena/track.py`.
  - `arena/check_adapters.py`: each repo's own control code on its own model, **9/9 PASS** (max diff ≤ 7.4e-6).
    SONIC has no Python reference (C++ only); it is validated by tracking quality.
  - `arena/scorecard.py` → `docs/SCORECARD.md` (7 locomotion tests, 9 tracking clips).
  - Gotchas: recompute derived state after `mj_step`; `mjOBJ_XBODY` not `mjOBJ_BODY`; playground lin vel at the
    `imu_in_pelvis` site; holosoma sorts obs terms; GMT/TWIST read MuJoCo `sensordata` (one physics step behind qpos).
- `g1_rl/` task `g1_body` (Isaac Lab 2.3): student DWAQ (legs + waist pitch, upper body free), teachers picked from
  what the student observes: height command below nominal → GR00T WBC (fixed imitation weight, ~16% of envs),
  otherwise G1DWAQ (annealed weight). Run `logs/g1_body/2026-09-28_22-14-57` (1500 it, from the upstream DWAQ).
  - Released `checkpoints/g1_body.pt` = **iteration 700**: crosses the stairs, crouches to 0.52 m with 0.003 m error,
    survives 1000 N pushes, flat vx err 0.049. Later iterations lose the stairs in MuJoCo (1499 falls at 5 s) as the
    DWAQ imitation weight anneals → hold that weight (or anneal much later) in the next run.
- `third_party.yaml` + `tools/fetch_third_party.sh`: 9 pinned repos + HF `nvidia/GEAR-SONIC`, `LeCAR-Lab/BFM-Zero`.
  License flags: BFM-Zero CC-BY-NC; SONIC and GR00T WBC weights NVIDIA Open Model License. GR00T files come from LFS.
  `third_party/mujoco_playground/mujoco_menagerie` must be a symlink to `../mujoco_menagerie`.

## Scorecard summary (docs/SCORECARD.md)
- Velocity tracking: AGILE best (0.016 m/s), then G1DWAQ, GR00T WBC, g1_body. Only holosoma, AGILE and SONIC survive rough.
- Stairs: only G1DWAQ and g1_body cross. Crouch: g1_body (0.52 m, 0.003 m err) and GR00T WBC (0.50 m, 0.012 m).
- Pushes: G1DWAQ, SONIC and g1_body survive 1000 N. g1_walk37_* fall on flat; they need their own robot model.
- Tracking: SONIC best overall and the only one that holds heading on long clips (GMT drifts 3 m on a 38 s walk).

## Arena report (2026-09-29)
- `docs/ARENA_REPORT.md` (built by `tools/build_report.py` from `output/*.json`; GIFs by `tools/make_media.py`, charts by
  `tools/report_figures.py`). New: Safe100 (`arena/safe100.py`, checked in its own mjlab env with
  `tools/check_safe100.py` / `docker/Dockerfile.mjlab`, needs PYTORCH_JIT=0), unitree_rl_lab (`arena/unitree_lab.py`),
  HumanoidBench stair course + step sweep + Safe100 staircase (`arena/stairs_bench.py`).
- Stairs: only G1DWAQ (up to 22 cm) and g1_body (up to 20 cm, best HumanoidBench return 622) climb. Safe100 works only
  on MuJoCo-Warp (falls on CPU MuJoCo even with its own compiled model).
- GRAIL (NVlabs): terrain tracker = SONIC fine-tune + 11x11 height map + object obs; config in third_party/hf/GRAIL;
  port not done. Disk on this machine is ~full (5 GB free): the GRAIL clone (4.9 GB) was deleted.

## Open (in order)
1. **Next g1_body run**: keep the DWAQ imitation weight high longer (the stairs fade after ~700 it); add rough-terrain
   robustness (student falls on rough at 15 s, like its teachers) — holosoma/AGILE could teach rough.
2. **L1 with the LATENT recipe** (task 9): CVAE over SONIC-tracked motions + g1_body, command front-end → latents,
   wrists and hands commanded directly.
3. **Cloud**: GR00T N1.7 fine-tune with tag `UNITREE_G1` (decoupled WBC) against L1, data teleoperated in the arena.
   Dex3 fingers need teleop data; no pretrained policy moves them.

## Research (checked 2026-09-28)
- **DiT4DiT** (Mondo-Robotics/DiT4DiT, MIT): Cosmos-Predict2.5-2B features → DiT action head, 36-D joint-position
  chunks of 16. G1 whole body via decoupled WBC + ALOHA gripper, ~6 Hz. No G1 weights released.
- **WholeBodyVLA**: no code; AgiBot X2, not G1.
- **FRoM-W1** (OpenMOSS, Apache-2.0): H-GPT (Llama-3.1-8B LoRA, text→motion) + H-ACT OmniH2O G1 trackers on HF
  `OpenMOSS-Team/FRoM-W1`.
- **LATENT** (GalaxyGeneralRobotics, no license): only stage 1 (tracker) released.
- **GR00T N1.7** (Apache-2.0, gated Cosmos-Reason2-2B backbone): tags `UNITREE_G1` (decoupled WBC) and
  `UNITREE_G1_SONIC` (latent tokens). Real G1 walk-and-pick benchmark: 159/216 = 73.6%, UMI gripper.

## Commands
```bash
docker build -t g1-arena -f docker/Dockerfile.arena docker
docker run --rm -v $PWD:/workspace/g1_stairs g1-arena -m arena.check_adapters
docker run --rm -v $PWD:/workspace/g1_stairs g1-arena -m arena.run --policy all --terrain flat
docker run --rm -v $PWD:/workspace/g1_stairs g1-arena -m arena.scorecard --workers 14      # ~15 min, CPU
docker run --rm -v $PWD:/workspace/g1_stairs g1-arena -m arena.track --clip all
docker compose run --rm train g1_rl/train.py --headless --task g1_body ...   # Isaac, image g1-isaaclab
```
Note: Isaac Sim 5.1's renderer crashes on driver 595; video is rendered with Isaac Sim 6.0 (`g1_rl/render_isaac6.py`).
If the other lab jobs fill the GPUs, Isaac fails with "Failed to get DOF velocities" or CUDA OOM and the container
hangs: train on one GPU (`-e NVIDIA_VISIBLE_DEVICES=1`, the user's preference) and check it is free first.
