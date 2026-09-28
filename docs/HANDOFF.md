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
- `skills/`: adapters for G1DWAQ (DWAQ) and NVIDIA WBC-AGILE, each verified against its original
  (`tests/test_agile_adapter.py`, `skills/validate.py`). AGILE crouch bottoms out at ~0.62 m pelvis height (asked 0.50).
- `g1_rl/`: Isaac Lab 2.3 (Isaac Sim 5.1) multi-teacher distillation task `g1_body`. First run
  (`logs/g1_body/2026-09-28_04-39-19_body`, trained with the old dance teacher) kept the stairs skill but **ignored the
  height command** (0.10 m error; only 4.5% of envs were in the crouch context; imitation weight faded too early).
  Its checkpoint is not released.
- `third_party.yaml` + `tools/fetch_third_party.sh`: 9 pinned repos + HF `nvidia/GEAR-SONIC`, `LeCAR-Lab/BFM-Zero`
  into gitignored `third_party/`.
  - License flags: BFM-Zero is CC-BY-NC; SONIC and GR00T WBC weights are under NVIDIA's Open Model License.
  - GR00T meshes and ONNX come from LFS.
  - `third_party/mujoco_playground/mujoco_menagerie` is a symlink to `../mujoco_menagerie`; playground needs it.
- `arena/` (MuJoCo 3.6, image `g1-arena`, CPU): Unitree G1 29-DoF + Dex3 model (43 motors), flat / stairs / rough.
  - Adapters in `arena/policies.py` (`REGISTRY`): unitree_rl_gym, mujoco_playground, g1_walk37_{baseline,robust},
    gr00t_wbc, holosoma_{fastsac,ppo}, g1dwaq_stairs, agile_vel_height.
  - `arena/check_adapters.py` runs each repo's own control code on its own model: **all 7 PASS** (max diff ≤ 4e-6).
  - Gotchas that took time: recompute derived state after `mj_step`; use `mjOBJ_XBODY`, not `mjOBJ_BODY`;
    playground's lin vel is at the `imu_in_pelvis` site; holosoma sorts obs terms alphabetically.

## Open (in order)
1. **SONIC adapter**. Files in `third_party/hf/GEAR-SONIC/`:
   - encoder input 1762 → 64 tokens; decoder input 994 → 29 actions; planner 11 inputs → `mujoco_qpos [1,64,36]` at 30 Hz.
   - Semantics already read from
     `third_party/GR00T-WholeBodyControl/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/{src/g1_deploy_onnx_ref.cpp,include/policy_parameters.hpp,include/localmotion_kplanner.hpp}`
     and `docs/source/references/{observation_config,planner_onnx}.md`.
   - Decoder obs (`observation_config.yaml`): token 64 + 10-frame histories (step 1) of ang vel, q − default,
     qd, last actions, gravity.
   - History is stored in **IsaacLab joint order**. Action: `q = default + a[isaaclab_to_mujoco[i]] * g1_action_scale[i]`
     (all tables in `policy_parameters.hpp`).
   - kp = armature·(2π·10)²; kd = 2·2·armature·2π·10; ankles and waist roll/pitch use 2× kp.
   - Encoder mode `g1` = 0 (motion joints 10f step5, velocities, anchor orientation 6D).
   - Planner output is resampled 30 → 50 Hz, velocities by finite difference ×50, 8-frame blend on replan.
2. **Scorecard** (task 8): `python -m arena.run --policy all --terrain {flat,stairs,rough} --schedule {walk,stairs,stand}`;
   add pushes and moving arms. First flat walk (20 s): no falls except g1_walk37_robust. vx err:
   agile 0.017, g1dwaq 0.03, gr00t 0.042, holosoma_fastsac 0.10, playground 0.12, rl_gym 0.14.
3. **L1 student with the LATENT recipe** (task 9) in Isaac `g1_rl/` (reuse `distill.py`; fix crouch-context sampling).
4. **Cloud**: GR00T N1.7 fine-tune with tag `UNITREE_G1` (decoupled WBC) against L1, data teleoperated in the arena.
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
docker compose run --rm train g1_rl/train.py --headless --task g1_body ...   # Isaac, image g1-isaaclab
```
Note: Isaac Sim 5.1's renderer crashes on driver 595; video is rendered with Isaac Sim 6.0 (`g1_rl/render_isaac6.py`).
If the other lab jobs fill the GPUs, Isaac fails with "Failed to get DOF velocities".
