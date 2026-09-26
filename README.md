# Unitree G1 stair climbing (MuJoCo, containerized)

Runs the pretrained **DWAQ blind-locomotion policy** for the real **G1 29-DoF** from
[G1DWAQ_Lab](https://github.com/liuyufei-nubot/G1DWAQ_Lab) (trained in Isaac Lab 2.3 on 0–23 cm pyramid stairs and
deployed on hardware by its authors), in MuJoCo sim2sim, on a flight of 10 steps up (0.15 m rise / 0.31 m tread),
a platform at 1.65 m, and 10 steps down.

```
docker compose build                      # image g1-stairs:latest (~1.5 GB, CPU torch + MuJoCo)
docker compose run --rm record            # headless, EGL on GPU 1 -> output/g1_stairs.mp4 + pass/fail summary
docker compose up viewer                  # live MuJoCo viewer on $DISPLAY (default :1), loops episodes
docker compose run --rm record --vx 0.6 --duration 25 --out /output/slow.mp4   # any run_stairs.py flag
```

| file | what |
|---|---|
| `run_stairs.py` | reuses upstream `G1DwaqMujocoRunner` unchanged (policy, obs, PD gains, 200 Hz sim / 50 Hz control); adds autopilot, MP4 recording, metrics |
| `scenes/g1_stairs_scene.xml` | upstream stairs geometry; the upstream `stairs_scene.xml` does not load (`repeated name 'floor'`: the robot XML already defines floor/light) |
| `eval_sweep.py` | robustness sweep over speed × start offset × start yaw |
| `docker/Dockerfile`, `docker-compose.yaml` | container; only the needed upstream files are copied in |

Autopilot: forward `vx`, heading held at 0 with yaw rate, lateral drift corrected through `vy`. Steering with yaw
instead (heading-toward-centerline) caused mid-flight spins and only ~7/12 full crossings.

Sweep (12 starts per speed, `eval_sweep.py`):

| vx [m/s] | reached top | full up+down | clean (no backtrack) |
|---|---|---|---|
| 0.6 | 11/12 | 11/12 | 7/12 |
| 0.8 | 11/12 | 11/12 | 11/12 |
| 0.9 (default) | 12/12 | 11/12 | 11/12 |
| 1.0 | 12/12 | 12/12 | 11/12 |
