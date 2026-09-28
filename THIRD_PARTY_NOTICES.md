# Third-party notices

The code written for this repository is licensed under Apache-2.0 (`LICENSE`). The items below come from, or are
derived from, other projects and keep their own licenses.

| What | Where in this repo | Origin | License |
|---|---|---|---|
| Policy checkpoints you train from the G1DWAQ_Lab `g1_dwaq` model | `checkpoints/*.pt` (none released yet) | [G1DWAQ_Lab](https://github.com/liuyufei-nubot/G1DWAQ_Lab) | BSD-3-Clause (text below) |
| PPO + β-VAE update adapted from the TienKung-Lab `rsl_rl` fork's `DWAQPPO.update` | `g1_rl/distill.py` | G1DWAQ_Lab / TienKung-Lab / RSL-RL | BSD-3-Clause (text below) |
| Staircase geometry from `stairs_scene.xml` | `scenes/g1_stairs_scene.xml` | G1DWAQ_Lab | BSD-3-Clause (text below) |
| G1DWAQ_Lab itself (training code, robot model, pretrained policy) | `G1DWAQ_Lab/` (git submodule, not copied) | G1DWAQ_Lab | BSD-3-Clause |
| WBC-AGILE G1 velocity-height policy (TorchScript, ONNX, I/O spec) and two config files | `skills/imported/agile/` | [nvidia-isaac/WBC-AGILE](https://github.com/nvidia-isaac/WBC-AGILE) at the commit in `skills/imported/agile/COMMIT` | Apache-2.0 (+ BSD-3 portions), `skills/imported/agile/LICENCE` |
| Isaac Sim / Isaac Lab | pulled by `docker/Dockerfile.train` and the `render` service at build/run time, not redistributed | NVIDIA | Isaac Sim: NVIDIA license (accepted via `ACCEPT_EULA=Y`); Isaac Lab: BSD-3-Clause |
| MuJoCo, PyTorch, imageio, NumPy | installed from PyPI at build time | their projects | Apache-2.0 / BSD-style |

## G1DWAQ_Lab license (BSD-3-Clause)

```
BSD 3-Clause License

Copyright (c) 2021-2024, The RSL-RL Project Developers.
All rights reserved.
Original code is licensed under BSD-3-Clause.

Copyright (c) 2022-2025, The Isaac Lab Project Developers.
All rights reserved.
Original code is licensed under BSD-3-Clause.

Copyright (c) 2025-2026, The Legged Lab Project Developers.
All rights reserved.
Modifications are licensed under BSD-3-Clause.

Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
All rights reserved.
Modifications are licensed under BSD-3-Clause.

Copyright (c) 2025-2026, The G1 DWAQ Blind Stair Climbing Project Developers.
All rights reserved.
Modifications are licensed under BSD-3-Clause.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
