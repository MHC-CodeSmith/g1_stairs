# Unitree G1 29-DoF stair climbing: MuJoCo sim2sim of the pretrained DWAQ blind-locomotion policy (G1DWAQ_Lab).
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
# GL stacks: EGL (NVIDIA GPU, headless), OSMesa (CPU fallback, headless), GLFW/X11 (interactive viewer)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libegl1 libgl1 libglvnd0 libgles2 libosmesa6 libglfw3 \
        libx11-6 libxext6 libxrandr2 libxinerama1 libxcursor1 libxi6 libxkbcommon0 libxtst6 \
    && rm -rf /var/lib/apt/lists/*
# Point glvnd at the NVIDIA EGL vendor lib that the nvidia container runtime injects
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    printf '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}\n' \
        > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install mujoco==3.3.0 "numpy<2.3" imageio==2.37.0 imageio-ffmpeg==0.6.0

# pynput (upstream keyboard teleop) is omitted: run_stairs.py drives the robot with an autopilot and stubs it.
# Only what the sim2sim needs from G1DWAQ_Lab (policy checkpoint, MJCF + meshes, upstream runner)
WORKDIR /opt/TienKung-Lab
COPY G1DWAQ_Lab/TienKung-Lab/legged_lab/scripts/sim2sim_g1_dwaq.py legged_lab/scripts/
COPY G1DWAQ_Lab/TienKung-Lab/legged_lab/assets/unitree/g1/mjcf legged_lab/assets/unitree/g1/mjcf
COPY G1DWAQ_Lab/TienKung-Lab/logs/g1_dwaq/2026-01-16_00-46-00/model_9999.pt logs/g1_dwaq/2026-01-16_00-46-00/
COPY scenes/g1_stairs_scene.xml legged_lab/assets/unitree/g1/mjcf/
COPY run_stairs.py /opt/run_stairs.py

ENV TIENKUNG_ROOT=/opt/TienKung-Lab MUJOCO_GL=egl
ENTRYPOINT ["python", "/opt/run_stairs.py"]
CMD ["--mode", "record", "--out", "/output/g1_stairs.mp4"]
