# cuda runtime base: torch/onnxruntime wheels ship their own cuda libs, the driver comes from the host
ARG CUDA_IMAGE=nvidia/cuda:12.8.1-runtime-ubuntu22.04
FROM ghcr.io/astral-sh/uv:latest AS uv
FROM ${CUDA_IMAGE}

ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=1000

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git git-lfs curl wget ca-certificates pkg-config sudo \
        libeigen3-dev libyaml-cpp-dev \
        python3 python3-dev python3-venv \
        libglvnd0 libgl1 libglx0 libegl1 libgles2 libglu1-mesa libosmesa6 mesa-utils \
        libglfw3 libx11-6 libxext6 libxrender1 libxrandr2 libxinerama1 libxcursor1 \
        libxi6 libxxf86vm1 libxkbcommon-x11-0 x11-apps \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /usr/local/bin/

RUN groupadd -o -g ${USER_GID} ${USERNAME} \
    && useradd -o -m -s /bin/bash -u ${USER_UID} -g ${USER_GID} ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && mkdir -p /opt/venvs /opt/uv /workspace \
    && chown -R ${USER_UID}:${USER_GID} /opt/venvs /opt/uv /workspace

ENV UV_CACHE_DIR=/opt/uv/cache \
    UV_LINK_MODE=copy

USER ${USERNAME}

RUN --mount=type=cache,target=/opt/uv/cache,uid=${USER_UID},gid=${USER_GID} \
    uv venv --python /usr/bin/python3.10 /opt/venvs/base \
    && uv pip install --python /opt/venvs/base/bin/python 'numpy<2' mujoco onnxruntime

ENV PATH=/opt/venvs/base/bin:${PATH}

WORKDIR /workspace
CMD ["sleep", "infinity"]
