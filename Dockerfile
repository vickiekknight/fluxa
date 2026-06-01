# ============================================================
# Fluxa: Automated Sim-to-Real Robotics Pipeline
# ============================================================
#
# Prerequisites:
#   1. NVIDIA GPU with driver >= 535.129.03
#   2. NGC API key: https://ngc.nvidia.com/setup/api-key
#   3. Login to NGC registry:
#        docker login nvcr.io -u '$oauthtoken' -p <YOUR_NGC_API_KEY>
#
# Build:
#   docker compose build
#
# Run:
#   docker compose up -d
#
# Check latest Isaac Sim tags:
#   https://catalog.ngc.nvidia.com/orgs/nvidia/containers/isaac-sim
# ============================================================

FROM nvcr.io/nvidia/isaac-sim:4.5.0

# ── Environment ───────────────────────────────────────────────
ENV ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all \
    OMNI_SERVER=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0 \
    DEBIAN_FRONTEND=noninteractive \
    TERM=xterm

# ── System dependencies ───────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    net-tools \
    lsof \
    pciutils \
    unzip \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── IsaacLab v2.3.2 ──────────────────────────────────────────
# Installs: isaaclab, isaaclab_assets, isaaclab_tasks,
#           isaaclab_rl, isaaclab_mimic, isaaclab_contrib
RUN git clone https://github.com/isaac-sim/IsaacLab.git /isaac-sim/IsaacLab \
    && cd /isaac-sim/IsaacLab \
    && git checkout e10c302cd24 \
    && ln -sf /isaac-sim /isaac-sim/IsaacLab/_isaac_sim \
    && /isaac-sim/IsaacLab/isaaclab.sh --install

# ── CuRobo ───────────────────────────────────────────────────
# GPU-accelerated motion generation for robot manipulation
RUN git clone https://github.com/NVlabs/curobo.git /isaac-sim/curobo \
    && cd /isaac-sim/curobo \
    && git checkout 8726021 \
    && /isaac-sim/python.sh -m pip install --no-cache-dir tomli \
    && /isaac-sim/python.sh -m pip install --no-cache-dir -e ".[dev]" \
       --no-build-isolation

# ── Fluxa Python dependencies ─────────────────────────────────
COPY requirements.txt /tmp/fluxa_requirements.txt
RUN /isaac-sim/python.sh -m pip install --no-cache-dir \
    -r /tmp/fluxa_requirements.txt

# ── Fluxa source ─────────────────────────────────────────────
COPY . /isaac-sim/fluxa-agent-pack

# ── Shell aliases ─────────────────────────────────────────────
RUN echo 'alias runapp=/isaac-sim/runapp.sh' >> ~/.bashrc \
    && echo 'alias runheadless=/isaac-sim/runheadless.sh' >> ~/.bashrc \
    && echo 'alias python=/isaac-sim/python.sh' >> ~/.bashrc

WORKDIR /isaac-sim
EXPOSE 8765/tcp
CMD ["bash"]