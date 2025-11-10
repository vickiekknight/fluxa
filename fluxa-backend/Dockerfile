# Base Ubuntu 22.04 image
FROM ubuntu:22.04

# Build metadata labels
LABEL org.opencontainers.image.ref.name=ubuntu
LABEL org.opencontainers.image.version=22.04
LABEL com.nvidia.omniverse.service=ov-base-ubuntu-22
LABEL com.nvidia.omniverse.build.release=2025.2.0
LABEL com.nvidia.omniverse.build.family=local
LABEL com.nvidia.omniverse.build.branch=tag-2025.2.0
LABEL com.nvidia.omniverse.build.build=d45fe3be
LABEL com.nvidia.omniverse.build.build_tool_version=1.3.0
LABEL com.nvidia.omniverse.build.compliant=0

# NVIDIA GPU configuration
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

# Prevent hwcap from interfering with library loading
RUN touch /etc/ld.so.nohwcap

# Install base dependencies
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl \
        libatomic1 \
        libegl1 \
        libgl1 \
        libglu1-mesa \
        libglx0 \
        libgomp1 \
        libsm6 \
        libxi6 \
        libxrandr2 \
        libxt6 \
        unzip \
    && apt-get -y autoremove \
    && apt-get clean autoclean \
    && rm -rf /var/lib/apt/lists/*

# Configure GPU vendor files
COPY data/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json
COPY data/50_mesa.json /usr/share/glvnd/egl_vendor.d/50_mesa.json
COPY data/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json
COPY data/nvidia_layers.json /etc/vulkan/implicit_layer.d/nvidia_layers.json

ENV VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json

# Create omniverse directory structure
RUN mkdir -p /opt/nvidia/omniverse/

# Create ubuntu user
RUN useradd --create-home --shell /bin/bash ubuntu \
    && chown -R ubuntu:ubuntu /opt/nvidia

USER ubuntu
WORKDIR /home/ubuntu

# EULA script
COPY data/eula.sh /eula.sh
ENTRYPOINT ["/eula.sh"]

# === Isaac Sim specific layers ===

LABEL com.nvidia.omniverse.service=isaac-sim
LABEL com.nvidia.omniverse.build.release=5.0.0-rc.45
LABEL com.nvidia.omniverse.build.family=gl
LABEL com.nvidia.omniverse.build.branch=release-5.0
LABEL com.nvidia.omniverse.build.build=184afb15
LABEL com.nvidia.omniverse.build.build_tool_version=1.3.0
LABEL com.nvidia.omniverse.build.compliant=0

USER root

# Create Isaac Sim directory
RUN mkdir -p /isaac-sim
WORKDIR /isaac-sim

# Copy Isaac Sim installation files
# NOTE: You'll need to obtain these from NVIDIA Isaac Sim installation
COPY _inputs/isaac-sim /isaac-sim/

# Copy license and privacy scripts
COPY data/license.sh /isaac-sim/
COPY data/privacy.sh /isaac-sim/

# Copy open source licenses and source
RUN mkdir -p /isaac-sim/oss/source
COPY oss/source/isaac-sim /isaac-sim/oss/source
COPY oss/license/isaac-sim /isaac-sim/oss/license

# Restore canonical sources list
RUN mv /etc/apt/sources.list.canonical /etc/apt/sources.list

# Install additional dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libnghttp2-14 \
    && rm -rf /var/lib/apt/lists/*

# Create nvidia-omniverse config directory
RUN mkdir -p /root/.nvidia-omniverse/config

# Expose ports for streaming and networking
EXPOSE 49100/tcp 47998/udp

# Remove pip configuration
RUN rm /root/.pip/pip.conf

# Environment variables
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV OMNI_SERVER=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0
ENV MIN_DRIVER_VERSION=535.129.03

WORKDIR /isaac-sim

# Create symlink for extension examples
RUN ln -s exts/isaacsim.examples.interactive/isaacsim/examples/interactive extension_examples

# Create runapp.sh script
RUN echo '#! /bin/sh' > runapp.sh \
    && echo '/isaac-sim/license.sh && /isaac-sim/privacy.sh && /isaac-sim/isaac-sim.sh \\' >> runapp.sh \
    && echo '  --/persistent/isaac/asset_root/default="$OMNI_SERVER" \\' >> runapp.sh \
    && echo '  --merge-config="/isaac-sim/config/open_endpoint.toml" --allow-root "$@" ' >> runapp.sh \
    && chmod u+x runapp.sh

# Create runheadless.sh script
RUN echo '#! /bin/sh' > runheadless.sh \
    && echo '/isaac-sim/license.sh && /isaac-sim/privacy.sh && /isaac-sim/isaac-sim.streaming.sh \\' >> runheadless.sh \
    && echo '  --/persistent/isaac/asset_root/default="$OMNI_SERVER" \\' >> runheadless.sh \
    && echo '  --merge-config="/isaac-sim/config/open_endpoint.toml" --allow-root "$@" ' >> runheadless.sh \
    && chmod u+x runheadless.sh

# Add aliases to bashrc
RUN echo 'alias runapp=/isaac-sim/runapp.sh' >> ~/.bashrc \
    && echo 'alias runheadless=/isaac-sim/runheadless.sh' >> ~/.bashrc

# Set default entrypoint to run headless
ENTRYPOINT ["/bin/sh", "-c", "/isaac-sim/runheadless.sh"]

# === Your custom layer (1.05GB) ===
# Add your custom modifications here
# This represents the 12-day-old layer at the top
