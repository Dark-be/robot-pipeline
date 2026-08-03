FROM ubuntu:22.04
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
RUN dpkg -i cuda-keyring_1.1-1_all.deb

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

RUN apt-get update && apt-get install -y \
can-utils \
iproute2 \
python3-pip \
libgoogle-glog-dev \
libnlopt-dev \
libnlopt-cxx-dev \
cuda-toolkit-12-6 \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
cudnn9-cuda-12 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /root/prj/robot-pipeline

CMD ["/bin/bash"]
