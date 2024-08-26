FROM nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04

RUN apt-get update && \
    apt-get install -y \
        git \
        vim \
        htop \
        python3 \
        python3-pip && \
    rm -rf /var/lib/apt/lists/*

ARG USER_ID
ARG GROUP_ID
ARG NAME
RUN groupadd --gid ${GROUP_ID} ${NAME}
RUN useradd \
    --no-log-init \
    --create-home \
    --uid ${USER_ID} \
    --gid ${GROUP_ID} \
    -s /bin/sh ${NAME}

ARG WORKDIR_PATH
WORKDIR ${WORKDIR_PATH}

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install "jaxlib==0.3.0+cuda11.cudnn82" \
        -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN python3 -m pip install "jax==0.3.0"
COPY requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt
