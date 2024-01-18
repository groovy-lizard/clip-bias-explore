# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

RUN rm -f /etc/apt/sources.list.d/*.list

RUN apt-get update \
    && apt-get install -y build-essential \
    curl \
    ca-certificates \
    sudo \
    git \
    wget \
    bzip2 \
    unzip \
    libsdl1.2-dev \
    libsdl-image1.2-dev \
    libx11-6 \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    yasm \
    cmake \
    ffmpeg \
    nano \
    && rm -rf /var/lib/apt/lists/*


RUN mkdir /app
WORKDIR /app

#Create a non-root user and switch to it.
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
    && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

#All users can use /home/user as their home directory.
ENV HOME=/home/user
RUN mkdir $HOME/.cache $HOME/.config \
    && chmod -R 777 $HOME

# install micromamba
RUN curl -sL https://micro.mamba.pm/api/micromamba/linux-64/1.1.0 \
    | sudo tar -xvj -C /usr/local bin/micromamba

ENV MAMBA_EXE=/usr/local/bin/micromamba \
    MAMBA_ROOT_PREFIX=/home/user/micromamba \
    CONDA_PREFIX=/home/user/micromamba \
    PATH=/home/user/micromamba/bin:$PATH

# set up the base conda env by installing pytorch and friends.
COPY environment.yml /app/environment.yml

RUN micromamba create -qy -n base -f /app/environment.yml \
    && rm /app/environment.yml \
    && micromamba shell init --shell=bash --prefix="$MAMBA_ROOT_PREFIX" \
    && micromamba clean -qya

CMD [ "/bin/bash" ]
