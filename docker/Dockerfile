ARG PYTORCH="1.10.0"
ARG CUDA="11.4"
ARG CUDNN="8"
ARG MMCV="1.7.1"
ARG MMDET="2.28.2"
ARG MMSEG="0.20.0"
FROM tacc/tacc-ml:ubuntu20.04-cuda11-tf2.6-pt1.10
#FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# To fix GPG key error when running apt-get update
RUN apt-get update && apt-get install -y gnupg
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMCV, MMDetection and MMSegmentation
ARG PYTORCH
ARG CUDA
ARG MMCV
ARG MMDET
ARG MMSEG
RUN pip install openmim
#RUN ["/bin/bash", "-c", "mim install mmcv-full==${MMCV} -f https://download.openmmlab.com/mmcv/dist/cu${CUDA//./}/torch${PYTORCH}/index.html"]
RUN mim install mmcv-full mmdet mmsegmentation

# Install MMDetection3D
RUN conda clean --all
COPY . /mmdetection3d
WORKDIR /mmdetection3d
ENV FORCE_CUDA="1"
RUN pip install -r requirements/build.txt
RUN pip install -r requirements/optional.txt
RUN pip install -r requirements/runtime.txt
RUN pip install cumm-cu114 spconv-cu114
RUN pip install -v -e .

# Fix fraction gcd to math deprecation: https://github.com/smicallef/spiderfoot/issues/1124
RUN pip install networkx==2.5
RUN pip install tensorflow==2.6.0
RUN pip install protobuf==3.20.0

