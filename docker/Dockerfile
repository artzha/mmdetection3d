ARG PYTORCH="1.11.0"
ARG CUDA="11.7"
ARG CUDNN="8"
ARG MMCV="1.6.2"
ARG MMDET="2.28.1"
ARG MMSEG="0.30.0"
FROM tacc/tacc-ml:ubuntu20.04-cuda11-tf2.6-pt1.10
#FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel
RUN conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX 8.6"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# To fix GPG key error when running apt-get update
RUN apt-get update && apt-get install -y gnupg
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV MMCV_WITH_OPS=1
ENV FORCE_CUDA=1
ENV CUDA_HOME="${CONDA_PREFIX}"

# Install MMCV, MMDetection and MMSegmentation
ARG PYTORCH
ARG CUDA
ARG MMCV
ARG MMDET
ARG MMSEG
RUN pip install openmim
RUN ["/bin/bash", "-c", "mim install mmcv-full==${MMCV} -f https://download.openmmlab.com/mmcv/dist/cu${CUDA//./}/torch${PYTORCH}/index.html"]
RUN mim install mmdet mmsegmentation

# Install MMDetection3D
RUN conda clean --all
COPY . /mmdetection3d
WORKDIR /mmdetection3d
ENV FORCE_CUDA="1"
RUN pip install -r requirements/build.txt
RUN pip install -r requirements/optional.txt
RUN pip install -r requirements/runtime.txt
RUN pip install cumm-cu117 spconv-cu117
RUN pip install -v -e .

# Fix fraction gcd to math deprecation: https://github.com/smicallef/spiderfoot/issues/1124
RUN pip install networkx==2.5
RUN pip install tensorflow==2.5.0
RUN pip install protobuf==3.20.0

