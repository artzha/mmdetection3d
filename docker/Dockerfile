# FROM tacc/tacc-ml:ubuntu20.04-cuda11-tf2.6-pt1.10

# ARG PYTORCH="1.10.1"
# ARG CUDA="11.6"
# ARG CUDNN="8"
# ARG MMCV="1.7.0"
# ARG MMDET="2.28.1"
# ARG MMSEG="0.30.0"
# #FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel
# RUN conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia

# # RUN touch testfile
# # RUN /bin/bash -c echo $CUDA_HOME
# ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX 8.0"
# ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
# ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# # To fix GPG key error when running apt-get update
# RUN apt-get update && apt-get install -y gnupg
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub

# RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

# ENV MMCV_CUDA_ARGS='-gencode=arch=compute_80,code=sm_80'
# ENV MMCV_WITH_OPS=1
# ENV FORCE_CUDA=1
# ENV CUDA_HOME=$CONDA_PREFIX

# # Install MMCV, MMDetection and MMSegmentation
# ARG PYTORCH
# ARG CUDA
# ARG MMCV
# ARG MMDET
# ARG MMSEG
# RUN pip install openmim
# RUN ["/bin/bash", "-c", "mim install --no-cache-dir mmcv-full==${MMCV} -f https://download.openmmlab.com/mmcv/dist/cu${CUDA//./}/torch${PYTORCH}/index.html"]
# # RUN mim install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10/index.html
# RUN mim install --no-cache-dir mmdet mmsegmentation

# # Install MMDetection3D
# RUN conda clean --all
# COPY . /mmdetection3d
# WORKDIR /mmdetection3d
# ENV FORCE_CUDA="1"
# RUN pip install -r requirements/build.txt
# RUN pip install -r requirements/optional.txt
# RUN pip install -r requirements/runtime.txt
# RUN pip install cumm-cu116 spconv-cu116
# RUN pip install -v -e .

# # Fix fraction gcd to math deprecation: https://github.com/smicallef/spiderfoot/issues/1124
# RUN pip install networkx==2.5
# RUN pip install tensorflow==2.5.0
# RUN pip install protobuf==3.20.0

ARG PYTORCH="1.10.0"
ARG CUDA="11.3"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    FORCE_CUDA="1"

# Avoid Public GPG key error
# https://github.com/NVIDIA/nvidia-docker/issues/1631
RUN rm /etc/apt/sources.list.d/cuda.list \
    && rm /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-key del 7fa2af80 \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# (Optional, use Mirror to speed up downloads)
# RUN sed -i 's/http:\/\/archive.ubuntu.com\/ubuntu\//http:\/\/mirrors.aliyun.com\/ubuntu\//g' /etc/apt/sources.list && \
#    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Install the required packages
RUN apt-get update \
    && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMEngine, MMCV and MMDetection
RUN pip install openmim
RUN mim install --no-cache-dir mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
RUN mim install mmengine mmdet mmsegmentation

# Install MMDetection3D
RUN conda clean --all
COPY . /mmdetection3d
WORKDIR /mmdetection3d

ENV FORCE_CUDA="1"
RUN pip install -r requirements/build.txt
RUN pip install -r requirements/optional.txt
RUN pip install -r requirements/runtime.txt
RUN pip install cumm-cu113 spconv-cu113
RUN pip install -v -e .

# RUN pip install protobuf==3.9.2