#!/bin/bash
#SBATCH -J train_mmlab_model               # Job name
#SBATCH -o train_mmlab_model.%j            # Name of stdout output file (%j expands to jobId)
#SBATCH -p gpu-a100                        # Queue name
#SBATCH -N 3                               # Total number of nodes requested (128 cores/node)
#SBATCH -n 1                               # Total number of mpi tasks requested
#SBATCH -t 00:05:00                        # Run time (hh:mm:ss)
#SBATCH -A IRI23004                        # Allocation name

export OMP_NUM_THREADS=4

module load tacc-apptainer
module load tacc-singularity
cd /work/09156/arthurz/mmdetection3d

# singularity build mmdetection.sif docker://artzha/mmdetection3d:latest
export NUM_GPUS=3
export CUDA_VISIBLE_DEVICES=0,1,2

export PILLAR_WAYMO_MODEL=hv_pointpillars_secfpn_sbn_2x16_2x_waymo-3d-3class
export CONFIG_PILLARS_WAYMO=configs/pointpillars/${PILLAR_WAYMO_MODEL}.py
export PILLAR_WAYMO_WORK_DIR=work_dirs/${PILLAR_WAYMO_MODEL}

export PILLAR_NUSCENES_MODEL=hv_pointpillars_fpn_sbn-all_fp16_2x8_2x_nus-3d
export CONFIG_PILLARS_NUSCENES=configs/pointpillars/${PILLAR_NUSCENES_MODEL}.py
export PILLAR_NUSCENES_WORK_DIR=work_dirs/${PILLAR_NUSCENES_MODEL}

export PILLAR_KITTI_MODEL=hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class
export CONFIG_PILLARS_KITTI=configs/pointpillars/${PILLAR_KITTI_MODEL}.py
export PILLAR_KITTI_WORK_DIR=work_dirs/${PILLAR_KITTI_MODEL}

module load launcher_gpu
export LAUNCHER_WORKDIR=/work/09156/arthurz/mmdetection3d
export LAUNCHER_JOB_FILE=launcher_train_models

${LAUNCHER_DIR}/paramrun

