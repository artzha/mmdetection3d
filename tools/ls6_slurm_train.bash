#!/bin/bash
#SBATCH -J train_mmlab_model               # Job name
#SBATCH -o train_mmlab_model.%j            # Name of stdout output file (%j expands to jobId)
#SBATCH -p gpu-a100                        # Queue name
#SBATCH -N 1                               # Total number of nodes requested (128 cores/node)
#SBATCH -n 1                               # Total number of mpi tasks requested
#SBATCH -t 00:30:00                        # Run time (hh:mm:ss)
#SBATCH -A IRI23004                        # Allocation name

export OMP_NUM_THREADS=3
export SINGULARITYENV_CUDA_VISIBLE_DEVICES=0,1,2
module load tacc-apptainer
module load tacc-singularity
cd /work/09156/arthurz/mmdetection3d

# singularity build mmdetection.sif docker://artzha/mmdetection3d:latest
export NUM_GPUS=3
export CUDA_VISIBLE_DEVICES=0,1,2

#PointPillar Models
# export WAYMO_MODEL=hv_pointpillars_secfpn_sbn_2x16_2x_waymo-3d-3class
# export CONFIG_WAYMO=configs/pointpillars/${WAYMO_MODEL}.py
# export PILLAR_WAYMO_WORK_DIR=work_dirs/${PILLAR_WAYMO_MODEL}

# export NUSCENES_MODEL=hv_pointpillars_fpn_sbn-all_fp16_2x8_2x_nus-3d
# export CONFIG_NUSCENES=configs/pointpillars/${NUSCENES_MODEL}.py
# export NUSCENES_WORK_DIR=work_dirs/${NUSCENES_MODEL}

# export KITTI_MODEL=hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class
# export CONFIG_KITTI=configs/pointpillars/${KITTI_MODEL}.py
# export KITTI_WORK_DIR=work_dirs/${KITTI_MODEL}

# CenterPoint Models
export PORT=29501
export WAYMO_MODEL=centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_flip-tta_20e_waymo
export CONFIG_WAYMO=configs/centerpoint/${WAYMO_MODEL}.py
export WAYMO_WORK_DIR=work_dirs/${WAYMO_MODEL}

# Uncomment to launch kitti
# module load launcher_gpu
# export LAUNCHER_WORKDIR=/work/09156/arthurz/mmdetection3d
# export LAUNCHER_JOB_FILE=tools/launcher_train_kitti_pp

# Uncomment to launch waymo
module load launcher_gpu
export LAUNCHER_WORKDIR=/work/09156/arthurz/mmdetection3d
export LAUNCHER_JOB_FILE=tools/launcher_train_waymo

${LAUNCHER_DIR}/paramrun
