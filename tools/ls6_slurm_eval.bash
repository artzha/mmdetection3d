#!/bin/bash
#SBATCH -J eval_mmlab_model                # Job name
#SBATCH -o eval_mmlab_model.%j             # Name of stdout output file (%j expands to jobId)
#SBATCH -p gpu-a100                        # Queue name
#SBATCH -N 1                               # Total number of nodes requested (128 cores/node)
#SBATCH -n 1                               # Total number of mpi tasks requested
#SBATCH -t 01:00:00                        # Run time (hh:mm:ss)
#SBATCH -A IRI23004                        # Allocation name

export OMP_NUM_THREADS=3

module load tacc-apptainer
module load tacc-singularity
cd /work/09156/arthurz/mmdetection3d

#singularity build mmdetection.sif docker://artzha/mmdetection3d:latest
export NUM_GPUS=3
export SINGULARITYENV_CUDA_VISIBLE_DEVICES=0,1,2
export PORT=29501

export PILLAR_WAYMO_MODEL=hv_pointpillars_secfpn_sbn_2x16_2x_waymo-3d-3class
export CONFIG_PILLARS_WAYMO=configs/pointpillars/${PILLAR_WAYMO_MODEL}.py
export PILLAR_WAYMO_EVAL_FILE=work_dirs/${PILLAR_WAYMO_MODEL}/result/waymo_eval.pkl
export PILLAR_WAYMO_CKPT=work_dirs/${PILLAR_WAYMO_MODEL}/latest.pth

export PILLAR_NUSCENES_MODEL=hv_pointpillars_fpn_sbn-all_fp16_2x8_2x_nus-3d
export CONFIG_PILLARS_NUSCENES=configs/pointpillars/${PILLAR_NUSCENES_MODEL}.py
export PILLAR_NUSCENES_EVAL_FILE=work_dirs/${PILLAR_NUSCENES_MODEL}/result/nuscenes_eval.pkl
export PILLAR_NUSCENES_CKPT=work_dirs/${PILLAR_NUSCENES_MODEL}/latest.pth

export PILLAR_KITTI_MODEL=hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class
export CONFIG_PILLARS_KITTI=configs/pointpillars/${PILLAR_KITTI_MODEL}.py
export PILLAR_KITTI_WORK_DIR=work_dirs/${PILLAR_KITTI_MODEL}

export CP_NUSCENES_MODEL=centerpoint_0075voxel_second_secfpn_dcn_circlenms_4x8_cyclic_flip-tta_20e_nus_3class
export CONFIG_CP_NUSCENES=configs/centerpoint/${CP_NUSCENES_MODEL}.py
export CP_NUSCENES_EVAL_FILE=work_dirs/${CP_NUSCENES_MODEL}/result/nuscenes_eval.pkl
export CP_NUSCENES_CKPT=work_dirs/${CP_NUSCENES_MODEL}/latest.pth

#singularity exec --nv docker://artzha/mmdetection3d:latest bash tools/dist_test.sh ${CONFIG_PILLARS_WAYMO} ${PILLAR_WAYMO_CKPT} ${NUM_GPUS} --out ${PILLAR_WAYMO_EVAL_FILE} 

#singularity exec --nv docker://artzha/mmdetection3d:latest bash tools/dist_test.sh ${CONFIG_PILLARS_NUSCENES} ${PILLAR_NUSCENES_CKPT} ${NUM_GPUS} --out ${PILLAR_NUSCENES_EVAL_FILE}  --eval bbox

singularity exec --nv docker://artzha/mmdetection3d:latest bash tools/dist_test.sh ${CONFIG_CP_NUSCENES} ${CP_NUSCENES_CKPT} ${NUM_GPUS} --out ${CP_NUSCENES_EVAL_FILE} --eval bbox

