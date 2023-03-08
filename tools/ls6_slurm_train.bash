#!/bin/bash
#SBATCH -J train_mmlab_model               # Job name
#SBATCH -o train_mmlab_model.%j            # Name of stdout output file (%j expands to jobId)
#SBATCH -p gpu-a100                        # Queue name
#SBATCH -N 1                               # Total number of nodes requested (128 cores/node)
#SBATCH -n 1                               # Total number of mpi tasks requested
#SBATCH -t 00:10:00                        # Run time (hh:mm:ss)
#SBATCH -A IRI23004                        # Allocation name

export OMP_NUM_THREADS=4

module load tacc-apptainer
module load tacc-singularity
cd /work/09156/arthurz/mmdetection3d

#singularity build mmdetection.sif docker://artzha/mmdetection3d:latest
export NUM_GPUS             = 3
export CUDA_VISIBLE_DEVICES = 0,1,2

export PILLAR_WAYMO_MODEL   = hv_pointpillars_secfpn_2x16_2x_waymoD5-3d-3class
export CONFIG_PILLARS_WAYMO = configs/pointpillars/${PILLAR_WAYMO_MODEL}.py
export PILLAR_WORK_DIR      = work_dirs/${PILLAR_WAYMO_MODEL}
ibrun -n 1 -o 0 singularity exec docker://artzha/mmdetection3d:latest bash dist_train.sh ${CONFIG_PILLARS_WAYMO} ${NUM_GPUS} --work-dir ${PILLAR_WORK_DIR}

# export PILLAR_NUSCENES_MODEL    = hv_pointpillars_secfpn_2x16_2x_waymoD5-3d-3class
# export CONFIG_PILLARS_NUSCENES  = configs/pointpillars/${PILLAR_NUSCENES_MODEL}.py
# export PILLAR_WORK_DIR          = work_dirs/${PILLAR_NUSCENES_MODEL}
# ibrun -n 1 -o 1 singularity exec docker://artzha/mmdetection3d:latest bash dist_train.sh ${CONFIG_PILLARS_WAYMO} ${NUM_GPUS} --work-dir ${PILLAR_WORK_DIR} &

# wait