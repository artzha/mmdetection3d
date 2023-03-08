#!/bin/bash
#SBATCH -J create_mmlab_data               # Job name
#SBATCH -o create_mmlab_data.%j            # Name of stdout output file (%j expands to jobId)
#SBATCH -p normal                          # Queue name
#SBATCH -N 2                               # Total number of nodes requested (128 cores/node)
#SBATCH -n 2                               # Total number of mpi tasks requested
#SBATCH -t 00:05:00                        # Run time (hh:mm:ss)
#SBATCH --reservation Containers_Training  # a reservation only active during the training

cd $WORK/09156/arthurz/mmdetection3d
module load tacc-apptainer
ibrun singularity exec mmdetection3d.sif python3 /mmdetection3d/tools/create_data.py waymo --root-path /scratch/09156/arthurz/mmlab/waymo --out-dir /scratch/09156/arthurz/mmlab/waymo --workers 128 --extra-tag waymo
ibrun singularity exec mmdetection3d.sif python3 /mmdetection3d/tools/create_data.py nuscenes --root-path /scratch/09156/arthurz/mmlab/nuscenes --out-dir /scratch/09156/arthurz/mmlab/nuscenes --extra-tag nuscenes