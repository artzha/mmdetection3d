#!/bin/bash
#SBATCH -J create_mmlab_data               # Job name
#SBATCH -o create_mmlab_data.%j            # Name of stdout output file (%j expands to jobId)
#SBATCH -p normal                          # Queue name
#SBATCH -N 1                               # Total number of nodes requested (128 cores/node)
#SBATCH -n 2                               # Total number of mpi tasks requested
#SBATCH -t 06:00:00                        # Run time (hh:mm:ss)
#SBATCH -A IRI23004                        # Allocation name

export OMP_NUM_THREADS=2
module load tacc-apptainer
module load tacc-singularity
cd /work/09156/arthurz/mmdetection3d

module load launcher
export LAUNCHER_WORKDIR=/work/09156/arthurz/mmdetection3d
export LAUNCHER_JOB_FILE=${LAUNCHER_WORKDIR}/tools/launcher_create_data

${LAUNCHER_DIR}/paramrun