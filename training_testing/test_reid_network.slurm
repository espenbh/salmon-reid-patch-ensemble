#!/bin/sh
#SBATCH --account=share-ie-idi       
#SBATCH --job-name=constrative_training
#SBATCH --time=0-01:00:00         

#SBATCH --partition=GPUQ          
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu40g
#SBATCH --mem=40g
#SBATCH --nodes=1
#SBATCH --output=output.txt       
#SBATCH --error=output.err        

#SBATCH --mail-user=espen.b.hogstedt@ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

module purge

module load Anaconda3/2022.10
conda activate patch_embedding
python test_reid_network.py