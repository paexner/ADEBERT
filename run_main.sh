#!/bin/bash
#SBATCH --output=slurm-output-%j.txt
#SBATCH --error=slurm-error-%j.err
#SBATCH --partition=tue.gpu.q
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1


module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0
pip cache purge
pip install -r requirements_gpu.txt
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

## Execute the script or command
export TF_GPU_ALLOCATOR=cuda_malloc_async
export CUDA_LAUNCH_BLOCKING=1
python main.py autoencoder sample_id