#!/bin/bash

#SBATCH --job-name=skan_basis2    # Job name
#SBATCH --partition=gpuA800       # GPU queue
#SBATCH -N 1                      # Only need 1 node
#SBATCH --ntasks-per-node=1       # 1 task per node
#SBATCH --cpus-per-task=9         # 9 CPU cores per task
#SBATCH --gres=gpu:1              # 1 GPU per task
#SBATCH --array=0-29              # 30 tasks (3 basis functions * 10 rounds)
#SBATCH --output=%A_%a.out        # Standard output
#SBATCH --error=%A_%a.err         # Standard error output

# Activate conda environment
module load anaconda/3-2023.09
source activate ukan9

# Define basis functions
basis_functions=("lshifted_softplus" "larctan" "lsin")

# Get the current task ID
TASK_ID=$SLURM_ARRAY_TASK_ID

# Calculate which basis function and round to run
basis_idx=$((TASK_ID / 10))
run_idx=$(((TASK_ID % 10) + 1))  # Numbered from 1 to 10

# Get basis function
basis=${basis_functions[$basis_idx]}

# Construct output folder name
output_name="busi_skan_${basis}_275_${run_idx}"

echo "Running SKAN with basis $basis - Round $run_idx - input_list=[128,275,256]"
echo "Output folder: $output_name"

# Run training command
python train.py \
    --arch UKAN \
    --dataset busi \
    --input_w 256 \
    --input_h 256 \
    --name "$output_name" \
    --data_dir ./inputs \
    --model "SKAN" \
    --basis_function "$basis" \
    --batch_size 4 \
    --input_list "128,275,256"

# Check if training completed successfully
if [ $? -ne 0 ]; then
    echo "Error occurred while running SKAN experiment with $basis round $run_idx"
    exit 1
fi

echo "Completed SKAN experiment with $basis round $run_idx" 