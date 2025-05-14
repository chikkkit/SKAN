#!/bin/bash

#SBATCH --job-name=cvc_exp        # Job name
#SBATCH --partition=gpuA800       # GPU queue
#SBATCH -N 1                      # Only need 1 node
#SBATCH --ntasks-per-node=1       # 1 task per node
#SBATCH --cpus-per-task=9         # 9 CPU cores per task
#SBATCH --gres=gpu:1              # 1 GPU per task
#SBATCH --array=0-49              # Array job (0-49)
#SBATCH --output=%A_%a.out        # Standard output
#SBATCH --error=%A_%a.err         # Standard error output

# Activate conda environment
module load anaconda/3-2023.09
source activate ukan9

# Define model array
models=("MLP" "KAN" "SKAN")
basis_functions=("lsin" "larctan" "lshifted_softplus")

# Get the current task ID
TASK_ID=$SLURM_ARRAY_TASK_ID

# Calculate which model and experiment round to run
model_idx=$((TASK_ID / 10))
run_idx=$((TASK_ID % 10 + 1))  # Numbered from 1 to 10

# Determine the specific model and basis function
if [ $model_idx -eq 0 ]; then
    model="MLP"
    basis=""
    input_list="128,160,256"
elif [ $model_idx -eq 1 ]; then
    model="KAN"
    basis=""
    input_list="128,160,256"
else
    model="SKAN"
    basis_idx=$((model_idx - 2))
    basis=${basis_functions[$basis_idx]}
    input_list="128,390,256"
fi

# Construct the output folder name
if [ "$model" = "SKAN" ]; then
    output_name="cvc_${model,,}_${basis}_${run_idx}"
else
    output_name="cvc_${model,,}_${run_idx}"
fi

echo "Running experiment for $model with basis $basis - Round $run_idx"
echo "Output folder: $output_name"

# Run the training command
python train.py \
    --arch UKAN \
    --dataset CVC-ClinicDB \
    --input_w 256 \
    --input_h 256 \
    --name "$output_name" \
    --data_dir ./inputs \
    --model "$model" \
    --basis_function "$basis" \
    --batch_size 4 \
    --input_list "$input_list"

# Check if the training completed successfully
if [ $? -ne 0 ]; then
    echo "Error occurred while running $model experiment round $run_idx"
    exit 1
fi

echo "Completed $model experiment round $run_idx" 