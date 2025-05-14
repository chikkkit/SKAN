#!/bin/bash

#SBATCH --job-name=pred_prey      # Job name
#SBATCH --partition=cpuXeon6458   # CPU queue
#SBATCH -N 1                      # Only need 1 node
#SBATCH --ntasks-per-node=1       # 1 task per node
#SBATCH --cpus-per-task=2         # 2 CPU cores per task
#SBATCH --array=0-49              # Array job with 50 tasks (0-49)
#SBATCH --output=%A_%a.out        # Standard output
#SBATCH --error=%A_%a.err         # Standard error output

# Activate conda environment
module load anaconda/3-2023.09
source activate ode_env  # Replace with your environment name

# Define model and basis function arrays
models=("SKAN" "MLP" "KAN")
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
elif [ $model_idx -eq 1 ]; then
    model="KAN"
    basis=""
else
    model="SKAN"
    basis_idx=$((model_idx - 2))
    basis=${basis_functions[$basis_idx]}
fi

# Construct output directory
if [ "$model" = "SKAN" ]; then
    output_dir="output/${model}_${basis}_${run_idx}"
else
    output_dir="output/${model}_${run_idx}"
fi

# Create output directory
mkdir -p "$output_dir"

echo "Running experiment for $model with basis $basis - Round $run_idx"
echo "Output directory: $output_dir"

# Run Python script
if [ "$model" = "SKAN" ]; then
    python predator_prey.py \
        --model "$model" \
        --basis "$basis" \
        --output_dir "$output_dir"
else
    python predator_prey.py \
        --model "$model" \
        --output_dir "$output_dir"
fi

# Check if training completed successfully
if [ $? -ne 0 ]; then
    echo "Error occurred while running $model experiment round $run_idx"
    exit 1
fi

echo "Completed $model experiment round $run_idx"
