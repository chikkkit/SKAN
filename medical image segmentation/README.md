# Medical Image Segmentation with SKAN

## Introduction
First, follow the environment setup instructions from https://github.com/CUHK-AIM-Group/U-KAN. Note that some older Python packages may require manual source code modifications and solutions found online. A Linux system is recommended as many legacy packages are only available on Linux. The dataset download links can be found in the U-KAN GitHub repository.

## Implementation Details
We mainly modified the archs.py and train.py files. In archs.py, we integrated SKAN to replace Spl-KAN, and in train.py, we added new parameters for SKAN training.

## Usage

### Single Experiment
After setting up the environment, you can train the model using the following commands. Replace `{dataset}` with either `cvc`, `glas`, or `busi`:

```bash
# For MLP baseline
python train.py \
    --arch UKAN \
    --dataset {dataset} \
    --input_w 256 \
    --input_h 256 \
    --name "{dataset}_mlp_run{N}" \
    --data_dir ./inputs \
    --model MLP \
    --batch_size 4 \
    --input_list 128,160,256

# For KAN model
python train.py \
    --arch UKAN \
    --dataset {dataset} \
    --input_w 256 \
    --input_h 256 \
    --name "{dataset}_kan_run{N}" \
    --data_dir ./inputs \
    --model KAN \
    --batch_size 4 \
    --input_list 128,160,256

# For SKAN model with different basis functions
python train.py \
    --arch UKAN \
    --dataset {dataset} \
    --input_w 256 \
    --input_h 256 \
    --name "{dataset}_skan_{basis}_run{N}" \
    --data_dir ./inputs \
    --model SKAN \
    --basis_function {basis} \
    --batch_size 4 \
    --input_list 128,390,256

# Available basis functions for SKAN:
# {basis} can be: lsin, larctan, or lshifted_softplus
```

## Datasets
Dataset-specific notes:
- CVC-ClinicDB (256×256): Medical image dataset for polyp segmentation
- GlaS (512×512): Gland segmentation dataset
- BUSI (256×256): Breast ultrasound image dataset

## Batch Experiments

### Standard Experiments
For batch experiments, we provide SLURM scripts for each dataset:
- `run_on_cvc.slurm`: Runs 50 experiments on CVC-ClinicDB dataset
- `run_on_glas.slurm`: Runs 50 experiments on GlaS dataset (uses 512×512 resolution)
- `run_on_busi.slurm`: Runs 45 experiments on BUSI dataset

Each SLURM script will automatically run multiple experiments with different models and basis functions:
- 10 runs for MLP baseline
- 10 runs for KAN
- 30 runs for SKAN (10 runs for each basis function)

### Additional SKAN Experiments
We also provide two extra SLURM scripts to investigate SKAN performance with reduced parameters:
- `run_basis_1.slurm`: Runs SKAN with input_list=[128,160,256]
- `run_basis_2.slurm`: Runs SKAN with input_list=[128,275,256]

Each script runs 30 experiments (10 runs × 3 basis functions). Results will be saved with names:
- `busi_skan_{basis}_160_run{N}` for 160-dimension experiments
- `busi_skan_{basis}_275_run{N}` for 275-dimension experiments

### Running Batch Jobs
```bash
# For standard experiments
sbatch run_on_cvc.slurm
sbatch run_on_glas.slurm
sbatch run_on_busi.slurm

# For additional SKAN experiments on BUSI
sbatch run_basis_1.slurm  # 160-dimension version
sbatch run_basis_2.slurm  # 275-dimension version
```

## Output Format
Each experiment's results will be saved in the outputs directory with the corresponding experiment name following the format:
- `{dataset}_mlp_run{N}`
- `{dataset}_kan_run{N}`
- `{dataset}_skan_{basis}_run{N}`

where:
- `{dataset}`: cvc, glas, or busi
- `{basis}`: lsin, larctan, or lshifted_softplus
- `{N}`: run number (1-10)