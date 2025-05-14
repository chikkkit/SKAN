# ODE Solving Experiments
This folder contains experiments comparing different KAN variants on solving the Lotka-Volterra predator-prey differential equations. This implementation is based on [KAN-ODEs](https://github.com/DENG-MIT/KAN-ODEs) [1], with extensions to support SKAN variants.

## Usage
To run a single experiment:
```bash
python predator_prey.py \
    --model [MODEL] \
    --basis [BASIS] \
    --output_dir [OUTPUT_DIR]

# Available options:
# MODEL: SKAN, KAN, or MLP
# BASIS: lsin, larctan, or lshifted_softplus (only for SKAN, otherwise ignored)
```

For batch experiments using SLURM:
```bash
sbatch run_experiments.slurm
```

## Experiment Details

### Models
- MLP: Standard multilayer perceptron (baseline)
- KAN: Original Kolmogorov-Arnold Network (Spl-KAN)
- SKAN: Single-parameterized KAN with different basis functions:
  - LSS-SKAN (lshifted_softplus)
  - LSin-SKAN (lsin)
  - LArctan-SKAN (larctan)

### Training Configuration
- Training period: [0, 3.5]
- Testing period: [0, 14]
- Initial conditions: x(0) = 1, y(0) = 1
- Learning rate: 1e-3
- Epochs: 10,000

### Output
Results are saved in the specified output directory:
- Model checkpoints
- Loss curves
- Prediction plots
- CSV files with training metrics
- Visualization of predictions vs ground truth

The SLURM script will create separate output directories for each experiment:
- `output/MLP_run{N}`
- `output/KAN_run{N}`
- `output/SKAN_{basis}_run{N}`

where N is the run number (1-10) and basis is the chosen basis function for SKAN.