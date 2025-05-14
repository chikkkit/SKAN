# SKAN: Single-parameter Kolmogorov-Arnold Networks

This repository contains the experimental code for the paper "SKAN: Single-parameter Kolmogorov-Arnold Networks". The experiments are organized into four main folders:

## Repository Structure

### 1. pre-experiments/
Preliminary experiments comparing different grid sizes in Spl-KAN (the original KAN variant) on the MNIST dataset. These experiments helps verify the EKE (efficient KAN expansion) principle.

### 2. comparision of KAN variants on MNIST/
Comprehensive comparison between our proposed SKAN and other KAN variants on the MNIST dataset:
- Different SKAN variants with various basis functions
- Classic KAN variants (FourierKAN, WaveKAN, FastKAN, etc.)
- Extensive experiments with different learning rates and architectures

### 3. ODE solving/
Implementation and comparison of x-SKAN-ODE (where x represents different SKAN basis functions) with:
- KAN-ODE (based on Spl-KAN)
- Neural-ODE (based on MLP)

The experiments focus on solving the Lotka-Volterra predator-prey differential equations, demonstrating SKAN's capability in scientific computing tasks.

### 4. medical image segmentation/
Application of SKAN in medical image segmentation tasks, comparing:
- U-x-SKAN (our model with different basis functions)
- U-KAN (based on Spl-KAN)
- U-Net (based on MLP)

Experiments were conducted on three datasets:
- BUSI: Breast ultrasound image dataset
  - Additional experiments with reduced-parameter SKAN variants
- CVC-ClinicDB: Polyp segmentation dataset
- GlaS: Gland segmentation dataset

## Key Features
- Novel single-parameterized architecture (SKAN)
- Multiple basis function variants (LSS-SKAN, LSin-SKAN, LArctan-SKAN)
- Comprehensive benchmarking against existing KAN variants
- Applications in both scientific computing and computer vision tasks

## Usage
Each folder contains its own README with detailed instructions for running experiments. The code is organized to be modular and easy to extend for new applications.

## License
Currently, this repository contains only the LICENSE files from referenced codebases (some original repositories do not specify licenses). For double-blind review purposes, we have not included our own license to maintain anonymity. After publication, this repository will be released under the MIT License.