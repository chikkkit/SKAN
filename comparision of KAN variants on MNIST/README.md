# Comparison of KAN Variants on MNIST

## Introduction
This folder contains two experimental comparisons of different KAN variants on the MNIST dataset:

1. `SKAN_comparision.py`: Compares different SKAN variants with various basis functions based on activation functions (10 epochs)
2. `LSS_LArctan_SKAN_comparison.py`: Comprehensive comparison of all KAN variants including LSS-SKAN, LArctan-SKAN and other classic KAN variants (30 epochs)

## Usage
This code runs on Python 3.12.3. Required Python packages:

```python
numpy==2.1.2
pandas==2.2.3
scikit_learn==1.5.2
torch==2.4.1+cu121
torchvision==0.19.1+cu121
tqdm==4.66.4
fkan==0.0.2
rkan==0.0.3
skan==0.2.0
```

To run the experiments:
```bash
# For SKAN comparison with activation-based basis functions (10 epochs)
python SKAN_comparision.py

# For comprehensive comparison of all KAN variants (30 epochs)
python LSS_LArctan_SKAN_comparison.py
```

## Experiment Details

### SKAN Comparison
Tests various SKAN activation functions:
- lrelu, lleaky_relu, lswish, lmish, lsoftplus
- lhard_sigmoid, lelu, lshifted_softplus, lgelup

### Comprehensive KAN Comparison
Compares all KAN variants:

Classic KAN variants:
- FourierKAN
- WaveKAN
- FastKAN
- Spl-KAN (EfficientKAN)
- fKAN
- rKAN

SKAN variants:
- LSS-SKAN (lshifted_softplus)
- LSin-SKAN
- LCos-SKAN
- LArctan-SKAN

All experiments record:
- Training/test loss and accuracy
- F1 score
- Running time
- Parameter count 