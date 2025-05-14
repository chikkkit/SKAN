# Preliminary Experiments

## Introduction
This folder contains the preliminary experiments for comparing different grid sizes in Spl-KAN (the original KAN variant). The experiment aims to investigate the relationship between grid size, learning rate, and model performance on the MNIST dataset.

## Usage
This code runs on Python 3.12.3. To use the code, ensure that the following Python packages are installed:

```python
numpy==2.1.2
pandas==2.2.3
scikit_learn==1.5.2
torch==2.4.1+cu121
torchvision==0.19.1+cu121
tqdm==4.66.4
```

To run the preliminary experiment:
```bash
python preExp.py
```

## Experiment Details
The experiment tests Spl-KAN with:
- Grid sizes: [1, 2, 3, 4, 5]
- Learning rates: 
  - 0.01 to 0.09 (step: 0.01)
  - 0.001 to 0.009 (step: 0.001)
  - 0.1 to 1.0 (step: 0.1)
- Hidden layer size is automatically adjusted to maintain approximately 80,000 parameters
- Training for 10 epochs on MNIST

Results are saved in CSV format with metrics including:
- Training/test loss and accuracy
- F1 score
- Running time
- Parameter count
- Hidden layer size 