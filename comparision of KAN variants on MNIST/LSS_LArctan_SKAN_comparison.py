import tqdm
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from sklearn.metrics import f1_score
import numpy as np
from rkan.torch import JacobiRKAN, PadeRKAN
from fkan.torch import FractionalJacobiNeuralBlock as fJNB
import pandas as pd
import importlib
from skan import SKANNetwork

# Define custom basis functions for SKAN
def lsin(x, k):
    return k * torch.sin(x)

def lcos(x, k):
    return k * torch.cos(x)

def larctan(x, k):
    return k * torch.atan(x)

def lshifted_softplus(x, k):
    return torch.log(1 + torch.exp(k*x)) - np.log(2)

# MNIST dataset setup
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Lambda(lambda x: x.view(-1))
])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

res_name = time.asctime().replace(' ', '_').replace(':', '_') + ' Result.csv'

# Define all KAN variants to test
kan_variants = [
    # Classic KAN variants
    'FourierKAN', 'waveKAN', 'fastKAN', 'efficientKAN', 'fkan', 'rkan',
    # SKAN variants
    'lshifted_softplus', 'lsin', 'lcos', 'larctan'
]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

for lr in [*np.linspace(0.001, 0.01, 10), *np.linspace(0.0001, 0.0009, 9)]:
    for variant in kan_variants:
        # Initialize network based on variant type
        if variant == 'waveKAN':
            wavKAN = importlib.import_module("wavKAN")
            KAN = getattr(wavKAN, 'KAN')
            net = KAN([28 * 28, 26, 10], wavelet_type='dog').to(device)
        elif variant == 'fastKAN':
            fastkan = importlib.import_module("fastkan")
            FastKAN = getattr(fastkan, 'FastKAN')
            net = FastKAN([28*28, 11, 10]).to(device)
        elif variant == 'efficientKAN':
            efficientkan = importlib.import_module("efficientKAN")
            EfficientKAN = getattr(efficientkan, 'KAN')
            net = EfficientKAN([28*28, 17, 10], grid_size=1).to(device)
        elif variant == 'fkan':
            net = nn.Sequential(
                nn.Linear(28 * 28, 100),
                fJNB(3),
                nn.Linear(100, 10)
            ).to(device)
        elif variant == 'rkan':
            net = nn.Sequential(
                nn.Linear(28 * 28, 90),
                JacobiRKAN(3),
                nn.Linear(90, 90),
                PadeRKAN(2, 6),
                nn.Linear(90, 10)
            ).to(device)
        elif variant == 'FourierKAN':
            fkan = importlib.import_module("fftKAN")
            NaiveFourierKANLayer = getattr(fkan, 'NaiveFourierKANLayer')
            net = nn.Sequential(
                NaiveFourierKANLayer(28 * 28, 50, 1),
                NaiveFourierKANLayer(50, 10, 1)
            ).to(device)
        else:  # SKAN variants
            net = SKANNetwork([784, 100, 10], basis_function=eval(variant)).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=lr)
        
        # Training and evaluation loop
        train_loss = []
        train_accuracy = []
        test_loss = []
        test_accuracy = []
        F1s = []
        
        for epoch in tqdm.trange(30):
            # Training phase
            net.train()
            start_time = time.time()
            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                y_pred = net(x)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
            end_time = time.time()

            # Evaluation on training set
            net.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                loss = 0
                for x, y in train_loader:
                    x = x.to(device)
                    y = y.to(device)
                    y_pred = net(x)
                    loss += criterion(y_pred, y).item()
                    _, predicted = torch.max(y_pred.data, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()
                train_loss.append(loss / len(train_loader))
                train_accuracy.append(correct / total)

            # Evaluation on test set
            net.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                loss = 0
                reals = []
                preds = []
                for x, y in test_loader:
                    x = x.to(device)
                    y = y.to(device)
                    y_pred = net(x)
                    loss += criterion(y_pred, y).item()
                    _, predicted = torch.max(y_pred.data, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()
                    reals.extend(y.cpu().numpy())
                    preds.extend(predicted.cpu().numpy())
                
                F1 = f1_score(reals, preds, average='macro')
                test_loss.append(loss / len(test_loader))
                test_accuracy.append(correct / total)

            # Save results
            results = {
                'KAN variant': variant,
                'epoch': epoch,
                'lr': lr,
                'train loss': train_loss[-1],
                'train accuracy': train_accuracy[-1],
                'test loss': test_loss[-1],
                'test accuracy': test_accuracy[-1],
                'F1': F1,
                'run time': round(end_time - start_time, 4),
                'param num': count_parameters(net)
            }
            
            if 'res' not in locals():
                res = pd.DataFrame([results])
            else:
                res = pd.concat([res, pd.DataFrame([results])], ignore_index=True)
            
            res.to_csv('./result/' + res_name, index=False) 