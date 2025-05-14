import sys
sys.path.append('./modelnetwork')
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

# Use MNIST dataset
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Lambda(lambda x: x.view(-1))
])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

res_name = time.asctime().replace(' ', '_').replace(':', '_') + ' Result.csv'
res_name = "LR search SKAN" + res_name

# Single-parameterized functions
lfuns = ['lrelu', 'lleaky_relu', 'lswish', 'lmish', 'lsoftplus', 'lhard_sigmoid', 
         'lelu', 'lshifted_softplus', 'lgelup']

# Test these univariate functions and save results in dataframe
res = None

# Calculate network parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

for lr in [*np.linspace(0.01, 0.09, 9), *np.linspace(0.001, 0.009, 9), *np.linspace(0.0001, 0.0009, 10)]:
    for lfun in lfuns:
        if lfun == 'waveKAN':
            wavKAN = importlib.import_module("wavKAN")
            KAN = getattr(wavKAN, 'KAN')
            wavelet_types = 'dog'
            net = KAN([28 * 28, 26, 10], wavelet_type=wavelet_types).to(device)
        elif lfun == 'fastKAN':
            fastkan = importlib.import_module("fastkan")
            FastKAN = getattr(fastkan, 'FastKAN')
            net = FastKAN([28*28, 11, 10]).to(device)
        elif lfun == 'efficientKAN':
            efficientkan = importlib.import_module("efficientKAN")
            EfficientKAN = getattr(efficientkan, 'KAN')
            net = EfficientKAN([28*28, 17, 10], grid_size=1).to(device)
        elif lfun == 'fkan':
            net = nn.Sequential(
                nn.Linear(28 * 28, 100),
                fJNB(3),
                nn.Linear(100, 10)
            ).to(device)
        elif lfun == 'rkan':
            net = nn.Sequential(
                nn.Linear(28 * 28, 90),
                JacobiRKAN(3),      # Jacobi polynomial of degree 3
                nn.Linear(90, 90),
                PadeRKAN(2, 6),     # Pade [2/6]
                nn.Linear(90, 10)
            ).to(device)
        else:
            skan = importlib.import_module("skan_exp_version")
            base_function = getattr(skan, lfun)
            reluKANNetwork = getattr(skan, 'SKANNetwork')
            net = reluKANNetwork([784, 100, 10], base_function=base_function).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=lr)
        train_loss = []
        train_accuracy = []
        test_loss = []
        test_accuracy = []
        F1s = []
        for epoch in tqdm.trange(10):
            net.train()
            # Record training time
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
            
            # Calculate accuracy and loss on training set
            net.eval()
            correct = 0
            total = 0
            loss = 0
            with torch.no_grad():
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
            
            # Calculate accuracy and loss on test set
            net.eval()
            correct = 0
            total = 0
            loss = 0
            reals = []
            preds = []
            with torch.no_grad():
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
            param_num = count_parameters(net)
            if res is not None:
                res = res.reset_index(drop=True)
                res = pd.concat([res, pd.DataFrame({'function/KAN type': lfun, 'epoch': epoch, 'lr': lr, 'train loss': train_loss[-1], 
                                                    'train accuracy': train_accuracy[-1], 'test loss': test_loss[-1], 
                                                    'test accuracy': test_accuracy[-1], 'F1': F1, 'run time': round(end_time - start_time, 4),
                                                    'param num': param_num}, index=[-1])], axis=0)
            else:
                res = pd.DataFrame({'function/KAN type': lfun, 'epoch': epoch, 'lr': lr, 'train loss': train_loss[-1], 
                                    'train accuracy': train_accuracy[-1], 'test loss': test_loss[-1], 
                                    'test accuracy': test_accuracy[-1], 'F1': F1, 'run time': round(end_time - start_time, 4),
                                    'param num': param_num}, index=[-1])
            res.to_csv('./result/' + res_name, index=False)
