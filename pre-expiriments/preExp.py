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

res_name = 'preExp' + time.asctime().replace(' ', '_').replace(':', '_') + ' Result.csv'

# Single parameter nonlinear functions
grid_sizes = [1, 2, 3, 4, 5]

# Test the effectiveness of these univariate functions and save results in dataframe
res = None

# Calculate network parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
for lr in [*np.linspace(0.01, 0.09, 10), *np.linspace(0.001, 0.009, 9), *np.linspace(0.1, 1, 10)]:
    for grid_size in grid_sizes:
        efficientkan = importlib.import_module("efficientKAN")
        EfficientKAN = getattr(efficientkan, 'KAN')
        hidden_size = np.ceil(80000 / (794 * (grid_size+5))).astype(int)
        net = EfficientKAN([28*28, hidden_size, 10], grid_size=grid_size).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=lr)
        train_loss = []
        train_accuracy = []
        test_loss = []
        test_accuracy = []
        F1s = []
        for epoch in tqdm.trange(10):
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
            if res is not None:
                res = res.reset_index(drop=True)
                res = pd.concat([res, pd.DataFrame({'function': 'Spl-KAN', 'epoch': epoch, 'lr': lr, 'train loss': train_loss[-1], 
                                                    'train accuracy': train_accuracy[-1], 'test loss': test_loss[-1], 
                                                    'test accuracy': test_accuracy[-1], 'F1': F1, 'grid size': grid_size, 
                                                    'run time':round(end_time - start_time, 4), 'parameter num': count_parameters(net), 
                                                    'hidden size': hidden_size}, index=[-1])], axis=0)
            else:
                res = pd.DataFrame({'function': 'Spl-KAN', 'epoch': epoch, 'lr': lr, 'train loss': train_loss[-1], 
                                    'train accuracy': train_accuracy[-1], 'test loss': test_loss[-1], 
                                    'test accuracy': test_accuracy[-1], 'F1': F1, 'grid size': grid_size, 
                                    'run time':round(end_time - start_time, 4), 'parameter num': count_parameters(net), 
                                    'hidden size': hidden_size}, index=[-1])
            res.to_csv('./result/' + res_name, index=False)