import torch
import numpy as np

## define some basis function for skan
# y = max(0, x), ReLU
def lrelu(x, k):
    return torch.clamp(k[0]*x, min=0)

# y = max(kx, x), leaky ReLU
def lleaky_relu(x, k):
    return torch.max(k[0]*x, x)

# y = x / (1 + e^(-kx)), Swish
def lswish(x, k):
    return x / (1 + torch.exp(-k[0]*x))

# y = x / (1 + e^(-kx)), Mish
def lmish(x, k):
    return x * torch.tanh(torch.log(1 + torch.exp(k[0]*x)))

# y = log(1 + e^(kx)), Softplus
def lsoftplus(x, k):
    return torch.log(1 + torch.exp(k[0]*x))

# y = max(0, min(1, (kx + 0.5))), Hard sigmoid
def lhard_sigmoid(x, k):
    return torch.clamp(k[0]*x + 0.5, min=0, max=1)

# y = k * (e^(x/k) - 1) if x < 0, else x, Exponential Linear Unit (ELU)
def lelu(x, k):
    return torch.where(x < 0, k[0] * (torch.exp(x/k[0]) - 1), x)

# y = log(1 + e^(kx)) - log(2), Shifted Softplus
def lshifted_softplus(x, k):
    return torch.log(1 + torch.exp(k[0]*x)) - np.log(2)

# y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (kx + 0.044715 * k^3 * x^3))), Gaussian Error Linear Unit with Parameter (GELU-P)
def lgelup(x, k):
    return 0.5 * x * (1 + torch.tanh(np.sqrt(2/np.pi) * (k[0]*x + 0.044715 * k[0]**3 * x[0]**3)))

def lsin(x, k):
    return torch.sin(k[0]*x)

def lcos(x, k):
    return torch.cos(k[0]*x)

def larctan(x, k):
    return torch.arctan(k[0]*x)

## define some basis function for ...