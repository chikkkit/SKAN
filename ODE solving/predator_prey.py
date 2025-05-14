import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy
from torchdiffeq import odeint as torchodeint
from tqdm import tqdm
import sys
import argparse
sys.path.append("efficient_kan/")
from unikan import SKAN_pure
import unikan.basis as basis
import os
from efficient_kan.kan import KAN
import torch.nn as nn
import pandas as pd

# parser
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default="output")
parser.add_argument("--model", type=str, default="SKAN", choices=["SKAN", "KAN", "MLP"])
parser.add_argument("--basis", type=str, default="lshifted_softplus", choices=["", "lsin", "larctan", "lshifted_softplus"])
args = parser.parse_args()

#Generate LV predator-prey data
#dx/dt=alpha*x-beta*x*y
#dy/dt=delta*x*y-gamma*y

tf=14
tf_learn=3.5
N_t_train=35
N_t=int((35*tf/tf_learn))
lr=1e-3
num_epochs=10000
plot_freq=100
is_restart=False


##coefficients from https://arxiv.org/pdf/2012.07244
alpha=1.5
beta=1
gamma=3
delta=1


x0=1 
y0=1 


def pred_prey_deriv(X, t, alpha, beta, delta, gamma):
    x=X[0]
    y=X[1]
    dxdt = alpha*x-beta*x*y
    dydt = delta*x*y-gamma*y
    dXdt=[dxdt, dydt]
    return dXdt

X0=np.array([x0, y0])
t=np.linspace(0, tf, N_t)

soln_arr=scipy.integrate.odeint(pred_prey_deriv, X0, t, args=(alpha, beta, delta, gamma))

def plotter(output_dir, pred, soln_arr, epoch, loss_train, loss_test):
    # Create base directory structure
    base_dir = os.path.join(output_dir, "plots", "pred_prey")
    training_updates_dir = os.path.join(base_dir, "training_updates")
    
    # Use os.makedirs with exist_ok to avoid race conditions
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(training_updates_dir, exist_ok=True)

    # First plot: training updates
    plt.figure()
    plt.plot(t, soln_arr[:, 0].detach(), color='g')
    plt.plot(t, soln_arr[:, 1].detach(), color='b')
    plt.plot(t, pred[:, 0].detach(), linestyle='dashed', color='g')
    plt.plot(t, pred[:, 1].detach(), linestyle='dashed', color='b')
    plt.legend(['x_data', 'y_data', 'x_KAN-ODE', 'y_KAN-ODE'])
    plt.ylabel('concentration')
    plt.xlabel('time')
    plt.ylim([0, 8])
    plt.vlines(tf_learn, 0, 8)
    
    save_path_train = os.path.join(training_updates_dir, f"train_epoch_{epoch}.png")
    plt.savefig(save_path_train, dpi=200, facecolor="w", edgecolor="w", orientation="portrait")
    plt.close()
    
    # Second plot: loss curves
    plt.figure()
    plt.semilogy(torch.Tensor(loss_train), label='train')
    plt.semilogy(torch.Tensor(loss_test), label='test')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    
    save_path_loss = os.path.join(base_dir, "loss.png")
    plt.savefig(save_path_loss, dpi=200, facecolor="w", edgecolor="w", orientation="portrait")
    plt.close()

def plotter_opt(output_dir, pred, soln_arr, epoch, loss_train, loss_test):
    # Create optimal directory
    optimal_dir = os.path.join(output_dir, "plots/pred_prey/optimal")
    if not os.path.exists(optimal_dir):
        os.makedirs(optimal_dir)

    # Plot figure
    plt.figure()
    plt.plot(t, soln_arr[:, 0].detach(), color='g')
    plt.plot(t, soln_arr[:, 1].detach(), color='b')
    plt.plot(t, pred[:, 0].detach(), linestyle='dashed', color='g')
    plt.plot(t, pred[:, 1].detach(), linestyle='dashed', color='b')

    plt.legend(['x_data', 'y_data', 'x_KAN-ODE', 'y_KAN-ODE'])
    plt.ylabel('concentration')
    plt.xlabel('time')
    plt.ylim([0, 8])
    plt.vlines(tf_learn, 0, 8)
    
    # Save plot to file
    save_path_opt = os.path.join(optimal_dir, "train_trial.png")
    plt.savefig(save_path_opt, dpi=200, facecolor="w", edgecolor="w", orientation="portrait")
    plt.close()

# Create universal KAN network
def custom_init(weight):
        nn.init.kaiming_uniform_(weight, a=5 ** 0.5)
if args.model == "SKAN":
    model = SKAN_pure([2, 79, 2], basis_function=basis.__dict__[args.basis], init_method=custom_init)
elif args.model == "KAN":
    model = KAN([2, 10, 2],  # layer sizes
                grid_size=5)
elif args.model == "MLP":
    model = nn.Sequential(
        nn.Linear(2, 79),
        nn.Tanh(),
        nn.Linear(79, 2)
    )


#convery numpy training data to torch tensors: 
X0=torch.unsqueeze((torch.Tensor(np.transpose(X0))), 0)
X0.requires_grad=True
soln_arr=torch.Tensor(soln_arr)
soln_arr.requires_grad=True
soln_arr_train=soln_arr[:N_t_train, :]
t=torch.Tensor(t)
t_learn=torch.tensor(np.linspace(0, tf_learn, N_t_train))

# Add numerical stability protection
def calDeriv(t, X):
    # Add gradient clipping
    with torch.no_grad():
        X.clamp_(-100, 100)  # Limit state variable range
    dXdt = model(X)
    # Add gradient clipping
    with torch.no_grad():
        dXdt.clamp_(-100, 100)  # Limit derivative range
    return dXdt

loss_list_train=[]
loss_list_test=[]
#initialize ADAM optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

data_dict = {
    'epoch': [],
    'loss_train': [],
    'loss_test': []
}

# if is_restart==True:
#     model.load_ckpt(args.output_dir + '/ckpt_predprey')

loss_min=1e10 #arbitrarily large to overwrite later
opt_plot_counter=0

epoch_cutoff=10 #start at smaller lr to initialize, then bump it up

#p1=model.layers[0].spline_weight
#p2=model.layers[0].base_weight
#p3=model.layers[1].splin9e_weight
#p4=model.layers[1].base_weight
best_loss = float('inf')
save_freq = 1000  # save frequency

for epoch in tqdm(range(num_epochs)):
    opt_plot_counter+=1
    #if epoch==epoch_cutoffs[2]:
    #    model = kan.KAN(width=[2,3,2], grid=grids[1], k=3).initialize_from_another_model(model, X0_train)
    optimizer.zero_grad()

    #pred=torchodeint(calDeriv, X0, t_learn, adjoint_params=[p1, p2, p3, p4])
    pred=torchodeint(calDeriv, X0, t_learn)
    loss_train=torch.mean(torch.square(pred[:, 0, :]-soln_arr_train))
    loss_train.retain_grad()
    loss_train.backward()
    optimizer.step()
    loss_list_train.append(loss_train.detach().cpu())
    #pred_test=torchodeint(calDeriv, X0, t, adjoint_params=[])
    pred_test=torchodeint(calDeriv, X0, t)
    loss_list_test.append(torch.mean(torch.square(pred_test[N_t_train:,0, :]-soln_arr[N_t_train:, :])).detach().cpu())
    #if epoch ==5:
    #    model.update_grid_from_samples(X0)
    if loss_train < best_loss:
        best_loss = loss_train
        # Save best model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss
        }, os.path.join(args.output_dir, f'best_model_{args.model}.pt'))
        
        # Plot optimal model
        if opt_plot_counter >= 200:
            plotter_opt(args.output_dir, pred_test[:, 0, :], soln_arr, epoch, loss_list_train, loss_list_test)
            opt_plot_counter = 0

    print('Iter {:04d} | Train Loss {:.5f}'.format(epoch, loss_train.item()))
    ##########
    #########################make a checker that deepcopys the best loss into, like, model_optimal
    #########
    ######################and then save that one into the file, not just whatever the current one is
    if epoch % plot_freq ==0:
        plotter(args.output_dir, pred_test[:, 0, :], soln_arr, epoch, loss_list_train, loss_list_test)
    
    data_dict['epoch'].append(epoch)
    data_dict['loss_train'].append(loss_train.item())
    data_dict['loss_test'].append(loss_list_test[-1].item())
    pd.DataFrame(data_dict).to_csv(args.output_dir + '/loss_data.csv', index=False)

    # Save checkpoint
    if epoch % save_freq == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(args.output_dir, f'checkpoint_{args.model}.pt'))

        
