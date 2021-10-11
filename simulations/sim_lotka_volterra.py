import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import gpflow as gpf
import pandas as pd
import scipy.stats as stats
from matplotlib import cm
from BNQD import BNQD
from gpflow.kernels import Polynomial, Exponential, Matern32, SquaredExponential
from gpflow.likelihoods import Gaussian
from simulations.Dynamical_Systems import *
from kernels import SpectralMixture

plt.rc('axes', titlesize=26)  # fontsize of the axes title
plt.rc('axes', labelsize=20)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=16)  # fontsize of the tick labels
plt.rc('ytick', labelsize=16)  # fontsize of the tick labels
plt.rc('legend', fontsize=22)  # legend fontsize
plt.rc('figure', titlesize=28)  # fontsize of the figure title

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})

print('GPflow version    ', gpf.__version__)
print('BNQD version      ', BNQD.__version__)

# Lotka volterra parameters
alpha = 1.1
beta = 0.4
delta = 0.1
gamma = 0.4
init_pos = [10,10]

# Integration and discontinuity parameters
N_samples = 200
x_start, x_end = 0, 100
x0 = 50
alpha_disc = alpha
h = 1e-3

# Simulate and visualize data
lotka_volterra = LotkaVolterra(alpha, beta, delta, gamma, x0, alpha_disc)
time_series = lotka_volterra.simulate_rk45(init_pos=init_pos, x_start=x_start, x_end=x_end, h=h)
X, Y = lotka_volterra.get_random_samples(Y=time_series, N=N_samples, x_start=x_start, x_end=x_end, h=h)

fig, ax = lotka_volterra.plot_timeseries(X, Y, x0)
fig.suptitle(f"Lotka volterra with alpha discontinuity: {alpha} to {alpha_disc} at $x_0$={x0}")
plt.show()

## Run BND design
sm = SpectralMixture(Q=2, x=X[:,0], y=Y[:,0])
print(sm.trainable_parameters)
kernels = [sm]
Y = (Y - np.mean(Y))/np.std(Y)
bndd = BNQD((X, Y),
            likelihood=Gaussian(),
            kern_list=kernels,
            intervention_pt=x0,
            qed_mode='ITS')
bndd.train()
res_df = bndd.get_results()
print(res_df)


x_end = x_end + 100
n_samples = int((x_end-x_start)/h)
X_pred = np.zeros((n_samples, 2))
for i in range(2):
    X_pred[:,i] = np.arange(x_start, x_end, step=h)
y_pred = bndd.predict_y(X_pred)
((mu0_k, var0_k), (mu1_k, var1_k)) = y_pred[0]

fig, ax = plt.subplots(2,1,figsize=(12,8))
for i in range(2):
    ax[0].scatter(X[:,i], Y[:,i], marker='x', color='black')
    ax[0].plot(X_pred[:,i], mu0_k[:,i])
    ax[0].fill_between(X_pred[:,i], mu0_k[:,i] - 1.96 * var0_k[:,i], mu0_k[:,i] + 1.96 * var0_k[:,i], alpha=0.2)
    ax[0].set_title(f'Model 0')
for i in range(2):
    ax[1].scatter(X[:,i], Y[:,i], marker='x', color='black')
    ax[1].plot(X_pred[:, i], mu1_k[:, i])
    ax[1].fill_between(X_pred[:, i], mu1_k[:, i] - 1.96 * var1_k[:, i], mu1_k[:, i] + 1.96 * var1_k[:, i],
                       alpha=0.2)
    ax[1].set_title(f'Model 1')
plt.show()





