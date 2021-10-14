import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import gpflow as gpf
import pandas as pd
import scipy.stats as stats
from matplotlib import cm
from BNQD import BNQD
from kernels import SpectralMixture
from gpflow.kernels import Polynomial, Exponential, Matern32, SquaredExponential
from gpflow.likelihoods import Gaussian
from simulations.Dynamical_Systems import *

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

# simulation parameters
a = 0.2
b = 0.1
tau = 17
tau_disc = 20
n_power = 10
init_pos = [0]

# Integration and discontinuity parameters
x_start, x_end = 0, 25
x0 = 10.
h = 1e-5
x_start_sample, x_end_sample = 5, 25
N_samples = 200

# Simulate and visualize system
mackey_glass = MackeyGlass(a, b, tau, n_power, x0, tau_disc=20)
timeseries = mackey_glass.simulate_rk45(init_pos=init_pos, x_start=x_start, x_end=x_end, h=h)
X, Y = mackey_glass.get_random_samples(Y=timeseries, N=N_samples, x_start=x_start_sample, x_end=x_end_sample, h=h)
mackey_glass.plot_timeseries(X, Y, x0)
plt.show()

fig, ax = mackey_glass.plot_timeseries(np.arange(x_start, x_end, h), timeseries, x0, X, Y)
fig.suptitle(f"Mackay-Glass with tau discontinuity: {tau} to {tau_disc} at $x_0$={x0}")
plt.show()

## Run BND design
sm = SpectralMixture(Q=2, x=X, y=Y)
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


x_end = x_end + 20
n_samples = int((x_end-x_start)/h)
X_pred = np.zeros((n_samples, 2))
for i in range(2):
    X_pred[:,i] = np.arange(x_start, x_end, step=h)
y_pred = bndd.predict_y(X_pred)
((mu0_k, var0_k), (mu1_k, var1_k)) = y_pred[0]

fig, ax = plt.subplots(2,1,figsize=(12,8))
ax[0].scatter(X, Y, marker='x', color='black')
ax[0].plot(X_pred, mu0_k)
ax[0].fill_between(X_pred, mu0_k - 1.96 * var0_k, mu0_k + 1.96 * var0_k, alpha=0.2)
ax[0].set_title(f'Model 0')

ax[1].scatter(X, Y, marker='x', color='black')
ax[1].plot(X_pred, mu1_k)
ax[1].fill_between(X_pred, mu1_k - 1.96 * var1_k, mu1_k + 1.96 * var1_k,alpha=0.2)
ax[1].set_title(f'Model 1')
plt.show()







