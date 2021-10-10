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

# Integration and discontinuity parameters
x_start, x_end = 0, 20
x0 = 10.
h = 1e-5

# Simulate and visualize system
mackey_glass = MackeyGlass(a, b, tau, n_power, x0, tau_disc=20)
Y = mackey_glass.simulate_rk45(x_start=x_start, x_end=x_end, h=h)
X, Y = mackey_glass.get_random_samples(Y=Y, N=200, sigma_x=1, x_start=x_start, x_end=x_end)
mackey_glass.plot_timeseries(X, Y, x0)
plt.show()


