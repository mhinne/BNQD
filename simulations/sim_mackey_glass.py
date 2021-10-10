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

class MackeyGlass():

    def __init__(self, a, b, tau, n_power=10):
        """

        @param a:
        @param b:
        @param c:
        @param tau:
        @param x_range (tuple of [start, end] ) start and ending locations on the x axis
        @param x0 (float) discontinuity input location
        """

        self.a = a
        self.theta = theta
        self.n_power = n_power
        self.tau = tau
        self.gamma = gamma

    def simulate(self, h, x_start, x_end, x_disc, tau_disc):
        """
        Simulate Mackey Glass dynamical system from x_start to x_end with timesteps of size h,
        with a discontinutiy in theta at timepoint x_disc of new value tau_disc.
        @param h (float) size of the timesteps
        @param x_start (float) start point of simulation
        @param x_end (float) ending point of simulation
        @param x_disc (float) x location of the discontinuity in the system
        @param tau_disc_size: (float) new value of tau after the discontinuity
        @return: Y (array) simulated timeseries of length N
        """
        N = (x_end - x_start) / h
        Y = np.zeros((N))

        for i in range(N+1):
            f
            Y[i+1] = Y[i] + ()/6
        pass

    def f(self, Y, t):
        return a*


    def get_samples(self, simulated_timeseries, X_locations):
        pass


##
a = 0.2
b = 0.1
tau = 17


X = np.linspace
mackey_glass = MackeyGlass(beta, theta, n_power, tau, gamma)
