import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from BNQD import BNQD
import gpflow
from gpflow.likelihoods import Gaussian, Poisson, StudentT, SwitchedLikelihood
from gpflow.kernels import SquaredExponential, \
    Matern32, \
    Exponential, \
    Linear, \
    Constant, \
    Sum, \
    ChangePoints, \
    SeparateIndependent, \
    Product, \
    Cosine, \
    Periodic, \
    Polynomial

from gpflow.utilities import print_summary, deepcopy
from kernels import SpectralMixture
from gpflow.utilities import print_summary
import tensorflow as tf
import gpflow as gpf
from utilities import plot_m0, plot_m1, plot_effect_size, split_data

plt.rc('axes', titlesize=24)        # fontsize of the axes title
plt.rc('axes', labelsize=18)        # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)       # fontsize of the tick labels
plt.rc('ytick', labelsize=12)       # fontsize of the tick labels
plt.rc('legend', fontsize=20)       # legend fontsize
plt.rc('figure', titlesize=32)      # fontsize of the figure title

print('TensorFlow version', tf.__version__)
print('GPflow version    ', gpf.__version__)
print('BNQD version      ', BNQD.__version__)


def signal(x, x0=0):
    return 2*((x <= x0)*(np.sin(4*x) + np.cos(3*x)) + (x>x0)*(np.sin(12*x) + np.cos(3*x)))


def plot_fit(x, mean, var, color, label, ax=None):
    if ax is None:
        ax = plt.gca()

    x = x.flatten()
    mean = mean.numpy().flatten()
    intv = 1.96*np.sqrt(var.numpy().flatten())
    ax.plot(x, mean, c=color, label=label)
    ax.fill_between(x, mean + intv, mean - intv, color=color, alpha=0.2)


#


x0 = 0
n, nf = 50, 200
sigma = 0.8
xf = np.linspace(-np.pi, np.pi, num=nf)
x = np.linspace(-np.pi, np.pi, num=n)
f = signal(xf, x0=0)
y = np.random.normal(loc=signal(x, x0=0), scale=sigma)

plt.figure(figsize=(12, 6))
ax = plt.gca()
ax.plot(xf, f, c='k', label='True signal')
ax.scatter(x, y, c='r', label='Obs')
ax.set_xlim([-np.pi, np.pi])
ax.axvline(x=x0, c='k', ls='--')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
plt.show()

likelihood = Gaussian()
kernel_list = [SpectralMixture(Q=2)]
# kernel_list = [SquaredExponential()]

qed = BNQD(data=(x, y),
           likelihood=likelihood,
           kern_list=kernel_list,
           intervention_pt=x0,
           qed_mode='ITS')
qed.train()

x_new = np.atleast_2d(xf[xf >= x0]).T

(m0_mu, m0_var), (m1_mu, m1_var) = qed.predict_y(xf)[0]

num_samples = 10
m1_A_mu, m1_A_var = qed.counterfactual_y(x_new)[0]
m1_A_samples = qed.counterfactual_f_samples(x_new, num_samples=num_samples)[0]


plt.figure(figsize=(12, 6))
ax = plt.gca()
plot_fit(xf, m0_mu, m0_var, color='g', label='m0', ax=ax)
plot_fit(xf, m1_mu, m1_var, color='r', label='m1', ax=ax)
plot_fit(x_new, m1_A_mu, m1_A_var, color='C1', label='m1_A', ax=ax)
ax.plot(x_new, m1_A_samples[:, :, 0].numpy().T, color='C1', lw=1, ls=':')
ax.plot(xf, f, c='k', label='True signal')
ax.scatter(x, y, c='k', label='Obs')
ax.set_xlim([-np.pi, np.pi])
ax.axvline(x=x0, c='k', ls='--')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(fontsize=14)
plt.show()



