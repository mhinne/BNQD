import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import gpflow as gpf

from BNQD import BNQD
from gpflow.kernels import SquaredExponential, Linear
from gpflow.likelihoods import Gaussian
from gpflow.utilities import print_summary

plt.rc('axes', titlesize=24)        # fontsize of the axes title
plt.rc('axes', labelsize=18)        # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)       # fontsize of the tick labels
plt.rc('ytick', labelsize=12)       # fontsize of the tick labels
plt.rc('legend', fontsize=20)       # legend fontsize
plt.rc('figure', titlesize=32)      # fontsize of the figure title

print('TensorFlow version  ', tf.__version__)
print('NumPy version       ', np.__version__)
print('GPflow version      ', gpf.__version__)
print('BNQD version        ', BNQD.__version__)


def f_1(x):
    return np.sin(6*x)


jitter = 1e-6

# CASE 1: only one variable is affected by the intervention
# CASE 2: multiple variables are affected by the intervention


x0 = 0.5
xmin, xmax = 0, 1
n_1, n_2 = 50, 50
sigma = 0.4
d = 1.7

# We could simply have x_1==x_2, but in the general case we might observe for different inputs
x_1 = np.linspace(xmin, xmax, n_1)
x_2 = np.linspace(xmin, xmax, n_2)

mu_1 = f_1(x_1)
mu_2 = mu_1 + 0.05*x_2**2 + 0.5 + d*(x_2 > x0)

y_1 = np.random.normal(loc=mu_1, scale=sigma)
y_2 = np.random.normal(loc=mu_2, scale=sigma)

output_dim = 2
rank = output_dim

kernel_list = [Linear(active_dims=[0]), SquaredExponential(active_dims=[0])]
data = [(x_1, y_1), (x_2, y_2)]
p = len(data)

kernel_names = ['Linear', 'Sq Exp']
mo_rdd = BNQD(data=data,
              likelihood=Gaussian(),
              kern_list=kernel_list,
              intervention_pt=x0,
              forcing_variable=0)
mo_rdd.train()

n = 201
x_new = np.linspace(0, 1, num=n)
predictions = mo_rdd.predict_y(x_new)
ix_intv = np.argmin((x_new-x0)**2)

epsilon = 1e-6
pred_at_x0 = mo_rdd.predict_y(np.array([x0 - epsilon, x0 + epsilon]))

colors = ['o', 'b']

fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(8, 3))
for k, ax in enumerate(axes):
    for i in range(p):
        (dat,) = ax.plot(data[i][0], data[i][1], '.', label='$x_{:d}$'.format(i+1))
        m0, m1 = predictions[i][k]
        m0_mu, m0_var = m0
        m1_mu, m1_var = m1
        m1_mu0 = m1_mu[:, 0].numpy()
        m1_mu1 = m1_mu[:, 1].numpy()

        m1_mu0[ix_intv] = np.nan
        m1_mu1[ix_intv] = np.nan

        ax.plot(x_new, m0_mu[:, 0], ls='--', color=dat.get_color())
        ax.plot(x_new, m0_mu[:, 1], ls='--', color=dat.get_color())
        ax.plot(x_new, m1_mu0, color=dat.get_color())
        ax.plot(x_new, m1_mu1, color=dat.get_color())
        ax.plot([x0, x0], [pred_at_x0[i][k][1][0][0], pred_at_x0[i][k][1][0][1]],
                'o', linestyle='none', markeredgecolor='k',
                markersize=10, color=dat.get_color(), zorder=99)

    ax.axvline(x=x0, ls=':', color='k')
    ax.set_title(kernel_list[k].name.capitalize().replace('_', ' '))
    ax.set_xlabel('x')
    ax.set_xlim([xmin, xmax])

plt.show()
