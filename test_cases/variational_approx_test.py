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


def linear_regression(x, a, b, x0=0.0, d=0):
    return a*x + b + d*(x >= x0)


#
# stable: p=1, obs={g, p, t}

obs_model = 't'
qed_mode = 'ITS'

print('1D case; {:s} design'.format(qed_mode))
kernel_list = [Linear(),
               Exponential(),
               Linear()*SquaredExponential(),
               SpectralMixture(Q=2)]
K = len(kernel_list)

n1, n2 = 80, 20
n = n1 + n2
x0 = 0.0
d = 3.0
xmin, xmax = -3, 1

x = np.concatenate((np.sort(np.random.uniform(low=xmin, high=x0, size=n1)),
                    np.sort(np.random.uniform(low=x0, high=xmax, size=n2))), axis=0)


k1 = Linear()*SquaredExponential(lengthscales=1.0)
k2 = Linear()*SquaredExponential(lengthscales=0.2)
k = ChangePoints([k1, k2], locations=[x0], steepness=10)

f = np.random.multivariate_normal(mean=np.zeros((n)), cov=k.K(x[:, None]))


if obs_model is 'g':
    sigma = 0.2
    y = np.random.normal(loc=f, scale=sigma)
    likelihood = Gaussian()
elif obs_model is 'p':
    link_function = np.exp
    y = np.random.poisson(lam=link_function(f))
    likelihood = Poisson()
elif obs_model is 't':
    sigma = 0.2
    y = f + sigma*np.random.standard_t(df=3, size=n)
    likelihood = StudentT()
else:
    raise NotImplementedError('Unknown observation model')

# Default mean function is Constant := m(x) = c
qed = BNQD(data=(x, y),
           likelihood=likelihood,
           kern_list=kernel_list,
           intervention_pt=x0,
           qed_mode=qed_mode,
           variational_hyperparams=True)
qed.train()
print(qed.get_results())

fig, axes = plt.subplots(nrows=K,
                         ncols=3,
                         figsize=(14, K*3), sharey=True)

colors = cm.get_cmap('tab10', 10)

pad = 1.0

for k in range(K):
    plot_m0(bnqd_obj=qed,
            pred_range=(xmin, xmax),
            ax=axes[k, 0],
            kernel=k,
            plot_opts=dict(color=colors(k), padding=pad))
    plot_m1(bnqd_obj=qed,
            pred_range=(xmin, xmax),
            ax=axes[k, 1],
            kernel=k,
            plot_opts=dict(color=colors(k), padding=pad))
    axes[k, 0].set_ylabel(kernel_list[k].name.capitalize())

    # we want to compare the predictions by M0 and M1, in the x>=x0 regime
    # In terms of length-scales, the smallest length-scale will also apply to the x<x0 regime,
    # but with an increase in output variance. Consequently, we see a good prediction/fit,
    # but still find strong evidence in favor of a discontinuity; the error is in x<x0.

    plot_m0(bnqd_obj=qed,
            pred_range=(x0, xmax),
            ax=axes[k, 2],
            kernel=k,
            plot_opts=dict(color='darkred', padding=pad))
    plot_m1(bnqd_obj=qed,
            pred_range=(x0, xmax),
            ax=axes[k, 2],
            kernel=k,
            plot_opts=dict(color='firebrick', padding=pad))
    axes[k, 2].plot(x[x >= x0], y[x >= x0], '.', c='k')


for i, ax in enumerate(axes[:, [0, 1]].flatten()):
    ax.plot(x, y, '.', c='k')
    ax.axvline(x=x0, ls=':', c='k', lw=2.0)
    ax.set_xlim([xmin, xmax+pad])
for ax in axes[:, 2]:
    ax.set_xlim([x0, xmax+pad])

for ax in axes[-1, [0, 1]]:
    ax.set_xlabel('x')
axes[-1, -1].set_xlabel('d')

axes[0, 0].set_title('Continuous')
axes[0, 1].set_title('Discontinuous')
plt.suptitle('{:s} observations'.format(likelihood.name.capitalize()))
plt.tight_layout()
plt.show()