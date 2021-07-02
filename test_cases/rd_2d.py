import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from BNQD import BNQD
from gpflow.likelihoods import Gaussian, Poisson, StudentT
from gpflow.kernels import SquaredExponential, Linear, Sum

from kernels import SpectralMixture

import tensorflow as tf
import gpflow as gpf
from utilities import plot_m0, plot_m1, plot_effect_size

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

obs_model = 'g'

print('MOGP 2D RD')
kernel_list = [Linear(),
               Linear()+SquaredExponential(),
               SquaredExponential(),
               SpectralMixture(Q=3)]
K = len(kernel_list)

# See: Frölich, M. & Huber, M. Including Covariates in the Regression Discontinuity Design. J. Bus. Econ. Stat. 37, 736–748 (2019).

n = 1000
p = 4
x0 = 0.0
alpha = 0.0  # 0.0, 0.2
beta = 0.4  # 0.4, 0.4

x, U, V, W = np.random.multivariate_normal(mean=np.zeros(p), cov=np.identity(p), size=n).T
x = np.sort(x)

D = 1*(x > x0)

X1 = alpha*D + 0.5*U
X2 = alpha*D + 0.5*V
y = D+0.5*x - 0.25*D*x + 0.25*x**2 + beta*(X1+X2) + beta/2*(X1**2 + X2**2) + W


fig, ax = plt.subplots(1, 1, figsize=(8, 4))

ax.plot(x, y, 'k.')
ax.plot(x, X1, 'r--', alpha=0.4)
ax.plot(x, X2, 'r-.', alpha=0.4)
ax.axvline(x=x0, ls=':')

plt.show()

# if obs_model is 'g':
#     sigma = 1.0
#     y = np.random.normal(loc=f, scale=sigma)
#     likelihood = Gaussian()
# elif obs_model is 'p':
#     link_function = np.exp
#     y = np.random.poisson(lam=link_function(f))
#     likelihood = Poisson()
# elif obs_model is 't':
#     sigma = 1.0
#     y = f + sigma*np.random.standard_t(df=3, size=n)
#     likelihood = StudentT()
# else:
#     raise NotImplementedError('Unknown observation model')

likelihood = Gaussian()
xmin, xmax = np.min(x), np.max(x)

# Default mean function is Constant := m(x) = c
qed = BNQD(data=(x, y),
           likelihood=likelihood,
           kern_list=kernel_list,
           intervention_pt=x0,
           qed_mode='RD')
qed.train()
print(qed.get_results())

fig, axes = plt.subplots(nrows=K,
                         ncols=3,
                         figsize=(12, K*3),
                         sharex='col',
                         sharey='col')

for i, ax in enumerate(axes[:, [0, 1]].flatten()):
    ax.plot(x, y, 'o', c='k')
    ax.axvline(x=x0, ls=':', c='k', lw=2.0)
    ax.set_xlim([xmin,xmax])

colors = cm.get_cmap('tab10', 10)

for k in range(K):
    plot_m0(bnqd_obj=qed,
            pred_range=(xmin, xmax),
            ax=axes[k, 0],
            kernel=k,
            plot_opts=dict(color=colors(k)))
    plot_m1(bnqd_obj=qed,
            pred_range=(xmin, xmax),
            ax=axes[k, 1],
            kernel=k,
            plot_opts=dict(color=colors(k)))
    plot_effect_size(bnqd_obj=qed,
                     kernel=k,
                     ax=axes[k, 2],
                     plot_opts=dict(color=colors(k)))
    axes[k, 0].set_ylabel(kernel_list[k].name.capitalize())
    # axes[k, 2].scatter(d, 0,
    #                    c='lightgrey',
    #                    edgecolors='black',
    #                    marker='o',
    #                    s=150,
    #                    zorder=10,
    #                    clip_on=False,
    #                    label='True discontinuity')

for ax in axes[-1, [0, 1]]:
    ax.set_xlabel('x')
axes[-1, -1].set_xlabel('d')

axes[0, 0].set_title('Continuous')
axes[0, 1].set_title('Discontinuous')
plt.suptitle('{:s} observations'.format(likelihood.name.capitalize()))
plt.tight_layout()
plt.show()
