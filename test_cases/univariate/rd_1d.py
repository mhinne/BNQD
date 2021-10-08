import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import scipy.stats as st
from BNQD import BNQD
from gpflow.likelihoods import Gaussian, Poisson, StudentT
from gpflow.kernels import SquaredExponential, Linear, Sum, Exponential
from gpflow.utilities import print_summary
from kernels import SpectralMixture

import tensorflow as tf
import gpflow as gpf
from utilities import plot_m0, plot_m1, plot_effect_size

plt.rc('axes', titlesize=24)  # fontsize of the axes title
plt.rc('axes', labelsize=18)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)  # fontsize of the tick labels
plt.rc('ytick', labelsize=12)  # fontsize of the tick labels
plt.rc('legend', fontsize=20)  # legend fontsize
plt.rc('figure', titlesize=32)  # fontsize of the figure title

print('TensorFlow version', tf.__version__)
print('GPflow version    ', gpf.__version__)
print('BNQD version      ', BNQD.__version__)


def linear_regression(x, a, b, x0=0.0, d=0):
    return a * x + b + d * (x >= x0)


def quadratic_regression(x, b0, b1, b2, x0=0.0, d=0):
    return b0 + b1 * x + b2 * x ** 2 + d * (x >= x0)


obs_model = 'g'
save_plot = False

print('1D case')
# kernel_list = [Linear(), Exponential(), SquaredExponential(), SpectralMixture(Q=2)]
kernel_list = [Linear(), SquaredExponential()]
K = len(kernel_list)

n = 50
x0 = 0.25
d = 0.5
xmin, xmax = -1, 1
x = np.sort(np.random.uniform(low=xmin, high=xmax, size=n))
# x = np.linspace(-1, 1, num=n)
# f = linear_regression(x, a=1.3, b=3.2, x0=x0, d=d)
f = quadratic_regression(x, b0=1.2, b1=0.2, b2=1.7, x0=x0, d=d)

if obs_model is 'g':
    sigma = 0.8
    y = np.random.normal(loc=f, scale=sigma)
    likelihood = Gaussian()
elif obs_model is 'p':
    link_function = np.exp
    y = np.random.poisson(lam=link_function(f))
    likelihood = Poisson()
elif obs_model is 't':
    sigma = 1.0
    y = f + sigma * np.random.standard_t(df=3, size=n)
    likelihood = StudentT()
else:
    raise NotImplementedError('Unknown observation model')

# Default mean function is Constant := m(x) = c
qed = BNQD(data=(x, y),
           likelihood=likelihood,
           kern_list=kernel_list,
           intervention_pt=x0,
           qed_mode='RD')
qed.train()
print('Training complete')
print(qed.get_results().to_string())

fig, axes = plt.subplots(nrows=K + 1,
                         ncols=3,
                         figsize=(12, K * 3),
                         sharex='col',
                         sharey='col')

for i, ax in enumerate(axes[0:-1, [0, 1]].flatten()):
    ax.plot(x, y, 'o', c='k', fillstyle='none')
    ax.axvline(x=x0, ls=':', c='k', lw=2.0)
    ax.set_xlim([xmin, xmax])

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
    axes[k, 2].scatter(d, 0,
                       c='lightgrey',
                       edgecolors='black',
                       marker='o',
                       s=150,
                       zorder=10,
                       clip_on=False,
                       label='True discontinuity')

for ax in axes[-1, [0, 1]]:
    ax.set_xlabel('x')
axes[-1, -1].set_xlabel('d')



d_range = np.linspace(-2, 5, num=100)
bma_kde = st.kde.gaussian_kde(qed.marginal_bma_effectsize(nsamples=100000), bw_method='silverman')
bma_density = bma_kde(d_range)
axes[-1, -1].plot(d_range, bma_density, lw=2, c='g')
axes[-1, -1].fill_between(d_range, np.zeros_like(d_range), bma_density, color='g', alpha=0.3)
axes[-1, -1].axvline(x=0, ls='--', c='k')
axes[-1, -1].scatter(d, 0,
                     c='lightgrey',
                     edgecolors='black',
                     marker='o',
                     s=150,
                     zorder=10,
                     clip_on=False,
                     label='True discontinuity')

# manually set correct limits on effect size plots...
axes[-1, -1].set_xlim([-2, 5])
axes[-1, -1].set_ylim([0, 0.7])

axes[0, 0].set_title('Continuous')
axes[0, 1].set_title('Discontinuous')
# plt.suptitle('{:s} observations'.format(likelihood.name.capitalize()))


plt.tight_layout()

if save_plot:
    plt.savefig(r'D:\SURFdrive\Teaching\Courses\2021-2022\AI for Healthcare\demo2.pdf',
                bbox_inches='tight', pad_inches=0)

plt.show()
