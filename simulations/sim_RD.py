import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import gpflow as gpf
import rdd
import pandas as pd

from matplotlib import cm
from BNQD import BNQD
from gpflow.kernels import Polynomial, Exponential, Matern32, SquaredExponential
from gpflow.likelihoods import Gaussian
from utilities import plot_m0, plot_m1, plot_effect_size, renormalize


plt.rc('axes', titlesize=24)        # fontsize of the axes title
plt.rc('axes', labelsize=18)        # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)       # fontsize of the tick labels
plt.rc('ytick', labelsize=12)       # fontsize of the tick labels
plt.rc('legend', fontsize=20)       # legend fontsize
plt.rc('figure', titlesize=32)      # fontsize of the figure title

print('GPflow version    ', gpf.__version__)
print('BNQD version      ', BNQD.__version__)

# TODO: simulations for BNDD, Rischard's two-step approach, linear RD and optimized linear RD
# TODO: important: show small n regime, where spike-and-slab BNDD regularization is most promising


def bndd_bayes_factor(x, y, x0, kernels):
    bndd = BNQD((x, y),
                likelihood=Gaussian(),
                kern_list=kernels,
                intervention_pt=x0,
                qed_mode='RD')
    bndd.train()
    logBF = bndd.get_bayes_factor()
    return logBF


def Rischard_pval_and_effect_size(x, y, es):
    pass


def freq_pval_and_effect_size(x, y):
    pass


def freq_bw_opt_pval_and_effect_size(x, y):
    pass


def simulate_data(xmin, xmax, f, x0, d, sigma):
    x = np.linspace(xmin, xmax, num=n)
    y = f(x) + sigma * np.random.normal(size=len(x)) + d*(x >= x0)
    return x, y


def plot_data(x, y, x0=None, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.plot(x, y, 'ko')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    if x0 is not None:
        ax.axvline(x=x0, ls='--', c='k')
#


functions = [lambda x: 2.3 + 0.75*x, lambda x: np.sin(x)]
kernels = [Polynomial(degree=1), Exponential(), Matern32(), SquaredExponential()]
K = len(kernels)

bf_threshold = 3.0
pval_threshold = 0.05

snr_levels = np.logspace(-3, 2, num=6, base=2)
sample_sizes = np.linspace(10, 100, num=2)

x0 = 0.0
xmin, xmax = -3, 3

nruns = 4

logBFs          = np.zeros((len(functions), len(snr_levels), len(sample_sizes), nruns, len(kernels)))
twostep_pvals   = np.zeros((len(functions), len(snr_levels), len(sample_sizes), nruns, len(kernels)))
freq_pvals      = np.zeros((len(functions), len(snr_levels), len(sample_sizes), nruns))
opt_freq_pvals  = np.zeros((len(functions), len(snr_levels), len(sample_sizes), nruns))

for i, f in enumerate(functions):
    print('Function {:d}'.format(i))
    for j, snr_level in enumerate(snr_levels):
        print('SnR level {:d}'.format(j))
        for k, n in enumerate(sample_sizes):
            print('Sample size {:d}'.format(k))
            for run in tqdm(range(nruns)):
                n = int(n)
                d_true = 1.0
                sigma = np.sqrt(d_true / snr_level)
                x, y = simulate_data(xmin=xmin, xmax=xmax, f=f, x0=x0, d=d_true, sigma=sigma)

                logBFs[i, j, k, run, :] = bndd_bayes_factor(x, y, x0, kernels)







## todo: make regression plot and compute effect sizes for:
## - GP
## - linear RD with all data
## - linear RD with optimized bandwidth

qed = BNQD((x, y),
           likelihood=Gaussian(),
           kern_list=kernels,
           intervention_pt=x0,
           qed_mode='RD')
qed.train()
bndd_results = qed.get_results()

fig, axes = plt.subplots(nrows=K,
                         ncols=3,
                         figsize=(12, K*3),
                         sharex='col',
                         sharey='col')

for i, ax in enumerate(axes[:, [0, 1]].flatten()):
    ax.plot(x, y, 'o', c='k')
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
    axes[k, 0].set_ylabel(kernels[k].name.capitalize())
    axes[k, 2].scatter(d_true, 0,
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

# manually set correct limits on effect size plots...
axes[-1, -1].set_xlim([-2, 5])
axes[-1, -1].set_ylim([0, 0.7])

axes[0, 0].set_title('Continuous')
axes[0, 1].set_title('Discontinuous')
plt.suptitle('BNDD approach')
plt.tight_layout()
plt.show()


es_total_bma = qed.get_total_bma_effect_sizes_mcmc()
es_total_m1 = qed.get_m1_bma_effect_sizes_mcmc()

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), sharey=True, sharex=True)
axes[0].hist(es_total_m1, bins=50, density=True)
axes[0].set_title('$p(d|M1)$ BMA over kernels')
axes[0].axvline(x=d_true, c='k')
axes[1].hist(es_total_bma, bins=50, density=True)
axes[1].set_title('$p(d)$ BMA over kernels')
axes[1].axvline(x=d_true, c='k')
plt.show()
