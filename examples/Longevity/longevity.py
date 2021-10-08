import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import gpflow as gpf
from rdd import rdd
import pandas as pd

from matplotlib import cm
from BNQD import BNQD
from gpflow.kernels import Polynomial, Exponential, Matern32, SquaredExponential
from utilities import plot_m0, plot_m1

plt.rc('axes', titlesize=24)        # fontsize of the axes title
plt.rc('axes', labelsize=20)        # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)       # fontsize of the tick labels
plt.rc('ytick', labelsize=14)       # fontsize of the tick labels
plt.rc('legend', fontsize=20)       # legend fontsize
plt.rc('figure', titlesize=32)      # fontsize of the figure title

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})

print('GPflow version    ', gpf.__version__)
print('BNQD version      ', BNQD.__version__)


##############################

def abline(ax, slope, intercept):
    """Plot a line from slope and intercept"""
    x_vals = np.array(ax.get_xlim())
    y_vals = intercept + slope * x_vals
    ax.plot(x_vals, y_vals, lw=2)


def get_bnqd_results(bnqd):
    results = dict()

    results['kernels'] = bnqd.kernels
    results['kernel_hyperparameters'] = bnqd.get_hyperparameters()
    results['d_bma_monte_carlo'] = bnqd.get_bma_effect_size_monte_carlo()
    results['d_m1_exact'] = bnqd.discontinuous_effect_size_mean_var()
    results['Bayes factor'] = bnqd.get_bayes_factor()
    results['marginal likelihoods'] = bnqd.get_marginal_likelihoods()
    results['BMA Bayes factor'] = bnqd.get_bma_bayes_factor()
    results['posterior model probabilities'] = bnqd.get_model_posterior()

    return results


#
def harmonic_mean_estimator(scores):
    return 1. / np.mean(1. / scores)


def run_frequentist_baseline(x, y, x0=0.0):
    results = dict()
    data_df = pd.DataFrame({'y': y, 'x': x})

    rdd_baseline = rdd.rdd(data_df, 'x', 'y', cut=x0, verbose=False)
    rdd_baseline_fit = rdd_baseline.fit()

    results['raw_effect_size'] = rdd_baseline_fit.params['TREATED']
    results['raw_pval'] = rdd_baseline_fit.pvalues['TREATED']

    bandwidth_opt = rdd.optimal_bandwidth(data_df['y'], data_df['x'], cut=x0)
    data_opt = rdd.truncated_data(data_df, 'x', bandwidth_opt, cut=x0)
    rdd_baseline_opt = rdd.rdd(data_opt, 'x', 'y', cut=x0, verbose=False)
    rdd_baseline_opt_fit = rdd_baseline_opt.fit()

    results['opt_effect_size'] = rdd_baseline_opt_fit.params['TREATED']
    results['opt_pval'] = rdd_baseline_opt_fit.pvalues['TREATED']
    results['opt_bandwidth'] = bandwidth_opt

    return results, rdd_baseline.fit(), rdd_baseline_opt.fit()


save_output_to_pdf = True

# Politics & longevity, preprocessing by Gelman

df = pd.read_csv(r'Gelman_processed_data.csv')

x = df['margin']
y = df['more_years']

x_nan = np.isnan(x)
x = x[~x_nan]
y = y[~x_nan]

y_nan = np.isnan(y)
x = x[~y_nan]
y = y[~y_nan]

cut_point = 0.0

cov_functions = [Polynomial(degree=1), Exponential(), Matern32()]
K = len(cov_functions)

bndd = BNQD((x, y),
            kern_list=cov_functions,
            intervention_pt=cut_point,
            qed_mode='RD')
bndd.train()

print(bndd.get_results())

xpred = np.linspace(-10, 10, 101)

fig, axes = plt.subplots(nrows=K+1, ncols=2, figsize=(10, 10), sharex=True, sharey=True)
for k in range(K):
    plot_m0(bndd, kernel=k, pred_range=(-10, 10), ax=axes[k, 0])
    plot_m1(bndd, kernel=k, pred_range=(-10, 10), ax=axes[k, 1])
    axes[k, 0].set_ylabel(cov_functions[k].name.capitalize())


freq_result, raw_rdd, opt_rdd = run_frequentist_baseline(x, y, cut_point)


def abline(ax, a, b, d, lw=2):
    xpred_pre = xpred[xpred < cut_point]
    xpred_post = xpred[xpred >= cut_point]

    y_vals_pre = b + a*xpred_pre
    y_vals_post = b + a*xpred_post + d
    ax.plot(xpred_pre, y_vals_pre, lw=lw, c='r')
    ax.plot(xpred_post, y_vals_post, lw=lw, c='r')

    # plot fit at x0
    ax.scatter([cut_point, cut_point], [b + a*cut_point, b + a*cut_point + d],
               c='lightgrey',
               edgecolors='black',
               marker='o', s=150, zorder=10)


# get linear regression (+ d) for RD
a_raw, b_raw, d_raw = raw_rdd.params['x'], raw_rdd.params['Intercept'], raw_rdd.params['TREATED']
a_opt, b_opt, d_opt = opt_rdd.params['x'], opt_rdd.params['Intercept'], opt_rdd.params['TREATED']

bw = freq_result['opt_bandwidth']

abline(axes[-1, 0], a=a_raw, b=b_raw, d=d_raw)
abline(axes[-1, 1], a=a_opt, b=b_opt, d=d_opt)
axes[-1, 1].axvspan(xmin=cut_point-bw, xmax=cut_point+bw, color='lightgrey', alpha=0.6)

axes[-1, 0].set_ylabel('Freq RD')
axes[-1, 0].set_title('All data', fontsize=12)
axes[-1, 1].set_title('Optimized bandwidth = {:0.2f}'.format(bw), fontsize=12)
for ax in axes.flatten():
    ax.plot(x, y, 'k.')
    ax.set_xlim([-10, 10])
    ax.axvline(x=cut_point, ls='--', c='k', lw=2)

plt.show()

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), sharex=True, sharey=True)
plot_m0(bndd, kernel=2, pred_range=(-10, 10), ax=axes[0], plot_opts={'color': 'steelblue',
                                                                     'linewidth': 4,
                                                                     'label': '$\mathcal{M}_0$'})
plot_m1(bndd, kernel=2, pred_range=(-10, 10), ax=axes[0], plot_opts={'color': 'indianred',
                                                                     'linewidth': 4,
                                                                     'linestyle': (0, (5, 5)),
                                                                     'label': '$\mathcal{M}_1$'})
axes[0].set_title('Matérn ($\\nu=3/2$)', fontsize=24)
axes[0].set_ylabel('Years alive post election', fontsize=22)
axes[0].legend(fancybox=False, framealpha=1.0, ncol=2, loc='lower center')
abline(axes[1], a=a_opt, b=b_opt, d=d_opt, lw=4)
axes[1].axvspan(xmin=cut_point-bw, xmax=cut_point+bw, color='lightgrey', alpha=0.6)
axes[1].set_title('Linear RD with opt. bw. = {:0.2f}'.format(bw), fontsize=24)

for ax in axes:
    ax.set_xlabel('Percentile pts', fontsize=22)
    ax.plot(x, y, 'k.')
    ax.set_xlim([-10, 10])
    ax.axvline(x=cut_point, ls='--', c='k', lw=2)
    ax.set_xticks([-10, -5, 0.0, 5, 10])
    ax.tick_params(axis='x', labelsize=18)
axes[0].tick_params(axis='y', labelsize=18)
plt.tight_layout()
if save_output_to_pdf:
    plt.savefig('longevity.pdf', bbox_inches='tight', pad_inches=0)
plt.show()
