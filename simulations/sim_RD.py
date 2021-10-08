import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import gpflow as gpf
from rdd import rdd
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


def compute_twostep_pvals(mu, var):
    sd = np.sqrt(var)
    m = len(mu)
    pvals = np.zeros(m)
    for i in range(m):
        if mu[i] < 0:
            pvals[i] = 1 - stats.norm.cdf(x=0, loc=mu[i], scale=sd[i])
        else:
            pvals[i] = stats.norm.cdf(x=0, loc=mu[i], scale=sd[i])
    return pvals


def simulate_data(xmin, xmax, f, x0, d, sigma):
    x = np.sort(np.random.uniform(xmin, xmax, size=n))
    y = f(x, x0) + sigma * np.random.normal(size=len(x)) + d * (x >= x0)
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
def run_simulations(n,
                    nruns=100,
                    results_dir=r'D:\SURFdrive\Projects\BNQD_MOGP\simulations\results'):
    # BNDD
    logBFs = np.zeros((F, K + 1, D, nruns))
    BNDD_bma_effectsize = np.zeros((F, K + 1, D, nruns))  # add one for BMA across all kernels

    # Two-step Rischards
    BNDD_m1_effectsize = np.zeros((F, K + 1, D, nruns))  # add one for BMA across all kernels
    twostep_pvals = np.zeros((F, K, D, nruns))

    # RDD linear baseline
    freq_pvals = np.zeros((F, D, nruns))
    freq_opt_pvals = np.zeros((F, D, nruns))
    freq_effectsize = np.zeros((F, D, nruns))
    freq_opt_effectsize = np.zeros((F, D, nruns))

    file_prefix = 'n={:d}_F={:d}_K={:d}_results'.format(n, F, K)

    for run in tqdm(range(nruns)):
        runfile = results_dir + os.path.sep + file_prefix + '_run={:02d}.npz'.format(run)
        if os.path.isfile(runfile):
            with np.load(runfile) as rundata:
                logBFs_run              = rundata['logBFs_run']
                BNDD_bma_effectsize_run = rundata['BNDD_bma_effectsize_run']
                BNDD_m1_effectsize_run  = rundata['BNDD_m1_effectsize_run']
                twostep_pvals_run       = rundata['twostep_pvals_run']
                freq_pvals_run          = rundata['freq_pvals_run']
                freq_opt_pvals_run      = rundata['freq_opt_pvals_run']
                freq_effectsize_run     = rundata['freq_effectsize_run']
                freq_opt_effectsize_run = rundata['freq_opt_effectsize_run']
        else:
            try:
                # BNDD
                logBFs_run = np.zeros((F, K + 1, D))
                BNDD_bma_effectsize_run = np.zeros((F, K + 1, D))  # add one for BMA across all kernels

                # Two-step Rischard's approach
                BNDD_m1_effectsize_run = np.zeros((F, K + 1, D))  # add one for BMA across all kernels
                twostep_pvals_run = np.zeros((F, K, D))

                # RDD linear baseline
                freq_pvals_run = np.zeros((F, D))
                freq_opt_pvals_run = np.zeros((F, D))
                freq_effectsize_run = np.zeros((F, D))
                freq_opt_effectsize_run = np.zeros((F, D))

                for i, f in enumerate(functions):
                    # print(f_names[i])
                    for j, d_true in enumerate(d_trues):
                        # print(d_true)
                        x, y = simulate_data(xmin=xmin, xmax=xmax, f=f, x0=x0, d=d_true, sigma=sigma)

                        bndd = BNQD((x, y),
                                    likelihood=Gaussian(),
                                    kern_list=kernels,
                                    intervention_pt=x0,
                                    qed_mode='RD')
                        bndd.train()
                        res_df = bndd.get_results()

                        # BNDD
                        logBFs_run[i, :, j] = res_df['log BF']
                        BNDD_bma_effectsize_run[i, :, j] = res_df['E[p(d | D)]']

                        # 2-step baseline
                        BNDD_m1_effectsize_run[i, :, j] = res_df['E[p(d | D, M1)]']
                        mu = np.asarray(res_df['E[p(d | D, M1)]'][0:-1])
                        var = np.asarray(res_df['V[p(d | D, M1)]'][0:-1])
                        twostep_pvals_run[i, :, j] = compute_twostep_pvals(mu, var)

                        # RDD linear baseline with optionally optimized bandwidth (Imbens-Kalyanaraman)
                        data_df = pd.DataFrame({'y': y, 'x': x})

                        rdd_baseline = rdd.rdd(data_df, 'x', 'y', cut=x0, verbose=False)
                        rdd_baseline_fit = rdd_baseline.fit()

                        bandwidth_opt = rdd.optimal_bandwidth(data_df['y'], data_df['x'], cut=x0)
                        data_opt = rdd.truncated_data(data_df, 'x', bandwidth_opt, cut=x0)
                        rdd_baseline_opt = rdd.rdd(data_opt, 'x', 'y', cut=x0, verbose=False)
                        rdd_baseline_opt_fit = rdd_baseline_opt.fit()

                        freq_pvals_run[i, j] = rdd_baseline_fit.pvalues['TREATED']
                        freq_opt_pvals_run[i, j] = rdd_baseline_opt_fit.pvalues['TREATED']

                        freq_effectsize_run[i, j] = rdd_baseline_fit.params['TREATED']
                        freq_opt_effectsize_run[i, j] = rdd_baseline_opt_fit.params['TREATED']
                np.savez(runfile,
                         logBFs_run=logBFs_run,
                         BNDD_bma_effectsize_run=BNDD_bma_effectsize_run,
                         BNDD_m1_effectsize_run=BNDD_m1_effectsize_run,
                         twostep_pvals_run=twostep_pvals_run,
                         freq_pvals_run=freq_pvals_run,
                         freq_opt_pvals_run=freq_opt_pvals_run,
                         freq_effectsize_run=freq_effectsize_run,
                         freq_opt_effectsize_run=freq_opt_effectsize_run)
            except Exception as e:
                print(e)

        logBFs[:, :, :, run] = logBFs_run
        BNDD_bma_effectsize[:, :, :, run] = BNDD_bma_effectsize_run

        BNDD_m1_effectsize[:, :, :, run] = BNDD_m1_effectsize_run
        twostep_pvals[:, :, :, run] = twostep_pvals_run

        freq_pvals[:, :, run] = freq_pvals_run
        freq_opt_pvals[:, :, run] = freq_opt_pvals_run

        freq_effectsize[:, :, run] = freq_effectsize_run
        freq_opt_effectsize[:, :, run] = freq_opt_effectsize_run
    return logBFs, \
           BNDD_bma_effectsize, \
           BNDD_m1_effectsize, \
           twostep_pvals, \
           freq_pvals, \
           freq_opt_pvals, \
           freq_effectsize, \
           freq_opt_effectsize


#
def plot_simulation_results(logBFs,
                            BNDD_bma_effectsize,
                            BNDD_m1_effectsize,
                            twostep_pvals,
                            freq_pvals,
                            freq_opt_pvals,
                            freq_effectsize,
                            freq_opt_effectsize):
    def error_fun(a, b):
        return np.abs(a - b)

    def mean_and_intv(x):
        mu = np.mean(x, axis=0)
        lb, ub = mu - 0.5 * np.std(x, axis=0), mu + 0.5 * np.std(x, axis=0)
        return mu, lb, ub

    def plot_effectsize_result(ax, data, color, label=None, ls='-'):
        nruns = data.shape[-1]
        errors = error_fun(data.T, np.tile(d_trues, (nruns, 1)))
        mu, lb, ub = mean_and_intv(errors)
        ax.fill_between(d_trues, lb, ub, alpha=0.1, color=color)
        ax.plot(d_trues, mu, c=color, ls=ls, label=label, lw=2)

    def plot_2step_pval(ax, data, color, label=None, ls='-'):
        mu, lb, ub = mean_and_intv(data.T)
        ax.fill_between(d_trues, lb, ub, alpha=0.1, color=color)
        ax.plot(d_trues, mu, c=color, ls=ls, label=label, lw=2)

    def plot_logBF(ax, data, color, label=None, ls='-'):
        mu, lb, ub = mean_and_intv(data.T)
        ax.fill_between(d_trues, lb, ub, alpha=0.1, color=color)
        ax.plot(d_trues, mu, c=color, lw=2)

    bf_threshold = 3.0
    pval_threshold = 0.05

    error_row = 0
    logbf_row = 1
    pval_row = 2

    fig, axes = plt.subplots(nrows=3, ncols=F, sharex=True, figsize=(20, 8))
    # First row, absolute error
    for i, f in enumerate(functions):
        axes[error_row, i].set_title(f_names[i], fontsize=24)
        for k in range(K):
            # Conditional results
            plot_effectsize_result(axes[error_row, i], BNDD_m1_effectsize[i, k, :, :], color=colors[k],
                                   label='{:s}, $M_1$'.format(kernels[k].name.capitalize().replace('_', ' ')),
                                   ls=ls_2step)
            # Marginal results
            plot_effectsize_result(axes[error_row, i], BNDD_bma_effectsize[i, k, :, :], color=colors[k],
                                   label='{:s}, BMA'.format(kernels[k].name.capitalize().replace('_', ' ')),
                                   ls=ls_bndd)

        # Marginal-conditional (M1) results
        plot_effectsize_result(axes[error_row, i], BNDD_m1_effectsize[i, -1, :, :], color=colors[K], label='BMA $M_1$',
                               ls=ls_2step)

        # Hyper-marginal (total BMA) results
        plot_effectsize_result(axes[error_row, i], BNDD_bma_effectsize[i, -1, :, :], color=colors[K], label='Total BMA',
                               ls=ls_bndd)

        # frequentist results
        plot_effectsize_result(axes[error_row, i], freq_effectsize[i, :, :], color='k', label='All data',
                               ls=ls_fs_all)

        plot_effectsize_result(axes[error_row, i], freq_opt_effectsize[i, :, :], color='k', label='Opt. bw.',
                               ls=ls_fs_opt)
        # axes[error_row, i].set_ylim(bottom=0)

    # Second row, log Bayes factors
    for i, f in enumerate(functions):
        for k in range(K):
            plot_logBF(axes[logbf_row, i], data=logBFs[i, k, :, :], color=colors[k], ls=ls_bndd)
        plot_logBF(axes[logbf_row, i], data=logBFs[i, K, :, :], color=colors[K], ls=ls_bndd)
        axes[logbf_row, i].axhline(y=np.log(bf_threshold), lw=0.5, c='k', alpha=0.5)
        axes[logbf_row, i].axhline(y=-np.log(bf_threshold), lw=0.5, c='k', alpha=0.5)

    # Third row, p-values
    for i, f in enumerate(functions):
        for k in range(K):
            plot_2step_pval(axes[pval_row, i], twostep_pvals[i, k, :, :], color=colors[k], ls=ls_2step)
        plot_2step_pval(axes[pval_row, i], freq_pvals[i, :, :], color='k', ls=ls_fs_all)
        plot_2step_pval(axes[pval_row, i], freq_opt_pvals[i, :, :], color='k', ls=ls_fs_opt)
        axes[pval_row, i].axhline(y=pval_threshold, lw=0.5, c='k', alpha=0.5)
        axes[pval_row, i].set_ylim(bottom=0)

    for ax in axes.flatten():
        ax.set_xlim([d_trues[0], d_trues[-1]])

    for ax in axes[1:-1, :].flatten():
        ax.set_xticks(np.arange(0, 5))
        ax.set_xticklabels([])

    for ax in axes[-1, :]:
        ax.set_xticks(np.arange(0, 5))
        ax.set_xticklabels(np.arange(0, 5))
        ax.set_xlabel('$d$')

    axes[error_row, 0].set_ylabel('Error')
    axes[logbf_row, 0].set_ylabel('log BF')
    axes[pval_row, 0].set_ylabel('$p$-value')

    handles, labels = axes[error_row, 0].get_legend_handles_labels()

    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.figlegend(handles, labels,
                  loc='lower center',
                  ncol=K + 2,
                  fontsize=14,
                  frameon=False,
                  bbox_to_anchor=(0.5, 0.0))
    if save_output_to_pdf:
        plt.savefig('simulation_results.pdf', bbox_inches='tight', pad_inches=0)
    plt.show()


#
def run_visualisation_example():
    def lin_reg(a, b, d):
        xpred_pre = x_range[x_range < x0]
        xpred_post = x_range[x_range >= x0]

        y_vals_pre = b + a * xpred_pre
        y_vals_post = b + a * xpred_post + d
        return np.concatenate((y_vals_pre, y_vals_post), axis=0)

    examplefile = r'D:\SURFdrive\Projects\BNQD_MOGP\simulations\examples.npz'

    d_range = [0.0, 2.0, 4.0]
    num_data = 100
    num_pred = 101
    x_range = np.linspace(xmin, xmax, num=101)

    if os.path.isfile(examplefile):
        with np.load(examplefile) as example_data:
            data_x = example_data['data_x'],
            data_y = example_data['data_y'],
            predictions_m0_mu       = example_data['predictions_m0_mu']
            predictions_m1_mu       = example_data['predictions_m1_mu']
            predictions_m0_var      = example_data['predictions_m0_var']
            predictions_m1_var      = example_data['predictions_m1_var']
            predictions_rdd_raw     = example_data['predictions_rdd_raw']
            predictions_rdd_opt     = example_data['predictions_rdd_opt']
            effectsizes_m1          = example_data['effectsizes_m1']
            effectsizes_bma         = example_data['effectsizes_bma']
            effectsizes_total_bma   = example_data['effectsizes_total_bma']
    else:
        predictions_m0_mu = np.zeros((len(d_range), F, K, num_pred))
        predictions_m1_mu = np.zeros((len(d_range), F, K, num_pred))
        predictions_m0_var = np.zeros((len(d_range), F, K, num_pred))
        predictions_m1_var = np.zeros((len(d_range), F, K, num_pred))

        predictions_rdd_raw = np.zeros((len(d_range), F, num_pred))
        predictions_rdd_opt = np.zeros((len(d_range), F, num_pred))

        effectsizes_m1 = np.zeros((len(d_range), F, K))
        effectsizes_bma = np.zeros((len(d_range), F, K))
        effectsizes_total_bma = np.zeros((len(d_range), F))

        data_x = np.zeros((len(d_range), F, num_data))
        data_y = np.zeros((len(d_range), F, num_data))

        for j, d in enumerate(d_range):
            for i, f in enumerate(functions):
                x, y = simulate_data(xmin=xmin, xmax=xmax, f=f, x0=x0, d=d, sigma=sigma)
                data_x[j, i, :] = x
                data_y[j, i, :] = y

                # BNDD predictions
                bndd = BNQD((x, y),
                            likelihood=Gaussian(),
                            kern_list=kernels,
                            intervention_pt=x0,
                            qed_mode='RD')
                bndd.train()
                predictions = bndd.predict_y(x_range)
                for k, kernel in enumerate(kernels):
                    predictions_m0_mu[j, i, k, :] = predictions[k][0][0].numpy().flatten()
                    predictions_m0_var[j, i, k, :] = predictions[k][0][1].numpy().flatten()
                    predictions_m1_mu[j, i, k, :] = predictions[k][1][0].numpy().flatten()
                    predictions_m1_var[j, i, k, :] = predictions[k][1][1].numpy().flatten()
                res_df = bndd.get_results()

                effectsizes_m1[j, i, :] = res_df['E[p(d | D, M1)]'][0:-1]
                effectsizes_bma[j, i, :] = res_df['E[p(d | D)]'][0:-1]
                effectsizes_total_bma[j, i] = res_df['E[p(d | D)]'][-1]

                # frequentist predictions

                # RDD linear baseline with optionally optimized bandwidth (Imbens-Kalyanaraman)
                data_df = pd.DataFrame({'y': y, 'x': x})

                rdd_baseline = rdd.rdd(data_df, 'x', 'y', cut=x0, verbose=False)
                rdd_baseline_fit = rdd_baseline.fit()

                bandwidth_opt = rdd.optimal_bandwidth(data_df['y'], data_df['x'], cut=x0)
                data_opt = rdd.truncated_data(data_df, 'x', bandwidth_opt, cut=x0)
                rdd_baseline_opt = rdd.rdd(data_opt, 'x', 'y', cut=x0, verbose=False)
                rdd_baseline_opt_fit = rdd_baseline_opt.fit()

                a_raw, b_raw, d_raw = rdd_baseline_fit.params['x'], \
                                      rdd_baseline_fit.params['Intercept'], \
                                      rdd_baseline_fit.params['TREATED']

                a_opt, b_opt, d_opt = rdd_baseline_opt_fit.params['x'], \
                                      rdd_baseline_opt_fit.params['Intercept'], \
                                      rdd_baseline_opt_fit.params['TREATED']

                predictions_rdd_raw[j, i, :] = lin_reg(a_raw, b_raw, d_raw)
                predictions_rdd_opt[j, i, :] = lin_reg(a_opt, b_opt, d_opt)

        np.savez(examplefile,
                 data_x=data_x,
                 data_y=data_y,
                 predictions_m0_mu=predictions_m0_mu,
                 predictions_m1_mu=predictions_m1_mu,
                 predictions_m0_var=predictions_m0_var,
                 predictions_m1_var=predictions_m1_var,
                 predictions_rdd_raw=predictions_rdd_raw,
                 predictions_rdd_opt=predictions_rdd_opt,
                 effectsizes_m1=effectsizes_m1,
                 effectsizes_bma=effectsizes_bma,
                 effectsizes_total_bma=effectsizes_total_bma)
    return data_x,\
           data_y, \
           predictions_m0_mu, \
           predictions_m1_mu, \
           predictions_m0_var, \
           predictions_m1_var, \
           predictions_rdd_raw, \
           predictions_rdd_opt, \
           effectsizes_m1, \
           effectsizes_bma, \
           effectsizes_total_bma


#
def plot_visualisation_example(data_x,
                               data_y,
                               predictions_m0_mu,
                               predictions_m1_mu,
                               predictions_m0_var,
                               predictions_m1_var,
                               predictions_rdd_raw,
                               predictions_rdd_opt,
                               effectsizes_m1,
                               effectsizes_bma,
                               effectsizes_total_bma):
    fig, axes = plt.subplots(nrows=3, ncols=F, sharex=True, sharey='col', figsize=(18, 6))
    d_range = [0.0, 2.0, 4.0]
    for j, d in enumerate(d_range):
        axes[j, 0].set_ylabel('$d = {:0.1f}$'.format(d))
        for i, f in enumerate(functions):
            x, y = simulate_data(xmin=xmin, xmax=xmax, f=f, x0=x0, d=d, sigma=sigma)
            x_pre = np.linspace(xmin, 0, num=100)
            x_post = np.linspace(0, xmax, num=100)
            f_pre = f(x_pre, x0)
            f_post = f(x_post, x0) + d
            axes[j, i].scatter(x, y, marker='o',
                               edgecolors='grey',
                               c='grey',
                               alpha=0.3,
                               s=30)
            for xu, fu in zip([x_pre, x_post], [f_pre, f_post]):
                axes[j, i].plot(xu, fu, color='k', lw=1)


            x_range = np.linspace(xmin, xmax, num=101)
            for k, kernel in enumerate(kernels):
                mu_m0 = predictions_m0_mu[j, i, k, :]
                mu_m1 = predictions_m1_mu[j, i, k, :]
                axes[j, i].plot(x_range, mu_m0, lw=1, ls='-', color=colors[k])
                axes[j, i].plot(x_range, mu_m1, lw=1, ls='--', color=colors[k])
                es = effectsizes_bma[j, i, k]
                es_x = -0.8 + 0.12*k
                axes[j, i].plot([es_x, es_x],
                                [mu_m0[50]-0.5*es, mu_m0[50]+0.5*es],
                                lw=5,
                                color=colors[k],
                                label=kernel.name.capitalize().replace('_', ' '))

            # the vertical location of the BMA is a bit of a hack, because we don't store the posterior model
            # probabilities here, which would be needed to properly determine the marginal regression. Might reconsider
            # when adding the VI results.
            es = effectsizes_total_bma[j, i]
            es_x = -0.92
            axes[j, i].plot([es_x, es_x],
                            [mu_m0[50] - 0.5 * es, mu_m0[50] + 0.5 * es],
                            lw=5,
                            color='k', label='BMA')

            axes[j, i].axvline(x=x0, ls='--', color='k')
            axes[j, i].set_xlim([-1, 1])

    for i, ax in enumerate(axes[0, :]):
        ax.set_title(f_names[i], fontsize=24)
    for ax in axes[-1, :]:
        ax.set_xlabel('$x$')

    handles, labels = axes[2, 0].get_legend_handles_labels()
    plt.figlegend(handles, labels,
                  loc='lower center',
                  ncol=K + 1,
                  fontsize=16,
                  frameon=False,
                  bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    if save_output_to_pdf:
        plt.savefig('simulation_examples.pdf', bbox_inches='tight', pad_inches=0)

    plt.show()

#


f_linear = lambda x, x0: 0.23 + 0.89 * x
f_quad = lambda x, x0: 3 * x ** 2 * (x < x0) + 4 * x ** 2 * (x >= x0)
f_cube = lambda x, x0: 3 * x ** 3 * (x < x0) + 4 * x ** 3 * (x >= x0)
f_lee = lambda x, x0: (0.48 + 1.27 * x + 7.18 * x ** 2 + 20.21 * x ** 3 + 21.54 * x ** 4 + 7.33 * x ** 5) * (x < x0) + \
                      (0.48 + 0.84 * x - 3 * x ** 2 + 7.99 * x ** 3 - 9.01 * x ** 4 + 3.56 * x ** 5) * (x >= x0)
f_cate1 = lambda x, x0: 0.42 + 0.84 * x - 3.0 * x ** 2 + 7.99 * x ** 3 - 9.01 * x ** 4 + 3.56 * x ** 5
f_cate2 = lambda x, x0: 0.42 + 0.84 * x + 7.99 * x ** 3 - 9.01 * x ** 4 + 3.56 * x ** 5
f_ludwig = lambda x, x0: (3.71 + 2.3 * x + 3.28 * x ** 2 + 1.45 * x ** 3 + 0.23 * x ** 4 + 0.03 * x ** 5) * (x < x0) + \
                         (3.71 + 18.49 * x - 54.81 * x ** 2 + 74.3 * x ** 3 - 45.02 * x ** 4 + 9.83 * x ** 5) * (
                                     x >= x0)
f_curvature = lambda x, x0: (0.48 + 1.27 * x - 3.44 * x ** 2 + 14.147 * x ** 3 + 23.694 * x ** 4 + 10.995 * x ** 5) * (
            x < x0) + \
                            (0.48 + 0.84 * x - 0.3 * x ** 2 - 2.397 * x ** 3 - 0.901 * x ** 4 + 3.56 * x ** 5) * (
                                        x >= x0)

functions = [f_linear, f_quad, f_cube, f_lee, f_cate1, f_cate2, f_ludwig, f_curvature]
f_names = ['Linear', 'Quad', 'Cubic', 'Lee', 'CATE1', 'CATE2', 'Ludwig', 'Curvature']
kernels = [Polynomial(degree=1), Exponential(), Matern32(), SquaredExponential()]

sigma = 1.0
d_trues = np.asarray([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
n = 100

x0 = 0.0
xmin, xmax = -1, 1

F = len(functions)
K = len(kernels)
D = len(d_trues)

colors = cm.Set1(range(K + 1))

save_output_to_pdf = True

ls_2step = '--'
ls_bndd = '-'
ls_fs_all = ':'
ls_fs_opt = '-'

if True:
    results = run_simulations(n)
    plot_simulation_results(*results)

if True:
    vis_results = run_visualisation_example()
    plot_visualisation_example(*vis_results)
