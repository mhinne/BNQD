import numpy as np
import matplotlib.pyplot as plt


plt.rc('axes', titlesize=26)  # fontsize of the axes title
plt.rc('axes', labelsize=20)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=16)  # fontsize of the tick labels
plt.rc('ytick', labelsize=16)  # fontsize of the tick labels
plt.rc('legend', fontsize=18)  # legend fontsize
plt.rc('figure', titlesize=28)  # fontsize of the figure title

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})


# plt.rc('figure', titlesize=28)  # fontsize of the figure title

def shifting_discontinuity_mean_function(x, d=0):
    """
    f(x) = sin((12+d) x) + (2/3)cos((25+d) x) where forall x>0, d=0
    :arg d: order of discontinuity in frequency
    """
    return (x <= 0) * np.sin(12 * x) + (x > 0) * np.sin((12 + d) * x) + (x <= 0) * 0.66 * np.cos(25 * x) + (
                x > 0) * 0.66 * np.cos((25 + d) * x)


def rmse(f_true, f_gp):
    return np.sqrt(np.mean((f_true - f_gp) ** 2))


def correlation(a, b):
    return np.corrcoef(a, b)[0, 1]


colors = {'cont': '#2a2b2d', 'dc': '#d9541e', 'di': '#2da8d8'}

exp_path = r'its/fullruns/'

true_effect = np.load(exp_path + 'true_effect.npy')
effect_sizes = np.load(exp_path + 'effect_sizes.npy')
bayes_factors = np.load(exp_path + 'bayes_factors.npy')
posterior_samples = np.load(exp_path + 'samples_disc.npy')

effect_sizes_ARMA = np.load(exp_path + 'effect_sizes_ARIMA.npy')
bayes_factors_ARMA = np.load(exp_path + 'bayes_factors_ARIMA.npy')

D, nruns, n = true_effect.shape

nsamples = 100
n = posterior_samples.shape[3]

x = np.linspace(-1.5, 1.5, n)
x0 = int((n - 1) / 2)

x_arma = np.linspace(0, 1.25, 100)

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(30, 10), sharex=True, sharey=True)
plt.subplots_adjust(wspace=0.0, hspace=0.0)

for d, ax in enumerate(fig.axes):
    print('Shift = {:d}'.format(d))
    f = shifting_discontinuity_mean_function(x, d)
    f0 = shifting_discontinuity_mean_function(x, 0)
    d_true = (f - f0)[x0:]

    f_arma = shifting_discontinuity_mean_function(x_arma, d)
    f0_arma = shifting_discontinuity_mean_function(x_arma, 0)
    d_true_arma = f_arma - f0_arma

    true_rmse = rmse(f, f0)
    run_range = np.arange(0 + d * nruns, nruns * (d + 1))
    samples = posterior_samples[run_range, :, :, :, 0]
    errors = list()
    arma_errors = []

    for run in range(nruns):
        for sample1 in range(nsamples):
            sample_c = samples[run, 0, sample1, x0:]
            for sample2 in range(nsamples):
                sample_i = samples[run, 1, sample2, 0::2]
                d_sample = sample_i - sample_c

                error = rmse(d_true, d_sample)
                errors.append(error)
        arma_run = effect_sizes_ARMA[d, run, :]
        arma_error = rmse(d_true_arma, arma_run)
        arma_errors.append(arma_error)

    ax.hist(errors, bins=50, density=True, color='#d9541e', label='GPR $r$', alpha=0.5)
    ax.axvline(x=np.mean(arma_errors), color='k', label='ARMA $r$', lw=4, ls='--')

    ax.annotate('$\\alpha={:d}$'.format(d), xy=(0.5, 4.1), fontsize=36, ha='left', va='center')
    # ax.annotate('$\log BF={:0.1f} \pm {:0.1f}$'.format(np.mean(bayes_factors[d,:]),
    #                                                    np.std(bayes_factors[d,:])/np.sqrt(nruns)),
    #             xy=(0.55, 2.0), fontsize=36, ha='left', va='center',
    #             bbox={'alpha': 0.8, 'facecolor':'w', 'edgecolor': 'None'})

for ax in axes[2, :]:
    ax.set_xlabel('RMSE')

for ax in axes[:, 0]:
    ax.set_ylabel('Density')

handles, labels = axes[0, 0].get_legend_handles_labels()

plt.figlegend(handles, labels, ncol=3, loc='upper center', fancybox=False,
              bbox_to_anchor=(0.4, 0.99), frameon=False)

plt.savefig(exp_path + 'aggregate_plot.pdf', bbox_inches='tight', pad_inches=0)

# a = posterior_samples[0,0,5,x0:,0]
# b = posterior_samples[0,1,5,0::2,0]
# plt.figure()
# plt.plot(d_true[0::3])
# plt.plot(d_true_arma)

# plt.plot(np.linspace(0, 1.5, 601), posterior_samples[0,1,5,:,0])

logBFs_GPR = np.mean(bayes_factors, axis=1)
logBFs_GPR_se = np.std(bayes_factors, axis=1) / np.sqrt(nruns)

logBFs_ARMA = np.mean(bayes_factors_ARMA, axis=1)
logBFs_ARMA_se = np.std(bayes_factors_ARMA, axis=1) / np.sqrt(nruns)

str_gpr = r''
for d in range(D):
    str_gpr += '${:0.1f} \pm {:0.1f}$'.format(logBFs_GPR[d], logBFs_GPR_se[d]) + ' & '

str_arma = r''
for d in range(D):
    str_arma += '${:0.1f} \pm {:0.1f}$'.format(logBFs_ARMA[d], logBFs_ARMA_se[d]) + ' & '

plt.show()