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



def plot_mixture_component(x, y, col, ax=None):
    if ax is None:
        ax = plt.gca()
    bot = np.min(y)
    ax.plot(x, y, color=col, lw=1)
    ax.fill_between(x, np.squeeze(y), bot * np.ones((len(x))), color=col, alpha=0.1)


#
def plot_f(x, f, var, ax=None, color='black', alpha=0.2, ls='solid', label='', scale=(0, 1)):
    if ax is None:
        ax = plt.gca()

    mean = yscale(f, scale[0], scale[1])
    ub = yscale(f + 2. * np.sqrt(var), scale[0], scale[1])
    lb = yscale(f - 2. * np.sqrt(var), scale[0], scale[1])
    ax.plot(x, mean, color=color, linestyle=ls, label=label, lw=1)
    ax.fill_between(x, ub, lb, color=color, alpha=alpha)


#
S = 3

fig = plt.figure(figsize=(12, 3))
gs = fig.add_gridspec(3, 2, wspace=0.1, hspace=0.08, width_ratios=[3, 1])

for subject in [2]:
    print('Subject {:d}'.format(subject + 1))

    filename = 'meditation {:d}'.format(subject + 1)

    exp_path = filename + '\\'

    control = np.load(exp_path + 'meanvar_control_discontinuous.npy')
    intervention = np.load(exp_path + 'meanvar_intervention_discontinuous.npy')
    continuous = np.load(exp_path + 'continuous.npy')

    continuous_pdfs = np.load(exp_path + 'continuous_pdfs.npy')
    control_pdfs = np.load(exp_path + 'control_pdfs.npy')
    intervention_pdfs = np.load(exp_path + 'intervention_pdfs.npy')
    samples = np.load(exp_path + 'functionsamples{:d}.npy'.format(subject + 1), allow_pickle=True)
    bayesfactor = np.load(exp_path + 'bayes_factor.npy')
    print('Bayes factor: {:0.2f}'.format(bayesfactor[subject]))

    X, Y, b = np.load(exp_path + 'data.npy', allow_pickle=True)

    y_mu, y_sigma = np.load(exp_path + 'standardization_terms.npy')

    mean_disc_control, std_disc_control = control
    mean_disc_interv, std_disc_interv = intervention
    mean_continuous, std_continuous = continuous

    colors = {'cont': '#2a2b2d', 'dc': '#d9541e', 'di': '#2da8d8'}

    yscale = lambda y_raw, mu, sigma: y_raw * sigma + mu

    x0 = b

    n = mean_continuous.shape[0]
    nc = mean_disc_control.shape[0]
    ni = mean_disc_interv.shape[0]

    ax0 = fig.add_subplot(gs[0:3, 0])

    ax0.plot(X - x0, yscale(Y, y_mu, y_sigma), lw=1, linestyle='--', color='k', label='Observations')
    ax0.set_xlim([X[0], X[-1]] - x0)
    ax0.set_xlabel('Time (h)')
    ax0.set_ylabel('Heart rate (bpm)')
    ax0.axvline(x=0, color='k', linestyle=':', lw=1)

    x = np.linspace(X[0], X[-1], n) - x0
    plot_f(x, mean_continuous[:, 0],
           std_continuous[:, 0], ax0, colors['cont'], label='$f_0(x)$',
           scale=(y_mu, y_sigma))

    xc = np.linspace(X[0], X[-1], nc) - x0
    plot_f(xc[xc <= x0], mean_disc_control[xc <= x0, 0],
           std_disc_control[xc <= x0, 0], ax0, colors['dc'],
           label='$f_1^{A}(x), \,x< x_0$', scale=(y_mu, y_sigma))

    plot_f(xc[xc >= x0], mean_disc_control[xc >= x0, 0],
           std_disc_control[xc >= x0, 0], ax0, colors['dc'], ls='--',
           label='$f_1^{A}(x),\, x\geq x_0$', scale=(y_mu, y_sigma))

    xi = np.linspace(x0, X[-1], ni) - x0
    plot_f(xi, mean_disc_interv[xi >= x0, 0],
           std_disc_interv[xi >= x0, 0], ax0, colors['di'],
           label='$f_1^{B}(x), \,x\geq x_0$',
           scale=(y_mu, y_sigma))

    # plot posterior samples of extrapolation

    nsamples = samples.shape[1]
    for i in range(2):
        ax0.plot(xc, yscale(samples[0, i], y_mu, y_sigma), color=colors['dc'], ls=':', lw=1)

    es = np.load(exp_path + 'causal_effects.npy') * y_sigma

    print('Mean causal effect: {:0.2f}, maximum causal effect: {:0.2f}'.format(np.mean(es), np.max(es)))

    oldticks = ax0.get_xticks()
    newticks = ['00:{:02d}'.format(int(np.round(tick / 60.))) if tick >= 0 else '-00:{:02d}'.format(
        -1 * int(np.round(tick / 60.))) for tick in oldticks]

    ax0.set_xticklabels(newticks)

    handles, labels = ax0.get_legend_handles_labels()
    lgd = plt.figlegend(handles, labels, ncol=5, loc='upper center', fancybox=False,
                  bbox_to_anchor=(0.45, 1.15), frameon=False)

    max_freq = 1.5
    xx = np.linspace(0, max_freq, continuous_pdfs.shape[1])
    Q = continuous_pdfs.shape[0]

    eps = 1.0e-4

    Q0 = continuous_pdfs.shape[0]
    Qc = control_pdfs.shape[0]
    Qi = intervention_pdfs.shape[0]

    print('Q_0: {:d}'.format(Q0))
    print('Q_c: {:d}'.format(Qc))
    print('Q_i: {:d}'.format(Qi))

    ax1 = fig.add_subplot(gs[0, 1])
    for i in range(Q0):
        plot_mixture_component(xx, np.log(continuous_pdfs[i, :, 0] + eps), col=colors['cont'], ax=ax1)
    ax1.set_xticks([])

    ax2 = fig.add_subplot(gs[1, 1], sharey=ax1)
    for i in range(Qc):
        plot_mixture_component(xx, np.log(control_pdfs[i, :, 0] + eps), col=colors['dc'], ax=ax2)
    ax2.set_xticks([])
    ax2.set_ylabel('log $S(\\omega)$')

    ax3 = fig.add_subplot(gs[2, 1], sharey=ax1)
    for i in range(Qi):
        plot_mixture_component(xx, np.log(intervention_pdfs[i, :, 0] + eps), col=colors['di'], ax=ax3)

    ax1.set_yticks([])
    for ax in [ax1, ax2, ax3]:
        ax.set_ylim(bottom=np.log(eps))
        ax.set_xlim([xx[0], xx[-1]])

    ax3.set_xlabel('$\\omega$')


plt.savefig('meditation_its.pdf',
            bbox_extra_artists=(lgd,),
            bbox_inches='tight',
            pad_inches=0)
plt.show()