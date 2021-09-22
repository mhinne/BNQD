import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st


def split_data(data, x0, forcing_variable, include_x0=False):
    if isinstance(data, tuple):
        X, Y = data
    else:
        X = data

    if include_x0:
        ix_pre = np.where(X[:, forcing_variable] <= x0)[0]
    else:
        ix_pre = np.where(X[:, forcing_variable] < x0)[0]
    ix_post = np.where(X[:, forcing_variable] >= x0)[0]
    if isinstance(data, tuple):
        return (X[ix_pre, :], Y[ix_pre, :]), (X[ix_post, :], Y[ix_post, :])
    else:
        return X[ix_pre, :], X[ix_post, :]

#
def augment_input(X, p):
    stackable_arrays = [np.hstack((X[j][:, np.newaxis], j * np.ones((X[j].shape[0], 1)))) for j in range(p)]
    return np.vstack(stackable_arrays)


#
def augment_output(Y):
    n, p = Y.shape
    stackable_arrays = [np.hstack((Y[:, j][:, np.newaxis], j * np.ones((n, 1)))) for j in range(p)]
    return np.vstack(stackable_arrays)


#
def renormalize(ns):
    """trick to get normalization of evidences.
    https://stats.stackexchange.com/questions/66616/converting-normalizing-very-small-likelihood-values-to-probability
    """
    mx = np.max(ns)
    epsilon = 1e-16
    n = len(ns)
    threshold = np.log(epsilon) - np.log(n)
    ds = ns - mx
    # line below screws BMA computation and isn't needed when all models have somewhat reasonable evidence
    # ds[ds < threshold] = 0  # throw out values below precision
    return np.exp(ds) / np.sum(np.exp(ds))


def logsumexp(x):
    c = x.max()
    return c + np.log(np.sum(np.exp(x - c)))


def logmeanexp(x):
    c = x.max()
    return c + np.log(np.mean(np.exp(x - c), axis=0))

#
def plot_m0(bnqd_obj, kernel, pred_range=None, ax=None, hdi=0.95, plot_opts=dict()):
    if ax is None:
        ax = plt.gca()
    X = bnqd_obj.data[0]
    if pred_range is None:
        fv = bnqd_obj.forcing_variable
        xmin, xmax = np.floor(np.min(X[:, fv])), np.ceil(np.max([X[:, fv]]))
    else:
        xmin, xmax = pred_range
    padding = plot_opts.get('padding', 0.0)
    x_pred = np.linspace((1+padding)*xmin, (1+padding)*xmax, plot_opts.get('res', 200))
    mu, var = bnqd_obj.M0[kernel].predict_y(x_pred[:, np.newaxis])
    mu = mu.numpy().flatten()
    var = var.numpy().flatten()

    ax.plot(x_pred, mu,
            lw=plot_opts.get('linewidth', 2),
            ls=plot_opts.get('linestyle', '-'),
            c=plot_opts.get('color', 'r'),
            label=plot_opts.get('label', ''))
    if hdi == 0.95:
      intv = 2*np.sqrt(var)
    else:
      intv = np.sqrt(var)

    ax.fill_between(x_pred, mu + intv, mu - intv,
                    color=plot_opts.get('color', 'r'),
                    alpha=plot_opts.get('alpha', 0.4))

#


def plot_m1(bnqd_obj, kernel, pred_range=None, ax=None, hdi=0.95, show_x0=True, plot_opts=dict()):
    if ax is None:
        ax = plt.gca()
    X = bnqd_obj.data[0]
    if pred_range is None:
        fv = bnqd_obj.forcing_variable
        xmin, xmax = np.floor(np.min(X[:, fv])), np.ceil(np.max([X[:, fv]]))
    else:
        xmin, xmax = pred_range
    padding = plot_opts.get('padding', 0.0)
    x_pred = np.linspace((1+padding)*xmin, (1+padding)*xmax, plot_opts.get('res', 200))
    x0 = bnqd_obj.x0
    if xmin < x0:
        pred_pre = bnqd_obj.M1[kernel].predict_y(x_pred[x_pred < x0, np.newaxis])
        pred_post = bnqd_obj.M1[kernel].predict_y(x_pred[x_pred >= x0, np.newaxis])

        for i, x, mu, var in zip([0, 1], [x_pred[x_pred < x0], x_pred[x_pred >= x0]],
                              [pred_pre[0].numpy().flatten(), pred_post[0].numpy().flatten()],
                              [pred_pre[1].numpy().flatten(), pred_post[1].numpy().flatten()]):
            if i == 0:
                ax.plot(x, mu,
                        lw=plot_opts.get('linewidth', 2),
                        ls=plot_opts.get('linestyle', '-'),
                        c=plot_opts.get('color', 'r'),
                        label=plot_opts.get('label', ''))
            else:
                ax.plot(x, mu,
                        lw=plot_opts.get('linewidth', 2),
                        ls=plot_opts.get('linestyle', '-'),
                        c=plot_opts.get('color', 'r'))

            intv = var
            if hdi == 0.95:
                intv = 2.0*np.sqrt(intv)

            ax.fill_between(x, mu + intv, mu - intv,
                            color=plot_opts.get('color', 'r'),
                            alpha=plot_opts.get('alpha', 0.4))
    else:
        pred_post = bnqd_obj.M1[kernel].predict_y(x_pred[:, np.newaxis])

        x, mu, var = x_pred, pred_post[0].numpy().flatten(), pred_post[1].numpy().flatten()
        ax.plot(x, mu,
                lw=plot_opts.get('linewidth', 2),
                ls=plot_opts.get('linestyle', '-'),
                c=plot_opts.get('color', 'r'),
                label=plot_opts.get('label', ''))
        intv = var
        if hdi == 0.95:
            intv = 2.0 * np.sqrt(intv)

        ax.fill_between(x, mu + intv, mu - intv,
                        color=plot_opts.get('color', 'r'),
                        alpha=plot_opts.get('alpha', 0.4))


    if show_x0 and xmin < x0:
        ax.scatter([x0, x0], [pred_pre[0].numpy().flatten()[-1], pred_post[0].numpy().flatten()[0]],
                   c=plot_opts.get('pt_fillcolor', 'lightgrey'),
                   edgecolors=plot_opts.get('pt_edgecolor', 'black'),
                   marker='o', s=150, zorder=10)
#


def extrapolate_m0(bnqd_obj, kernel, pred_range = None, ax=None, hdi=0.95, show_x0=True, plot_opts=dict()):
    if ax is None:
        ax = plt.gca()
    X = bnqd_obj.data[0]
    if pred_range is None:
        fv = bnqd_obj.forcing_variable
        xmin, xmax = np.floor(np.min(X[:, fv])), np.ceil(np.max([X[:, fv]]))
    else:
        xmin, xmax = pred_range
    x_pred = np.linspace(xmin, xmax, plot_opts.get('res', 200))
    x0 = bnqd_obj.x0

    pred_pre = bnqd_obj.M1[kernel].predict_y(x_pred[x_pred < x0, np.newaxis])
    pred_post = bnqd_obj.M1[kernel].predict_y(x_pred[x_pred >= x0, np.newaxis])

    for x, mu, var in zip([x_pred[x_pred < x0], x_pred[x_pred >= x0]],
                          [pred_pre[0].numpy().flatten(), pred_post[0].numpy().flatten()],
                          [pred_pre[1].numpy().flatten(), pred_post[1].numpy().flatten()]):
        ax.plot(x, mu,
                lw=plot_opts.get('linewidth', 2),
                ls=plot_opts.get('linestyle', '-'),
                c=plot_opts.get('color', 'r'))
        intv = var
        if hdi == 0.95:
            intv = 2.0*np.sqrt(intv)

        ax.fill_between(x, mu + intv, mu - intv,
                        color=plot_opts.get('color', 'r'),
                        alpha=plot_opts.get('alpha', 0.4))

    if show_x0:
        ax.scatter([x0, x0], [pred_pre[0].numpy().flatten()[-1], pred_post[0].numpy().flatten()[0]],
                   c=plot_opts.get('pt_fillcolor', 'lightgrey'),
                   edgecolors=plot_opts.get('pt_edgecolor', 'black'),
                   marker='o', s=150, zorder=10)
#


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)

    From: https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
#


def plot_effect_size(bnqd_obj, kernel, ax=None, plot_opts=dict()):
    if ax is None:
        ax = plt.gca()

    log_bf = bnqd_obj.get_bayes_factor()[kernel]
    pmp_k = bnqd_obj.get_model_posterior()[kernel, :]
    d_mu, d_var = bnqd_obj.get_effect_sizes()[kernel, :]
    d_sd = np.sqrt(d_var)
    d_min, d_max = d_mu-3*d_sd, d_mu + 3*d_sd
    d_range = np.linspace(d_min, d_max)

    n_samples = plot_opts.get('n_samples', 50000)
    n_0, n_1 = [int(np.round(n)) for n in n_samples * pmp_k]

    if n_1 > 0:
        samples = np.append(np.zeros(n_0), np.random.normal(loc=d_mu, scale=d_sd, size=n_1))
        bma_kde = st.kde.gaussian_kde(samples, bw_method='silverman')
        bma_density = bma_kde(d_range)
        ax.plot(d_range, bma_density, lw=2, c=plot_opts.get('color', 'r'), label='BMA')
        ax.fill_between(d_range, np.zeros_like(d_range), bma_density,
                        color=plot_opts.get('color', 'r'),
                        alpha=plot_opts.get('alpha', 0.3))
        m1_density = st.norm.pdf(d_range, loc=d_mu, scale=d_sd)

        lf = plot_opts.get('lighten_factor', 1.5)

        ax.plot(d_range, m1_density, lw=2,
                c=lighten_color(plot_opts.get('color', 'r'), lf), label='M_1')
        ax.fill_between(d_range, np.zeros_like(d_range), m1_density,
                        color=lighten_color(plot_opts.get('color', 'r'), lf),
                        alpha=plot_opts.get('alpha', 0.3))
    ax.axvline(x=0, c='k', ls='--', lw=2.0, label='M_0')
    ax.set_ylim(bottom=0)
    ax.set_xlim([d_min, d_max])
    ax.set_title('log BF = {:0.2f}'.format(log_bf))

#
