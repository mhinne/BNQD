import numpy as np
import geopandas as gpd
from tqdm import tqdm
import pandas as pd
import cbsodata
import matplotlib.pyplot as plt
import os
import gpflow as gpf

from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyproj import Proj, transform
from read_election_data import read_data, total_vote_count
from BNQD import BNQD
from gpflow.kernels import Polynomial, Matern32, SquaredExponential
from gpflow.likelihoods import Gaussian

plt.rc('axes', titlesize=24)        # fontsize of the axes title
plt.rc('axes', labelsize=20)        # fontsize of the x and y labels
plt.rc('xtick', labelsize=16)       # fontsize of the tick labels
plt.rc('ytick', labelsize=16)       # fontsize of the tick labels
plt.rc('legend', fontsize=22)       # legend fontsize
plt.rc('figure', titlesize=32)      # fontsize of the figure title

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})

print('GPflow version    ', gpf.__version__)
print('BNQD version      ', BNQD.__version__)


def is_below_border_scaled(x, a=0.20, b=395000):
    l = a * (x[:, 0] * s[0] + mu[0]) + b
    return (x[:, 1] * s[1] + mu[1]) <= l


save_output_to_pdf = True
fn = 'election_data.npz'
geodata_url = 'https://geodata.nationaalgeoregister.nl/cbsgebiedsindelingen/wfs?request=GetFeature&service=WFS&version=2.0.0&typeName=cbs_gemeente_2017_gegeneraliseerd&outputFormat=json'

if not os.path.isfile(fn):
    # Collecting these data is a fairly slow process, so we store intermediate results.
    print('Preprocessing data')

    vote_data = read_data()
    total_votes = total_vote_count(vote_data)

    x_train = list()
    x_test = list()
    vote_data = read_data()
    totals = total_vote_count(vote_data)

    X = np.column_stack((np.array([vote_data[label]['Lon'] for label in vote_data.keys()]),
                         np.array([vote_data[label]['Lat'] for label in vote_data.keys()])))
    n, p = X.shape
    total_votes = total_vote_count(vote_data)
    y = np.array([float(
        vote_data[label]['PVV'] + vote_data[label]['SP'] + vote_data[label]['50PLUS'] + vote_data[label]['FvD']) /
                  total_votes[label] for label in vote_data.keys()])
    x = np.zeros((n, p))

    inProj = Proj(init='epsg:4326')  # world coordinates
    outProj = Proj(init='epsg:28992')  # Dutch coordinates
    x[:, 0], x[:, 1] = transform(inProj, outProj, X[:, 0], X[:, 1])

    # Retrieve data with municipal boundaries from PDOK
    municipal_boundaries = gpd.read_file(geodata_url)

    data = pd.DataFrame(cbsodata.get_data('83765NED',
                                          select=['WijkenEnBuurten', 'Codering_3', 'GeboorteRelatief_25']))
    data['Codering_3'] = data['Codering_3'].str.strip()

    # Link data from Statistics Netherlands to geodata
    municipalities = pd.merge(municipal_boundaries, data, left_on="statcode",
                              right_on="Codering_3")

    # slow procedure
    for municipality in tqdm(municipalities['statnaam']):

        record = dict()
        record['name'] = municipality
        if municipality in vote_data.keys():
            # print('Voting data available')
            # in lon/lat
            x1, x2 = vote_data[municipality]['Lon'], vote_data[municipality]['Lat']
            x1, x2 = transform(inProj, outProj, x1, x2)
            record['x1'] = x1
            record['x2'] = x2
            pop_frac = float(
                vote_data[municipality]['PVV'] + vote_data[municipality]['SP'] + vote_data[municipality]['50PLUS'] +
                vote_data[municipality]['FvD']) / totals[municipality]

            record['y'] = pop_frac
            # in epsg:28992
            record['geom'] = municipalities[municipalities['statnaam'] == municipality]['geometry']
            x_train.append(record)

        else:
            record['geom'] = municipalities[municipalities['statnaam'] == municipality]['geometry']
            for shape in record['geom']:
                centroid = shape.centroid
                record['x1'] = centroid.x
                record['x2'] = centroid.y
        x_test.append(record)

    x0, x1, y0, y1 = 1000, 300000, 280000, 620000

    mu = [(x1+x0)/2, (y1+y0)/2]
    s = [x1-x0, y1-y0]

    ntrain = len(x_train)
    ntest = len(x_test)

    xtrain = np.zeros((ntrain, 2))
    ytrain = np.zeros((ntrain, 1))

    for i in range(ntrain):
        xtrain[i, :] = np.array([x_train[i]['x1'] - mu[0], x_train[i]['x2'] - mu[1]]) / s
        ytrain[i] = x_train[i]['y']

    xtest = np.zeros((ntest, 2))
    regions = list()
    for i in range(ntest):
        xtest[i, :] = np.array([x_test[i]['x1'] - mu[0], x_test[i]['x2'] - mu[1]]) / s
        regions.append(x_test[i]['name'])

    np.savez(fn, x=x, y=y, xtrain=xtrain, ytrain=ytrain, xtest=xtest, mu=mu, s=s, regions=regions, allow_pickle=True)
else:
    print('Formatted data available, loading')
    npz = np.load(fn, allow_pickle=True)
    x = npz['x']
    y = npz['y']
    xtrain = npz['xtrain']
    ytrain = npz['ytrain']
    xtest = npz['xtest']
    ntest = len(xtest)
    mu = npz['mu']
    s = npz['s']
    municipal_boundaries = gpd.read_file(geodata_url)
    regions = npz['regions']


# NB: a somewhat reasonable initial lengthscale is needed for the Matern
kernel_list = [Polynomial(degree=1), Matern32(lengthscales=0.01)]
kernel_names = ['Polynomial ($d=1$)', 'Matern ($\\nu=3/2$)']
geordd = BNQD(data=(xtrain, ytrain),
              likelihood=Gaussian(),
              kern_list=kernel_list,
              split_function=is_below_border_scaled)
geordd.train()

# In the MIGP case, we have no standard effect size measure, so we implement that manually below.
print(geordd.get_results())


def pred_to_dataframe(pred):
    mu_c = pred[0]
    pred_pop_frac = dict()
    for i in range(ntest):
        pred_pop_frac[regions[i]] = mu_c[i]

    pred_df = pd.DataFrame.from_dict(pred_pop_frac).T
    pred_df.rename(columns={0: 'pop_vote_frac'}, inplace=True)
    return pred_df
#


x0, x1, y0, y1 = 1000, 300000, 280000, 620000

# phantom border
a = 0.20
b = 395000
pb_x = np.linspace(x0, x1, num=100)
pb_y = a * pb_x + b

K = len(kernel_list)

cmap = cm.viridis

fig, axes = plt.subplots(nrows=K, ncols=2, figsize=(20, 10 * K))

for i, kernel in enumerate(kernel_list):
    ax = axes[i, 0]
    ax.text(0, 1.1 * (y1 + y0) / 2, kernel_names[i], horizontalalignment='center',
            verticalalignment='center', rotation=90, fontsize=36)
    cont_pred_df = pred_to_dataframe(geordd.M0[i].predict_y(xtest))
    map_votes = pd.merge(municipal_boundaries, cont_pred_df, left_on='statnaam',
                         right_index=True)
    map_votes.plot(ax=ax, column='pop_vote_frac', cmap=cmap, vmin=0.1, vmax=0.65)
    ax.axis('off')
    disc_pred_df = pred_to_dataframe(geordd.M1[i].predict_y(xtest))
    map_votes = pd.merge(municipal_boundaries, disc_pred_df, left_on='statnaam',
                         right_index=True)
    ax = axes[i, 1]
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    map_votes.plot(ax=ax, column='pop_vote_frac', cmap=cmap,
                   vmin=0.1, vmax=0.65, legend=True, cax=cax,
                   legend_kwds={'label': 'Populist vote fraction'})
    ax.axis('off')

axes[0, 0].set_title('Continuous', fontsize=36)
axes[0, 1].set_title('Discontinuous', fontsize=36)

# plot the observations and the actual border
ms = 180
for ax in fig.axes:
    pts = ax.scatter(x[:, 0], x[:, 1], s=ms, marker='o',
                     c=y, edgecolors='k', cmap=cmap,
                     zorder=10)
    ax.plot(pb_x, pb_y, lw=5, ls='--', c='k', zorder=11)

plt.suptitle('Fraction of votes for populist parties', fontsize=44)
plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
if save_output_to_pdf:
    plt.savefig('Dutch_elections_figure.pdf', bbox_inches='tight', pad_inches=0)
plt.show()


x_scaled_b = np.array([(pb_x - mu[0]) / s[0], (pb_y - mu[1]) / s[1]]).T
xrange = np.arange(0, 100)

pred_pre = geordd.predict_y(x_scaled_b + 1e-6)
pred_post = geordd.predict_y(x_scaled_b - 1e-6)

fig = plt.figure(figsize=(12, 8))
ax = plt.gca()

colors = cm.magma(np.linspace(0.5, 0.1, num=K))

for k in range(K):
    _, pred_m1_pre = pred_pre[k]
    pred_m1_pre_mu, pred_m1_pre_var = pred_m1_pre

    _, pred_m1_post = pred_post[k]
    pred_m1_post_mu, pred_m1_post_var = pred_m1_post

    es_m1_mu = pred_m1_post_mu - pred_m1_pre_mu
    es_m1_var = pred_m1_pre_var + pred_m1_post_var
    ax.plot(xrange, es_m1_mu, c=colors[k, :], label='{:s} kernel'.format(kernel_names[k]), lw=4.0)
    ax.fill_between(xrange,
                    np.squeeze(es_m1_mu + 0.5 * np.sqrt(es_m1_var)),
                    np.squeeze(es_m1_mu - 0.5 * np.sqrt(es_m1_var)),
                    color=colors[k, :], alpha=0.3)

ax.axhline(y=0, c='k', ls='--', label='No effect', lw=4)
ax.set_xlabel('Phantom border (west-east)')
ax.set_xlim([xrange[0], xrange[-1]])
ax.set_ylabel('Effect size')
plt.suptitle('2D effect size at the phantom border')
ax.legend(loc='lower left', framealpha=0.95, fancybox=False)
if save_output_to_pdf:
    plt.savefig('Dutch_elections_effectsize_figure.pdf', bbox_inches='tight', pad_inches=0)
plt.show()
