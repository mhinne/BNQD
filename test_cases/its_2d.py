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
    Product

from gpflow.utilities import print_summary, deepcopy
from kernels import SpectralMixture, MultiOutputKernel, IndependentKernel

from scipy.stats import binom, poisson

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

obs_model = 'g'
p = 2


def f_1(x):
    return np.sin(6*x)

def sigmoid(x, k, x0):
    return 1 / (1+np.exp(-k*(x-x0)))

def f_2(x, x0):
    return (1-sigmoid(x, 10, x0))*np.sin(6*x+0.7) + sigmoid(x, 10, x0)*np.sin(30*x+0.7)


x0 = 0.5
xmin, xmax = 0, 1
n_1, n_2 = 10, 25
sigma = 0.3

x_1 = np.linspace(xmin, xmax, n_1)
x_2 = np.linspace(xmin, 0.8*xmax, n_2)

mu_1 = f_1(x_1)
mu_2 = f_2(x_2, x0)

y_1 = np.random.normal(loc=mu_1, scale=sigma)
y_2 = np.random.normal(loc=mu_2, scale=sigma)

colors = cm.get_cmap('tab10', 10)
x_pred = np.linspace(xmin, xmax, 100)

plt.figure()
for i, x, y, f in zip(np.arange(2), [x_1, x_2], [y_1, y_2], [f_1(x_pred), f_2(x_pred, x0)]):
    plt.plot(x, y, 'o', c=colors(i))
    plt.plot(x_pred, f, ls='--', c=colors(i))
plt.xlim([xmin, xmax])
plt.axvline(x=x0, c='k')
plt.show()

data = [(x_1, y_1), (x_2, y_2)]
if type(data) is list:
    p = len(data)
    # assume for now that if we have multiple outputs, we have only one input
    X = list()
    Y = list()
    for i in range(p):
        x, y = data[i]
        x, y = x.reshape(-1, 1), y.reshape(-1, 1)
        X.append(np.hstack((x, i*np.ones_like(x))))
        Y.append(np.hstack((y, i*np.ones_like(y))))
    X = np.vstack(X)
    Y = np.vstack(Y)

data_split = split_data((X, Y), x0=x0, forcing_variable=0)
p = 2

base_kernel = Matern32(active_dims=[0])
kernel = IndependentKernel(kernels=[deepcopy(base_kernel), deepcopy(base_kernel)],
                           x0=x0, forcing_variable=0, name='indep')

x_pred = np.linspace(xmin, xmax, 20)

# K = kernel.K(x_pred[:, np.newaxis])
# plt.figure()
# plt.imshow(K)
# plt.show()

# in optimization, X in K(X,X2) is of shape (None, 1)????
m = gpflow.models.GPR(data=(x_1[:, None], y_1[:, None]), kernel=kernel)

# mogp_kernel = MultiOutputKernel(kernel, output_dim=p, rank=p)
# likelihood = Gaussian()
#
# m = gpflow.models.VGP((X, Y), kernel=mogp_kernel, likelihood=likelihood)

gpflow.optimizers.Scipy().minimize(
    m.training_loss, m.trainable_variables, options=dict(maxiter=10000), method="L-BFGS-B",
)




# todo: how to fix the changepoint for MOGP? See https://github.com/GPflow/GPflow/issues/1195
#  Can we implement the active_dims parameter ourselves?
#  Combination kernel already does not contain active_dims, but can check whether component kernels operate on
#  overlapping slices
#  Probably easiest to implement a new DiscreteChangepoint kernel

# base_kernel = ChangePoints([Matern32(active_dims=[0]), Matern32(active_dims=[0])],
#                            locations=[x0], steepness=999)
# Base kernel
# k = base_kernel
# # Coregion kernel
# coreg = gpflow.kernels.Coregion(output_dim=2, rank=2, active_dims=[1])
# kernel = Product([k, coreg], name='test')
# return Product([k, coreg], name='MO_{:s}'.format(k.name))

# kernel = MultiOutputKernel(base_kernel, output_dim=p, rank=p)
# kernel = Matern32(active_dims=[0])
#
# qed = BNQD(data=data,
#            likelihood=likelihood,
#            kern_list=kernel,
#            intervention_pt=x0,
#            qed_mode='ITS')
# qed.train()
# print(qed.get_results())



# x_aug = augment_input(x, p)
# y_aug = augment_output(y, p)
#
# mogp_k = MultiOutputKernel(SquaredExponential, output_dim=p, rank=p)
# likelihood = SwitchedLikelihood([Gaussian(), Gaussian()])
#
# # now build the GP model as normal
# m = gpflow.models.VGP((x_aug, y_aug), kernel=mogp_k, likelihood=likelihood)
#
# gpflow.optimizers.Scipy().minimize(
#     m.training_loss, m.trainable_variables, options=dict(maxiter=10000), method="L-BFGS-B",
# )
#
# x_pred_mogp_0 = np.hstack((x_pred[:, np.newaxis], np.zeros((200,1))))
#
# mu, var = m.predict_y(x_pred_mogp_0)
# mu = mu.numpy()[:, 0]
# intv = 2*np.sqrt(var.numpy()[:, 0])
# plt.plot(x_pred, mu, c=colors(0), ls='--')
# plt.fill_between(x_pred, mu + intv, mu - intv, alpha=0.3, color=colors(0))
#
# x_pred_mogp_1 = np.hstack((x_pred[:, np.newaxis], np.ones((200, 1))))
#
# mu, var = m.predict_y(x_pred_mogp_1)
# mu = mu.numpy()[:, 0]
# intv = 2 * np.sqrt(var.numpy()[:, 0])
# plt.plot(x_pred, mu, c=colors(1), ls='--')
# plt.fill_between(x_pred, mu + intv, mu - intv, alpha=0.3, color=colors(1))

# plt.plot(x_pred, mu.numpy()[200:, 0], c=colors(0), ls='--')

# output covariance:
# B = mogp_k.kernels[1].output_covariance().numpy()
# plt.show()

# for xpred, concat the desired output dimension

# TODO: What do we want to achieve here? Train on x <= x0 and then predict?
# TODO: Compare disjunct model with continuous model?
# TODO: in ITS, we could/would/should train on data pre-threshold
# TODO: simulation with y_1 drawn from a change-point kernel?

# qed = BNQD(data=(x, y),
#            likelihood=likelihood,
#            kern_list=mogp_k,
#            intervention_pt=x0)
# qed.train()

# todo: first implement MOGP-VGP here in plain code before adding to BNQD!
