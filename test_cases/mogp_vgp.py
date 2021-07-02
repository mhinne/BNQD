import gpflow
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from gpflow.ci_utils import ci_niter
from gpflow.utilities import print_summary

plt.rcParams["figure.figsize"] = (12, 6)
np.random.seed(123)

def f_1(x):
    return np.sin(6*x)


def sigmoid(x, k, x0):
    return 1 / (1+np.exp(-k*(x-x0)))


def f_2(x, x0):
    return (1-sigmoid(x, 10, x0))*np.sin(6*x+0.7) + sigmoid(x, 10, x0)*np.sin(30*x+0.7)


x0 = 0.5
xmin, xmax = 0, 1
n_1, n_2 = 50, 50
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




output_dim = 2  # Number of outputs
rank = 1  # Rank of W

# Base kernel
k = gpflow.kernels.Matern32(active_dims=[0])

# Coregion kernel
coreg = gpflow.kernels.Coregion(output_dim=output_dim, rank=rank, active_dims=[1])

kern = k * coreg

# This likelihood switches between Gaussian noise with different variances for each f_i:
lik = gpflow.likelihoods.SwitchedLikelihood(
    [gpflow.likelihoods.Gaussian(), gpflow.likelihoods.Gaussian()]
)

# now build the GP model as normal
m = gpflow.models.VGP((X, Y), kernel=kern, likelihood=lik)

# fit the covariance function parameters
maxiter = ci_niter(10000)
gpflow.optimizers.Scipy().minimize(
    m.training_loss, m.trainable_variables, options=dict(maxiter=maxiter), method="L-BFGS-B",
)
print_summary(m)

def plot_gp(x, mu, var, color, label):
    plt.plot(x, mu, color=color, lw=2, label=label)
    plt.fill_between(
        x[:, 0],
        (mu - 2 * np.sqrt(var))[:, 0],
        (mu + 2 * np.sqrt(var))[:, 0],
        color=color,
        alpha=0.4,
    )


def plot(m):
    plt.figure(figsize=(8, 4))
    Xtest = np.linspace(0, 1, 100)[:, None]
    (line,) = plt.plot(x_1, y_1, "x", mew=2)
    mu, var = m.predict_f(np.hstack((Xtest, np.zeros_like(Xtest))))
    plot_gp(Xtest, mu, var, line.get_color(), "Y1")

    (line,) = plt.plot(x_2, y_2, "x", mew=2)
    mu, var = m.predict_f(np.hstack((Xtest, np.ones_like(Xtest))))
    plot_gp(Xtest, mu, var, line.get_color(), "Y2")

    plt.legend()
    plt.show()


plot(m)