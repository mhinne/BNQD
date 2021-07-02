import gpflow as gpf
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from gpflow.kernels import Matern32
from kernels import MultiOutputKernel

from gpflow.ci_utils import ci_niter

from gpflow.utilities import print_summary

plt.rcParams["figure.figsize"] = (12, 6)
np.random.seed(123)


print('TensorFlow version', tf.__version__)
print('GPflow version    ', gpf.__version__)


obs_model = 'g'
def f_1(x):
    return np.sin(6*x)


def f_2(x, x0):
    return np.sin(6*(x-0.1))


x0 = 0.5
xmin, xmax = 0, 1
n_1, n_2 = 50, 50
sigma = 0.3

x_1 = np.linspace(xmin, xmax, n_1)
x_2 = np.linspace(xmin, 0.7*xmax, n_2)

mu_1 = f_1(x_1)
mu_2 = f_2(x_2, x0)

y_1 = np.random.normal(loc=mu_1, scale=sigma)
y_2 = np.random.normal(loc=mu_2, scale=sigma)

colors = cm.get_cmap('tab10', 10)

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
rank = 2  # Rank of W

base_kernel = Matern32(active_dims=[0])
mogp_kernel = MultiOutputKernel(base_kernel=base_kernel, output_dim=p, rank=rank)

# This likelihood switches between Gaussian noise with different variances for each f_i:
lik = gpf.likelihoods.SwitchedLikelihood(
    [gpf.likelihoods.Gaussian(), gpf.likelihoods.Gaussian()]
)

# now build the GP model as normal
mean_function = gpf.mean_functions.Constant()

m = gpf.models.VGP((X, Y), kernel=mogp_kernel, likelihood=lik, mean_function=mean_function)

# fit the covariance function parameters
maxiter = ci_niter(10000)
gpf.optimizers.Scipy().minimize(
    m.training_loss, m.trainable_variables, options=dict(maxiter=maxiter), method="L-BFGS-B",
)
print_summary(m)
x_pred = np.linspace(xmin, xmax, 100)


def plot_gp(x, mu, var, color, label):
    ax.plot(x, mu, color=color, lw=2, label=label)
    ax.fill_between(
        x,
        (mu - 2 * np.sqrt(var))[:, 0],
        (mu + 2 * np.sqrt(var))[:, 0],
        color=color,
        alpha=0.4,
    )


fig, ax = plt.subplots(1,1, figsize=(8,4))

(line,) = ax.plot(x_1, y_1, "x", mew=2)
mu, var = m.predict_f(np.hstack((x_pred[:, None], np.zeros_like(x_pred[:, None]))))
plot_gp(x_pred, mu, var, line.get_color(), "Y1")

(line,) = plt.plot(x_2, y_2, "x", mew=2)
mu, var = m.predict_f(np.hstack((x_pred[:, None], np.ones_like(x_pred[:, None]))))
plot_gp(x_pred, mu, var, line.get_color(), "Y2")
ax.axvline(x=x0, c='k', ls=':')
ax.set_xlim([xmin, xmax])

plt.legend()
plt.show()


def cov2cor(S):
    v = np.sqrt(np.diag(S))
    outer_v = np.outer(v, v)
    R = S / outer_v
    R[S == 0] = 0
    return R


B = cov2cor(m.kernel.kernels[1].output_covariance().numpy())
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.imshow(B)
plt.show()

print(B)

