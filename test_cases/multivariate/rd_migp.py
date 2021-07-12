import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from BNQD import BNQD
from gpflow.likelihoods import Gaussian, Poisson, StudentT
from gpflow.kernels import SquaredExponential, Linear, Sum, Exponential, White

from kernels import SpectralMixture, IndependentKernel

import tensorflow as tf
import gpflow as gpf
from gpflow.utilities import print_summary
from utilities import plot_m0, plot_m1, plot_effect_size

plt.rc('axes', titlesize=24)        # fontsize of the axes title
plt.rc('axes', labelsize=18)        # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)       # fontsize of the tick labels
plt.rc('ytick', labelsize=12)       # fontsize of the tick labels
plt.rc('legend', fontsize=20)       # legend fontsize
plt.rc('figure', titlesize=32)      # fontsize of the figure title

print('TensorFlow version  ', tf.__version__)
print('NumPy version       ', np.__version__)
print('GPflow version      ', gpf.__version__)
print('BNQD version        ', BNQD.__version__)

jitter = 1e-6


def quarter_circle(X):
    return tf.dtypes.cast(X[:, 0]**2 + X[:, 1]**2 >= 0.5,
                          tf.int32)


def line(X):
    a = 1.0
    b = -1.0
    return tf.dtypes.cast(a + b*X[:, 0] >= X[:, 1],
                          tf.int32)


def square(X):
    return tf.dtypes.cast(np.logical_or((X[:, 0] - 0.5) < 0.5, (X[:, 1] - 0.5) < 0.5), tf.int32)


print('1D GP')

kernel = IndependentKernel([SquaredExponential(variance=1.0, lengthscales=0.7),
                            SquaredExponential(variance=1.0, lengthscales=0.05)],
                           x0=0.5, forcing_variable=0)


n = 50
X = np.linspace(0, 1, n)[:, np.newaxis]
K = kernel.K(X)
sigma = 0.1

L = np.linalg.cholesky(K + jitter*np.identity(n))
z = np.random.normal(size=X.shape[0])
f = np.dot(L, z)
y = np.random.normal(loc=f, scale=sigma)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))
axes[0].imshow(K)
axes[0].set_title('Kernel')
axes[1].plot(X, y, ls='none', marker='o')
axes[1].axvline(x=0.5, color='k')
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].set_title('GP sample')
plt.suptitle('1D threshold function')
plt.show()

m = gpf.models.GPR(data=(X, y[:, None]),
                   kernel=kernel)
opt = gpf.optimizers.Scipy()
opt.minimize(m.training_loss,
             variables=m.trainable_variables,
             options={'maxiter': 1000})

print_summary(m)

print('2D GP')
kernel = IndependentKernel([SquaredExponential(variance=1.0, lengthscales=0.7),
                            SquaredExponential(variance=1.0, lengthscales=0.09)],
                           split_function=quarter_circle)

p = 10

U = np.linspace(0, 1, p)
x1, x2 = np.meshgrid(U, U)
X = np.vstack([x1.flatten(), x2.flatten()]).T
N, D = X.shape

K = kernel.K(X)

sigma = 0.1
L = np.linalg.cholesky(K + jitter*np.identity(N))
z = np.random.normal(size=N)
f_vec = np.dot(L, z)
f = np.reshape(f_vec, [p, p])
y_vec = np.random.normal(loc=f_vec, scale=sigma)
y = np.reshape(y_vec, [p, p])

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))
axes[0].imshow(K)
axes[0].set_title('Kernel')
axes[1].imshow(f)
axes[1].set_xlabel('$x_1$')
axes[1].set_ylabel('$x_2$')
axes[1].set_title('GP sample')
plt.suptitle('2D arbitrary split function')
plt.show()

m = gpf.models.GPR(data=(X, y_vec.reshape(-1, 1)),
                   kernel=kernel)
opt = gpf.optimizers.Scipy()
opt.minimize(m.training_loss,
             variables=m.trainable_variables,
             options={'maxiter': 1000})
print_summary(m)
