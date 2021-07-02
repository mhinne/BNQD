import numpy as np
import gpflow
from gpflow.kernels import Matern32
import matplotlib.pyplot as plt
from gpflow.utilities import print_summary
from kernels import IndependentKernel


def f(x):
    return np.sin(6*(x-0.7)) + (x>0.3)*1.0


x0 = 0.3
n = 100
x = np.linspace(0, 1, n)
sigma = 0.5
y = np.random.normal(loc=f(x), scale=sigma)
fv = 0
X = x[:, None]

kernel = IndependentKernel([Matern32(), Matern32()], x0=x0, name='indep')
x_pred = np.linspace(0, 1, 100)

K = kernel.K(x_pred[:, None])

m = gpflow.models.GPR(data=(X, y[:, None]), kernel=kernel)

gpflow.optimizers.Scipy().minimize(m.training_loss,
                                   m.trainable_variables,
                                   options=dict(maxiter=10000),
                                   method="L-BFGS-B")

print_summary(m)
x_pred = np.linspace(0, 1, 100)
mu, var = m.predict_y(x_pred[:, None])

fig, ax = plt.subplots(1, 1)
ax.plot(x, y, 'o')
ax.plot(x_pred, f(x_pred), ls='--', c='k')
ax.plot(x_pred, np.squeeze(mu), ls='-', c='k')
ax.fill_between(x_pred,
                mu.numpy().flatten() + 2*np.sqrt(var.numpy().flatten()),
                mu.numpy().flatten() - 2*np.sqrt(var.numpy().flatten()),
                alpha=0.2, color='k')
ax.axvline(x=x0, c='k')
plt.show()

