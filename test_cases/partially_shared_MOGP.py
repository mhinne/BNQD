import gpflow as gpf
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from gpflow.kernels import Matern32, Coregion
from kernels import MultiOutputKernel

from gpflow.ci_utils import ci_niter

from gpflow.utilities import print_summary

plt.rcParams["figure.figsize"] = (12, 6)
np.random.seed(123)

# plt.rc('axes', titlesize=24)        # fontsize of the axes title
# plt.rc('axes', labelsize=18)        # fontsize of the x and y labels
# plt.rc('xtick', labelsize=12)       # fontsize of the tick labels
# plt.rc('ytick', labelsize=12)       # fontsize of the tick labels
# plt.rc('legend', fontsize=20)       # legend fontsize
# plt.rc('figure', titlesize=32)      # fontsize of the figure title

print('TensorFlow version', tf.__version__)
print('GPflow version    ', gpf.__version__)
# print('BNQD version      ', BNQD.__version__)

# Create list of kernels for each output
kern_list = [gpf.kernels.SquaredExponential() + gpf.kernels.Linear() for _ in range(P)]
# Create multi-output kernel from kernel list
kernel = gpf.kernels.SeparateIndependent(kern_list)
# initialization of inducing input locations (M random points from the training inputs)
Z = Zinit.copy()
# create multi-output inducing variables from Z
iv = gpf.inducing_variables.SharedIndependentInducingVariables(
    gpf.inducing_variables.InducingPoints(Z)
)
# create SVGP model as usual and optimize
m = gpf.models.SVGP(kernel, gpf.likelihoods.Gaussian(), inducing_variable=iv, num_latent_gps=P)
optimize_model_with_scipy(m)