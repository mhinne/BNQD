import gpflow
import tensorflow as tf
from gpflow.kernels import Kernel, Sum, Product, Combination
import tensorflow_probability as tfp

f64 = gpflow.utilities.to_default_float

import numpy as np

from scipy.integrate import cumtrapz
import scipy.signal as signal
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

"""
MOGP following GPflow standard implementation, see https://gpflow.readthedocs.io/en/master/notebooks/advanced/coregionalisation.html
"""


class IndependentKernel(Combination):
    """
    This is simply a wrapper around a base kernel in which covariances between elements at different sides of a threshold
    x0 (applied to the forcing variable) are set to 0 a priori.

    :param kernels: the base kernels applied to either side of the threshold
    :param x0: the threshold that determines which kernel is used
    :param forcing_variable: the dimension of the input X to which the threshold is applied
    """

    def __init__(self, kernels, x0, forcing_variable=0, name=None):

        self.x0 = x0
        self.forcing_variable = forcing_variable
        super().__init__(kernels, name=name)

    def K(self, X, X2=None):
        # threshold X, X2 based on self.x0, and construct a joint tensor
        if X2 is None:
            X2 = X

        fv = self.forcing_variable
        mask1 = tf.dtypes.cast(X[:, fv] >= self.x0, tf.int32)
        mask2 = tf.dtypes.cast(X2[:, fv] >= self.x0, tf.int32)

        X_partitioned = tf.dynamic_partition(X, mask1, 2)
        X2_partitioned = tf.dynamic_partition(X2, mask2, 2)

        K_pre = self.kernels[0].K(X_partitioned[0], X2_partitioned[0])
        K_post = self.kernels[1].K(X_partitioned[1], X2_partitioned[1])

        zero_block_1 = tf.zeros([tf.shape(K_pre)[0], tf.shape(K_post)[1]], tf.float64)
        zero_block_2 = tf.zeros([tf.shape(K_post)[0], tf.shape(K_pre)[1]], tf.float64)
        upper_row = tf.concat([K_pre, zero_block_1], axis=1)
        lower_row = tf.concat([zero_block_2, K_post], axis=1)

        return tf.concat([upper_row, lower_row], axis=0)

    #
    def K_diag(self, X):
        fv = self.forcing_variable
        mask = tf.dtypes.cast(X[:, fv] >= self.x0, tf.int32)

        X_partitioned = tf.dynamic_partition(X, mask, 2)
        return tf.concat([self.kernels[0].K_diag(X_partitioned[0]),
                          self.kernels[1].K_diag(X_partitioned[1])],
                         axis=0)

    #
#


def MultiOutputKernel(base_kernel, output_dim, rank):
    assert rank <= output_dim, 'Rank must be <= output dimension'

    # Coregion kernel
    coreg = gpflow.kernels.Coregion(output_dim=output_dim, rank=rank, active_dims=[1])
    return Product([base_kernel, coreg], name='MO_{:s}'.format(base_kernel.name))

#

"""
Spectral Mixture kernel by David Leeftink, see https://github.com/DavidLeeftink/Spectral-Discontinuity-Design
"""

def SpectralMixture(Q, mixture_weights=None, frequencies=None, lengthscales=None, max_freq=1.0, max_length=1.0,
                    active_dims=None, x=None, y=None, fs=1, q_range=(1, 10)):
    """
    Spectral Mixture kernel as proposed by Wilson-Adams (2013)
    Currently supports only 1 dimension.
    Parts of code inspired by implementations of:
    - Sami Remes (https://github.com/sremes/nssm-gp/blob/master/nssm_gp/spectral_kernels.py)
    - Srikanth Gadicherla (https://github.com/imsrgadich/gprsm/blob/master/gprsm/spectralmixture.py)
    :arg
    """
    if y is not None:
        frequencies, lengthscales, mixture_weights = initialize_from_emp_spec(Q, x, y, fs)

    else:
        if mixture_weights is None:
            mixture_weights = [1.0 for i in range(Q)]
        if frequencies is None:
            frequencies = [((i + 1) / Q) * max_freq for i in range(Q)]
        if lengthscales is None:
            lengthscales = [max_length for _ in range(Q)]

    components = [
        SpectralMixtureComponent(i + 1, mixture_weights[i], frequencies[i], lengthscales[i], active_dims=active_dims)
        for i in range(Q)]
    return Sum(components, name='Spectral_mixture')  # if len(components) > 1 else components[0]


def initialize_from_emp_spec(Q, x, y, fs, plot=False):
    """
    Initializes the Spectral Mixture hyperparameters by fitting a GMM on the empirical spectrum,
    found by Lombscargle periodogram.
    Function largely taken from: https://docs.gpytorch.ai/en/v1.1.1/_modules/gpytorch/kernels/spectral_mixture_kernel.html#SpectralMixtureKernel.initialize_from_data_empspect
    Instead, here the lom-sccargle periodogram is used to fit the GMM to allow analysis of ununiformly sampled data.

    :param Q (int) number of spectral components in SM kernel
    :param x (np.array of float64) X values of input data
    :param y NumPy array of float64. Y values of input data

    return: frequencies lengthscales, mixture weights, all of which are NumPy arrays of shape (Q,)
    """

    freqs = np.linspace(0.01, fs, 1000)
    Pxx = signal.lombscargle(x, y, freqs, normalize=False)

    if plot:
        fig = plt.figure(figsize=(8, 4))
        plt.plot(freqs, Pxx, color='blue')
        plt.show()

    total_area = np.trapz(Pxx, freqs)
    spec_cdf = np.hstack((np.zeros(1), cumtrapz(Pxx, freqs)))
    spec_cdf = spec_cdf / total_area

    a = np.random.rand(1000, 1)
    p, q = np.histogram(a, spec_cdf)
    bins = np.digitize(a, q)
    slopes = (spec_cdf[bins] - spec_cdf[bins - 1]) / (freqs[bins] - freqs[bins - 1])
    intercepts = spec_cdf[bins - 1] - slopes * freqs[bins - 1]
    inv_spec = (a - intercepts) / slopes

    GMM = GaussianMixture(n_components=Q, covariance_type="full")
    GMM.fit(X=inv_spec)
    means = GMM.means_
    varz = GMM.covariances_
    weights = GMM.weights_

    emp_frequencies, emp_lengthscales, emp_mixture_weights = means.flatten(), varz.flatten(), weights.flatten()
    lengthscales = 1 / np.sqrt(emp_lengthscales)
    mixture_weights = emp_mixture_weights
    frequencies = emp_frequencies

    return frequencies, lengthscales, mixture_weights


class SpectralMixtureComponent(Kernel):
    """
    Single component of the SM kernel by Wilson-Adams (2013).
    k(x,x') = w * exp(-2 pi^2 * |x-x'| * sigma_q^2 ) * cos(2 pi |x-x'| * mu_q)
    """

    def __init__(self, index, mixture_weight, frequency, lengthscale, active_dims):
        super().__init__(active_dims=active_dims)
        self.index = index

        def logit_transform(min, max):
            a = tf.cast(min, tf.float64)
            b = tf.cast(max, tf.float64)
            affine = tfp.bijectors.AffineScalar(shift=a, scale=(b - a))
            sigmoid = tfp.bijectors.Sigmoid()
            logistic = tfp.bijectors.Chain([affine, sigmoid])
            return logistic

        logistic = logit_transform(0.0001, 9000000)  # numerical stability

        self.mixture_weight = gpflow.Parameter(mixture_weight, transform=logistic)
        self.frequency = gpflow.Parameter(frequency, transform=logistic)
        self.lengthscale = gpflow.Parameter(lengthscale, transform=logistic)

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        tau_squared = self.scaled_squared_euclid_dist(X, X2)
        exp_term = tf.exp(-2.0 * (np.pi ** 2) * tau_squared)

        # Following lines are taken from Sami Remes' implementation (see references above)
        f = tf.expand_dims(X, 1)
        f2 = tf.expand_dims(X2, 0)
        freq = tf.expand_dims(self.frequency, 0)
        freq = tf.expand_dims(freq, 0)
        r = tf.reduce_sum(freq * (f - f2), 2)
        cos_term = tf.cos(r)

        return self.mixture_weight * exp_term * cos_term  # * 2 * np.pi

    def K_diag(self, X):
        return tf.fill(tf.shape(X)[:-1], tf.squeeze(self.mixture_weight))

    def scaled_squared_euclid_dist(self, X, X2=None):
        """
        Function to overwrite gpflow.kernels.stationaries
        Returns ||(X - X2ᵀ) / ℓ||² i.e. squared L2-norm.
        """
        X = X / self.lengthscale
        Xs = tf.reduce_sum(tf.square(X), axis=1)

        if X2 is None:
            dist = -2 * tf.matmul(X, X, transpose_b=True)
            dist += tf.reshape(Xs, (-1, 1)) + tf.reshape(Xs, (1, -1))
            return dist

        X2 = X2 / self.lengthscale
        X2s = tf.reduce_sum(tf.square(X2), axis=1)
        dist = -2 * tf.matmul(X, X2, transpose_b=True)
        dist += tf.reshape(Xs, (-1, 1)) + tf.reshape(X2s, (1, -1))
        return dist
