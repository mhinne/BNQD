from gpflow.likelihoods import Likelihood, Gaussian
from gpflow.kernels import Kernel
from gpflow.mean_functions import MeanFunction
from gpflow.models import GPModel, GPR, VGP
from gpflow.models.model import InputData, MeanAndVariance, RegressionData
from gpflow.utilities import triangular, deepcopy, positive
from gpflow.kullback_leiblers import gauss_kl
from gpflow.models.util import data_input_to_tensor, inducingpoint_wrapper
from gpflow.conditionals import conditional
from gpflow.config import default_float, default_jitter
from gpflow.base import Parameter
from typing import Optional
from kernels import IndependentKernel, MultiOutputKernel

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class ContinuousModel:

    def __init__(self,
                 data,
                 kernel: Kernel,
                 likelihood: Likelihood,
                 mean_function: MeanFunction,
                 multi_output=False,
                 output_dim=None,
                 rank=None,
                 variational_hyperparams=False):

        if multi_output:
            assert output_dim is not None and rank is not None, 'Output_dim and rank must be set in multi-output ' \
                                                                'context.'
            kernel = MultiOutputKernel(kernel, output_dim=output_dim, rank=rank)
        if variational_hyperparams:
            print(f'Using variational approximation for the Bayes factor')
            self.gpmodel = FullyBayesianGP(data=data,
                                           kernel=kernel,
                                           likelihood=likelihood,
                                           mean_function=mean_function)
        elif isinstance(likelihood, Gaussian) and not isinstance(kernel, MultiOutputKernel):
            print(f'Using exact GP regression model')
            self.gpmodel = GPR(data=data,
                               kernel=kernel,
                               mean_function=mean_function)
        else:
            print(f"Using variational GP model with ML-II hyperparameter optimization")
            self.gpmodel = VGP(data=data,
                               kernel=kernel,
                               likelihood=likelihood,
                               mean_function=mean_function)
        self.title = 'Continuous model'

    #
    def log_marginal_likelihood(self):
        if isinstance(self.gpmodel, GPR):
            return self.gpmodel.log_marginal_likelihood()
        else:
            return self.gpmodel.elbo()

    #
    def objective(self):
        return -1.0 * self.log_marginal_likelihood()

    #
    def predict_f(self, x_new, full_cov=False, full_output_cov=False):
        return self.gpmodel.predict_f(x_new,
                                      full_cov=full_cov,
                                      full_output_cov=full_output_cov)

    #
    def predict_y(self, x_new, full_cov=False, full_output_cov=False):
        f_mean, f_var = self.predict_f(x_new,
                                       full_cov=full_cov,
                                       full_output_cov=full_output_cov)
        return self.gpmodel.likelihood.predict_mean_and_var(f_mean, f_var)

    #
#


class DiscontinuousModel:
    def __init__(self,
                 data,
                 kernel: Kernel,
                 likelihood: Likelihood,
                 mean_function: MeanFunction,
                 x0=0.0,
                 forcing_variable=0,
                 split_function=None,
                 separate_kernels=False,
                 multi_output=False,
                 output_dim=None,
                 rank=None,
                 variational_hyperparams=False):
        self.x0 = x0
        self.forcing_variable = forcing_variable
        self.likelihood = likelihood
        self.split_function = split_function

        if separate_kernels:
            kernels = [kernel, deepcopy(kernel)]
        else:
            kernels = 2*[kernel]

        if split_function is None:
            kernel = IndependentKernel(kernels,
                                       x0=self.x0,
                                       forcing_variable=self.forcing_variable,
                                       name='{:s}_indep'.format(kernel.name))
        else:
            kernel = IndependentKernel(kernels,
                                       split_function=self.split_function,
                                       name='{:s}_indep'.format(kernel.name))

        if multi_output:
            assert output_dim is not None and rank is not None, 'Output_dim and rank must be set in multi-output ' \
                                                                'context.'
            kernel = MultiOutputKernel(kernel, output_dim=output_dim, rank=rank)

        if variational_hyperparams:
            self.gpmodel = FullyBayesianGP(data=data,
                                           kernel=kernel,
                                           likelihood=likelihood,
                                           mean_function=mean_function)
        if isinstance(self.likelihood, Gaussian) and not isinstance(kernel, MultiOutputKernel):
            self.gpmodel = GPR(data=data, kernel=kernel, mean_function=mean_function)
        else:
            self.gpmodel = VGP(data=data, kernel=kernel, likelihood=likelihood, mean_function=mean_function)
        self.title = 'Discontinuous model'

    #
    def log_marginal_likelihood(self):
        if isinstance(self.likelihood, Gaussian):
            return self.gpmodel.log_marginal_likelihood()
        else:
            return self.gpmodel.elbo()

    #
    def objective(self):
        return -1.0 * self.log_marginal_likelihood()

    #
    # todo: update these functions to work with new kernel formulation
    def predict_y(self, x_new, full_cov=False, full_output_cov=False):
        return self.gpmodel.predict_y(x_new,
                                      full_cov=full_cov,
                                      full_output_cov=full_output_cov)

    #
    def predict_f(self, x_new, full_cov=False, full_output_cov=False):
        return self.gpmodel.predict_f(x_new,
                                      full_cov=full_cov,
                                      full_output_cov=full_output_cov)

    #
#
class FullyBayesianGP(VGP):
    """
    Variational GP model that also learns a variational distribution over the hyperparameters
    with different likelihoods and inference techniques.
    """

    def __init__(self, data: RegressionData,
            kernel: Kernel,
            likelihood: Likelihood,
            mean_function: Optional[MeanFunction] = None,
            num_latent_gps: Optional[int] = None,
            full_rank = False):

        super().__init__(data, kernel, likelihood, mean_function, num_latent_gps)

        # Sparse variational distributions
        self.q_mu = Parameter(np.zeros((self.num_data, self.num_latent_gps)))
        q_sqrt = np.array([np.eye(self.num_data) for _ in range(self.num_latent_gps)])
        self.q_sqrt = Parameter(q_sqrt, transform=triangular())

        # Hyperparameter variational distributions (mean field gaussian approximation)
        self.hyper_dim = len(kernel.trainable_parameters)
        self.hyper_prior_mean = tf.zeros([self.hyper_dim])
        self.hyper_prior_scale = tf.ones([self.hyper_dim]) if not full_rank else tf.eye(self.hyper_dim)
        self.log_hyper_prior = tfp.distributions.Normal(loc=self.hyper_prior_mean,
                                                   scale=self.hyper_prior_scale)
        self.q_mu_hyper = Parameter(np.ones((self.num_data, self.hyper_dim)))
        self.q_sqrt_hyper = Parameter(np.ones([self.hyper_dim]), transform=positive())

        self.log_theta = tfp.distributions.Normal(loc=tf.ones(self.hyper_dim),
                                                  scale=tf.ones(self.hyper_dim))


    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        return self.elbo()

    def elbo(self) -> tf.Tensor:
        r"""
        This method computes the variational lower bound on the likelihood,
        which is:

            E_{q(F)} [ \log p(Y|F) ] - KL[ q(F) || p(F)]

        with

            q(\mathbf f) = N(\mathbf f \,|\, \boldsymbol \mu, \boldsymbol \Sigma)

        """
        X_data, Y_data = self.data

        # Get prior KL.
        print(self.q_mu, self.q_sqrt)
        # print(self.q_mu_hyper, self.q_sqrt_hyper)
        sparse_prior_KL = gauss_kl(self.q_mu, self.q_sqrt)
        hyper_prior_KL = tfp.distributions.kl.kl_divergence(self.log_theta, self.log_hyper_prior)
        print(hyper_prior_KL)
        assert 1==2

        KL = sparse_prior_KL + hyper_prior_KL # or should this be a minus sign?

        # Get conditionals
        K = self.kernel(X_data) + tf.eye(self.num_data, dtype=default_float()) * default_jitter()
        L = tf.linalg.cholesky(K)
        fmean = tf.linalg.matmul(L, self.q_mu) + self.mean_function(X_data)  # [NN, ND] -> ND
        q_sqrt_dnn = tf.linalg.band_part(self.q_sqrt, -1, 0)  # [D, N, N]
        L_tiled = tf.tile(tf.expand_dims(L, 0), tf.stack([self.num_latent_gps, 1, 1]))
        LTA = tf.linalg.matmul(L_tiled, q_sqrt_dnn)  # [D, N, N]
        fvar = tf.reduce_sum(tf.square(LTA), 2)

        fvar = tf.transpose(fvar)

        # Get variational expectations.
        print(f'fmean, fvar, Y-data shapes: {fmean.shape} , {fvar.shape}, {Y_data.shape}')

        var_exp = self.likelihood.variational_expectations(fmean, fvar, Y_data)

        return tf.reduce_sum(var_exp) - KL

    def predict_f(self, Xnew: InputData,
                  full_cov: bool = False,
                  full_output_cov: bool = False) -> MeanAndVariance:
        X_data, _ = self.data
        #log_hyper_sample = self.get_log_theta_sample(n_samples=1)
        #tf.print(f'log hyper sample shape: {log_hyper_sample.shape}, value: {log_hyper_sample}')
        #for i, log_hyper_param in enumerate(log_hyper_sample):
        #    self.kernel.trainable_parameters[i]= tf.exp(log_hyper_sample)

        mu, var = conditional(
            Xnew, X_data, self.kernel, self.q_mu, q_sqrt=self.q_sqrt, full_cov=full_cov, white=True,
        )
        return mu + self.mean_function(Xnew), var

    def get_log_theta_sample(self, n_samples=1):
        return self.log_theta.sample([n_samples])
