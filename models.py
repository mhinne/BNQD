from gpflow.likelihoods import Likelihood, Gaussian
from gpflow.kernels import Kernel
from gpflow.mean_functions import MeanFunction
from gpflow.models import GPModel, GPR, VGP
from gpflow.utilities import triangular, deepcopy, print_summary

from kernels import IndependentKernel, MultiOutputKernel


class ContinuousModel:

    def __init__(self,
                 data,
                 kernel: Kernel,
                 likelihood: Likelihood,
                 mean_function: MeanFunction,
                 multi_output=False,
                 output_dim=None,
                 rank=None):

        if multi_output:
            assert output_dim is not None and rank is not None, 'Output_dim and rank must be set in multi-output ' \
                                                                'context.'
            kernel = MultiOutputKernel(kernel, output_dim=output_dim, rank=rank)

        if isinstance(likelihood, Gaussian) and not isinstance(kernel, MultiOutputKernel):
            self.gpmodel = GPR(data=data,
                               kernel=kernel,
                               mean_function=mean_function)
        else:
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
                 rank=None):
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
    def __counterfactual_model(self):
        """
        Constructs the counterfactual GP based on the pre-intervention kernel.

        @return:
        """
        # pre-intervention kernel:
        (X, Y) = self.gpmodel.data
        X = X.numpy()
        Y = Y.numpy()
        A_ix = X[:, 0] < self.x0
        data_A = (X[A_ix, :], Y[A_ix, :])

        k_A = self.gpmodel.kernel.kernels[0]
        if isinstance(self.likelihood, Gaussian) and not isinstance(k_A, MultiOutputKernel):
            counterfactual_gp = GPR(data=data_A,
                                    kernel=k_A,
                                    mean_function=self.gpmodel.mean_function,
                                    noise_variance=self.gpmodel.likelihood.variance)
        else:
            counterfactual_gp = VGP(data=data_A,
                                    kernel=k_A,
                                    likelihood=self.gpmodel.likelihood,
                                    mean_function=self.gpmodel.mean_function)
        return counterfactual_gp


    def counterfactual_y(self, x_new, full_cov=False, full_output_cov=False):
        """
        Predicts the counterfactual response for x >= x0, based on the GP trained for x < x0.

        @param x_new: New input locations
        @param full_cov:
        @param full_output_cov:
        @return: Returns extrapolations/predictions for x >= x0
        """

        assert min(x_new) >= self.x0, 'Counterfactual predictions can only follow the intervention.'

        cf_gp = self.__counterfactual_model()

        print_summary(cf_gp)
        print_summary(self.gpmodel)

        return cf_gp.predict_y(x_new, full_cov=full_cov, full_output_cov=full_output_cov)

    #
    def counterfactual_f_samples(self, x_new, num_samples, full_cov=False, full_output_cov=False):
        """
        Samples the counterfactual response for x >= x0, based on the GP trained for x < x0.

        @param x_new: New input locations
        @param full_cov:
        @param full_output_cov:
        @return: Returns samples of extrapolations/predictions for x >= x0
        """
        assert min(x_new) >= self.x0, 'Counterfactual predictions can only follow the intervention.'

        cf_gp = self.__counterfactual_model()

        return cf_gp.predict_f_samples(x_new,
                                       num_samples=num_samples,
                                       full_cov=full_cov,
                                       full_output_cov=full_output_cov)

        #


#