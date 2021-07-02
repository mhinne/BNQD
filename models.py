from gpflow.likelihoods import Likelihood, Gaussian
from gpflow.kernels import Kernel
from gpflow.mean_functions import MeanFunction
from gpflow.models import GPModel, GPR, VGP
from gpflow.utilities import triangular, deepcopy

from kernels import IndependentKernel


class ContinuousModel:

    def __init__(self,
                 data,
                 kernel: Kernel,
                 likelihood: Likelihood,
                 mean_function: MeanFunction):
        if isinstance(likelihood, Gaussian):
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
                 separate_kernels=False):
        self.x0 = x0
        self.forcing_variable = forcing_variable
        self.likelihood = likelihood

        if separate_kernels:
            # two kernel objects; i.e. different kernel hyperparameters pre and post x0
            kernel = IndependentKernel([kernel, deepcopy(kernel)],
                                       x0=self.x0,
                                       forcing_variable=self.forcing_variable,
                                       name='{:s}_indep'.format(kernel.name))
        else:
            # one kernel object; i.e. same kernel parameters pre and post x0
            kernel = IndependentKernel([kernel, kernel],
                                       x0=self.x0,
                                       forcing_variable=self.forcing_variable,
                                       name='{:s}_disc'.format(kernel.name))

        if isinstance(self.likelihood, Gaussian):
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