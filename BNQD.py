import numpy as np
import gpflow as gpf

from typing import Union, Optional
from prettytable import PrettyTable
from tqdm import tqdm
from gpflow.likelihoods import Likelihood, Gaussian
from gpflow.kernels import Kernel
from gpflow.mean_functions import MeanFunction, Constant
from gpflow.utilities import deepcopy
from gpflow.optimizers import Scipy
from models import ContinuousModel, DiscontinuousModel
from utilities import renormalize


class BNQD():

    __version__ = '2.0.2'

    def __init__(self,
                 data,
                 likelihood: Likelihood = None,
                 kern_list: Union[Kernel, list] = None,
                 mean_function: Optional[MeanFunction] = Constant(),
                 intervention_pt=0.0,
                 forcing_variable=0,
                 split_function=None,
                 qed_mode='RD',
                 rank=None):
        """

        @type likelihood: GPflow Likelihood
        """

        # self.likelihood = likelihood
        assert kern_list is not None, 'Please provide at least one GP covariance function.'
        self.kernels = kern_list if isinstance(kern_list, list) else [kern_list]
        if likelihood is None:
            likelihood = Gaussian()

        self.mean_function = mean_function
        self.x0 = intervention_pt
        self.forcing_variable = forcing_variable
        self.split_function = split_function
        self.results = None
        self.mogp = False
        self.migp = False

        if type(data) is tuple:
            X, Y = data
            if np.ndim(X) < 2:
                X = np.atleast_2d(X).T
            if np.ndim(Y) < 2:
                Y = np.atleast_2d(Y).T
            if not isinstance(likelihood, Gaussian):
                Y = Y.astype(np.float)

            self.migp = X.shape[1] > 1
            p = 1
        elif type(data) is list:
            # Multi-output GP case
            p = len(data)
            # assume for now that if we have multiple outputs, we have only one input
            X = list()
            Y = list()
            for i in range(p):
                x, y = data[i]
                x, y = np.atleast_2d(x).T, np.atleast_2d(y).T
                X.append(np.hstack((x, i * np.ones_like(x))))
                Y.append(np.hstack((y, i * np.ones_like(y))))
            X = np.vstack(X)
            Y = np.vstack(Y)

            # Assume same likelihood for each of p outputs
            likelihood = gpf.likelihoods.SwitchedLikelihood(
                p*[likelihood]
            )
            self.mogp = True
            self.p = p
            if rank is None:
                rank = p
        else:
            raise NotImplementedError('Data must be a tuple (GP/MIGP) or list (MOGP)')

        self.data = (X, Y)

        self.is_trained = False
        self.M0, self.M1 = list(), list()

        # Set up models for each kernel
        for kernel in self.kernels:
            kern0 = deepcopy(kernel)
            mu0 = deepcopy(mean_function)
            m0_k = ContinuousModel(data=self.data, kernel=kern0, likelihood=likelihood, mean_function=mu0,
                                   multi_output=self.mogp, output_dim=p, rank=rank)

            kern1 = deepcopy(kernel)
            mu1 = deepcopy(mean_function)
            m1_k = DiscontinuousModel(data=self.data, kernel=kern1, likelihood=likelihood, mean_function=mu1,
                                      x0=self.x0, forcing_variable=forcing_variable, split_function=split_function,
                                      separate_kernels=qed_mode == 'ITS', multi_output=self.mogp, output_dim=p,
                                      rank=rank)

            self.M0.append(m0_k)
            self.M1.append(m1_k)

    #
    def __check_training_status(self):
        assert self.is_trained, 'Model must be trained first'

    #
    def train(self, opt=None, train_opts=None, verbose=False):
        assert not self.is_trained, 'Model is already trained'
        if opt is None:
            opt = Scipy()

        if train_opts is None:
            train_opts = dict()

        for k in tqdm(range(len(self.kernels))):
            for m in [self.M0[k], self.M1[k]]:
                opt_log = opt.minimize(m.objective, m.gpmodel.trainable_variables,
                                       options=dict(maxiter=train_opts.get('max_iter', 10000)))

        self.is_trained = True

    #
    def predict_y(self, x_new):
        if np.ndim(x_new) < 2:
            x_new = np.atleast_2d(x_new).T

        if self.mogp:
            predictions = list()
            for i in range(self.p):
                x_new_i = np.hstack([x_new, i*np.ones_like(x_new)])
                predictions_i = list()
                for k in range(len(self.kernels)):
                    mu0_k, var0_k = self.M0[k].predict_y(x_new_i)
                    mu1_k, var1_k = self.M1[k].predict_y(x_new_i)
                    predictions_k = ((mu0_k, var0_k), (mu1_k, var1_k))
                    predictions_i.append(predictions_k)
                predictions.append(predictions_i)
            return predictions

        predictions = list()
        for k in range(len(self.kernels)):
            mu0_k, var0_k = self.M0[k].predict_y(x_new)
            mu1_k, var1_k = self.M1[k].predict_y(x_new)
            predictions_k = ((mu0_k, var0_k), (mu1_k, var1_k))
            predictions.append(predictions_k)
        return predictions

    #
    def __get_evidence(self, mode='BIC'):
        self.__check_training_status()
        K = len(self.kernels)
        evidence = np.zeros((K, 2))
        if mode == 'BIC':
            for k in range(K):
                L = self.M0[k].log_marginal_likelihood().numpy()
                n = self.data[0].shape[0]
                p = len(self.M0[k].gpmodel.kernel.trainable_parameters) \
                    + len(self.M0[k].gpmodel.likelihood.trainable_parameters)
                evidence[k, 0] = L - p / 2.0 * np.log(n)

                L = self.M1[k].log_marginal_likelihood().numpy()
                # pre- and post split length of X
                # n = self.data_split[0][0].shape[0] + self.data_split[1][0].shape[0]
                p = len(self.M1[k].gpmodel.kernel.trainable_parameters) \
                    + len(self.M1[k].gpmodel.likelihood.trainable_parameters)

                evidence[k, 1] = L - p / 2.0 * np.log(n)
        else:
            raise NotImplementedError
        self.__evidence = evidence
        return self.__evidence

    #
    def get_evidence(self):
        return self.__evidence

    #
    def __get_bayes_factor(self):
        self.__check_training_status()
        lml = self.__get_evidence()
        self.__bayes_factor = (lml[:, 1] - lml[:, 0]).T
        return self.__bayes_factor

    #
    def get_bayes_factor(self):
        if hasattr(self, '__bayes_factor'):
            return self.__bayes_factor
        else:
            return self.__get_bayes_factor()

    #
    def __get_effect_sizes(self):
        self.__check_training_status()
        epsilon = 1e-6
        K = len(self.kernels)
        res = np.zeros((K, 2))
        for k in range(K):
            preds = self.M1[k].predict_y(np.atleast_2d([self.x0 - epsilon, self.x0 + epsilon]).T)
            res[k, 0] = preds[0][1] - preds[0][0]
            res[k, 1] = preds[1][0] + preds[1][1]
        self.__effect_sizes = res
        return self.__effect_sizes

    #
    def get_effect_sizes(self):
        return self.__effect_sizes

    #
    def __get_bma_effect_sizes(self):
        self.__check_training_status()
        es_m1 = self.__get_effect_sizes()
        pmp = self.__get_model_posterior()
        self.__bma_effect_sizes = es_m1[:, 0] * pmp[:, 1]
        return self.__bma_effect_sizes

    #
    def get_bma_effect_sizes(self):
        return self.__bma_effect_sizes

    #
    def __get_bma_effect_sizes_mcmc(self, nsamples=50000):
        K = len(self.kernels)
        pmp = self.get_model_posterior()
        es = self.get_effect_sizes()

        self.__es_mcmc = np.zeros((K+1, nsamples))

        for k in range(K):
            n_1 = int(np.round(nsamples * pmp[k][1]))
            if n_1 > 0:
                d_mu, d_var = es[k, :]
                d_sd = np.sqrt(d_var)
                self.__es_mcmc[k, 0:n_1] = np.random.normal(loc=d_mu, scale=d_sd, size=n_1)
    #

    def get_bma_effect_sizes_mcmc(self):
        return self.__es_mcmc

    #

    def get_total_bma_effect_sizes_mcmc(self, nsamples=50000):
        renorm_pmp = renormalize(self.get_evidence())
        es = self.get_effect_sizes()
        total_prob_d0 = np.sum(renorm_pmp[:, 0])
        K = len(self.kernels)

        es_total_bma = []

        # samples from models M1
        for k in range(K):
            n = int(np.round(nsamples * renorm_pmp[k, 1]))
            if n > 0:
                d_mu, d_var = es[k, :]
                d_sd = np.sqrt(d_var)
                es_total_bma.extend(np.random.normal(loc=d_mu, scale=d_sd, size=n))
        # zero samples from models M0
        es_total_bma.extend(np.zeros(int(np.round(nsamples * total_prob_d0))))
        return es_total_bma
    #

    def get_m1_bma_effect_sizes_mcmc(self, nsamples=50000):
        renorm_pmp = renormalize(self.get_evidence()[:, 0])
        es = self.get_effect_sizes()
        K = len(self.kernels)

        es_total_bma_m1 = []

        for k in range(K):
            n = int(np.round(nsamples * renorm_pmp[k]))
            if n > 0:
                d_mu, d_var = es[k, :]
                d_sd = np.sqrt(d_var)
                es_total_bma_m1.extend(np.random.normal(loc=d_mu, scale=d_sd, size=n))
        return es_total_bma_m1
    #


    def __get_model_posterior(self):
        self.__check_training_status()
        K = len(self.kernels)
        pmp = np.zeros((K, 2))
        log_bf = self.__get_bayes_factor()
        bf = np.exp(log_bf)
        pmp[:, 0] = 1 / (1 + bf)
        pmp[:, 1] = 1 - pmp[:, 0]
        self.__model_posterior = pmp
        return self.__model_posterior

    #
    def get_model_posterior(self):
        return self.__model_posterior

    #
    def __marginal_likelihoods(self):
        self.__check_training_status()
        pass

    #
    def get_results(self):
        self.__check_training_status()
        tab = PrettyTable()

        # List all kernels
        tab.add_column('Kernel', [k.name.capitalize() for k in self.kernels])

        # Bayes factors
        log_bf = self.__get_bayes_factor()
        tab.add_column('log BF', log_bf)

        # (approximated) log hyper marginal likelihood
        evidences = self.__get_evidence()
        tab.add_column('p(D | M0)', evidences[:, 0])
        tab.add_column('p(D | M1)', evidences[:, 1])

        # Posterior model probability
        pmp = self.__get_model_posterior()
        tab.add_column('p(M0 | D)', pmp[:, 0])
        tab.add_column('p(M1 | D)', pmp[:, 1])

        # Total BMA results (marginalized over kernels, assuming uniform kernel distribution)
        # BF_BMA = p(D|m1) / p(D|m0) = [\sum_k p(D|k,m1)p(k|m1)] / [\sum_k p(D|k,m0)p(k,m0)]
        # Note that we assume a uniform prior over k given m0, m1, hence we have:
        K = len(self.kernels)
        bma_evidence = np.mean(evidences, axis=0)
        bma_log_bf = bma_evidence[1] - bma_evidence[0]
        bma_pmp_m0 = 1 / (1 + np.exp(bma_log_bf))
        bma_pmp_m1 = 1 - bma_pmp_m0

        if not self.migp:
            # Effect sizes given M1
            es_M1 = self.__get_effect_sizes()
            tab.add_column('E[p(d | D, M1)]', es_M1[:, 0])
            tab.add_column('V[p(d | D, M1)]', es_M1[:, 1])

            # Effect sizes given BMA (expectation only)
            es_BMA = self.__get_bma_effect_sizes()
            tab.add_column('E[p(d | D)]', es_BMA)

            # BMA expectation and variance (easy because this is a GMM)
            p_D_k_m1 = renormalize(evidences[:, 1])
            bma_m1_d_exp = np.dot(p_D_k_m1, es_M1[:, 0])
            bma_m1_d_var = np.dot(p_D_k_m1, es_M1[:, 1])

            # total BMA expectation
            es_all = np.zeros((K, 2))
            es_all[:, 1] = es_M1[:, 0]
            p_D_k = renormalize(evidences.flatten())
            bma_es = np.dot(p_D_k, es_all.flatten())

            tab.add_row(['BMA',
                         bma_log_bf,
                         bma_evidence[0],
                         bma_evidence[1],
                         bma_pmp_m0,
                         bma_pmp_m1,
                         bma_m1_d_exp,
                         bma_m1_d_var,
                         bma_es])
        else:
            tab.add_row(['BMA',
                         bma_log_bf,
                         bma_evidence[0],
                         bma_evidence[1],
                         bma_pmp_m0,
                         bma_pmp_m1])


        # formatting settings
        tab.float_format = '0.3'
        tab.align = 'r'
        tab.align['Kernel'] = 'l'

        return tab

    #
#
