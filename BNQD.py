# -*- coding: utf-8 -*-
"""Bayesian non-parametric quasi-experimental design.

"""

import numpy as np
import GPy
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats
import bisect
import warnings
import pandas as pd

from matplotlib import rc

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
rc('font', size=12)
rc('font', family='serif')
rc('axes', labelsize=10)

__version__ = "0.1.3"
__author__  = "Max Hinne"

# TODO: 
# - plot BMA of discontinuity across kernels
# - clean up code
# - add code documentation
# - remove code redundancies
# - add transformations if needed, and transform back for plotting
# - set GP param priors, see https://gpy.readthedocs.io/en/deploy/_modules/GPy/examples/regression.html, add to BNPQED
# - default plots for 2D applications
# - optional parameter for end-user: labelfunc, labelLUT or just boundary b; hide switching behaviour
# - add more options to main analysis object (i.e. share_hyp flag, effect size n MCMC samples, etc.)



# Superclass only used as abstract class
class GPRegressionModel():
    
        
    def __init__self(x, y, kernel, lik=GPy.likelihoods.Gaussian()):
        raise NotImplementedError
    
    def train(self, num_restarts=10, verbose=False):
        raise NotImplementedError
        
    def predict(self, x_test):
        raise NotImplementedError
        
    def get_log_marginal_likelihood(self, mode='BIC'):
        raise NotImplementedError
        
    def plot(self, x_test, axis=None, color=None):
        raise NotImplementedError


class ContinuousModel(GPRegressionModel):
    
    
    isOptimized = False
    def __init__(self, x, y, kernel, lik=GPy.likelihoods.Gaussian()):
        self.x = x
        self.y = y
        self.n = x.shape[0]
        self.kernel = kernel.copy()        
        # Manual construction sometimes adds Gaussian white noise, 
        # sometimes does not???
#        self.m = GPy.core.GP(X = x, Y = y, kernel = self.kernel, likelihood = lik)
        self.m = GPy.models.GPRegression(X = x, Y = y, kernel = self.kernel)
        self.ndim = np.ndim(x)
        self.BICscore = None
    #    
    def train(self, num_restarts=10, verbose=False):
        """Train the continuous model        
        """
        self.m.optimize_restarts(num_restarts=num_restarts, verbose=verbose)
        self.isOptimized = True
    #    
    def predict(self, x_test):
        if len(x_test.shape) == 1:
            x_test = np.atleast_2d(x_test).T
        return self.m.predict(x_test, kern=self.m.kern.copy())
    #
    def get_log_marginal_likelihood(self, mode='BIC'):
        """Computes the log marginal likelihood for the continuous model. Since 
        this is intractable, we instead approximate it.
        
        :param mode: Selects how to approximate the evidence. Currently, only
        BIC is implemented, which is a crude approximation, but works well in
        our examples and simulations. 
        
        :return: Returns log p(D|M).
        """
        if mode == 'BIC':
            if not self.isOptimized:
                print('Parameters have not been optimized; training now')
                self.train()
            
            if self.BICscore is None:
                k = self.m.num_params
                L = self.m.log_likelihood()
                BIC = L - k/2*np.log(self.n)
                self.BICscore = BIC
            return self.BICscore
        elif mode in ['laplace', 'Laplace']:
            raise NotImplementedError('Laplace approximation is not yet implemented')
        elif mode == 'AIS':
            raise NotImplementedError('Annealed importance sampling is not yet implemented')
        else:
            raise NotImplementedError('Unrecognized marginal likelihood approximation {:s}'.format(mode))
    #
    def plot(self, x_test, axis=None, plotOptions=dict(), scaleFunction=None, 
             scaleData=None):
        if axis is None:
            axis = plt.gca()
           
        color = plotOptions.get('color', 'darkgreen')
        alpha = plotOptions.get('alpha', 0.3)
        linestyle = plotOptions.get('linestyle', 'solid')
        label = plotOptions.get('label', 'Optimized prediction')
            
        mu, Sigma2 = self.predict(x_test)         
        Sigma = np.sqrt(Sigma2)
        
        lower = np.squeeze(mu - 0.5*Sigma)
        upper = np.squeeze(mu + 0.5*Sigma)
        
        if scaleFunction is not None:
            mu = scaleFunction(mu, scaleData)
            lower = scaleFunction(lower, scaleData)
            upper = scaleFunction(upper, scaleData)
            
            
        if self.kernel.input_dim == 1:
            axis.plot(x_test, mu, label=label, color=color, 
                      linestyle=linestyle)
            axis.fill_between(x_test, lower, upper, alpha=alpha, color=color, 
                              edgecolor='white')
        elif self.kernel.input_dim == 2:
            p = int(np.sqrt(x_test.shape[0]))
            x0 = np.reshape(x_test[:,0], newshape=(p,p))
            x1 = np.reshape(x_test[:,1], newshape=(p,p))
            mu_res = np.reshape(mu, newshape=(p,p))
            axis.plot_surface(X=x0, Y=x1, Z=mu_res, color=color, 
                              antialiased=True, alpha=0.5, linewidth=0)
            
            axis.grid(False)
            axis.xaxis.pane.set_edgecolor('black')
            axis.yaxis.pane.set_edgecolor('black')
            axis.xaxis.pane.fill = False
            axis.yaxis.pane.fill = False
            axis.zaxis.pane.fill = False
        else:
            raise('Dimensionality not implemented')
    #               
    
    
class DiscontinuousModel():
    
    
    isOptimized = False
    
    def __init__(self, x, y, kernel, labelFunc=None, labelLUT=None,
                 lik=GPy.likelihoods.Gaussian()):
        
        self.ndim = np.ndim(x)
        if self.ndim==2 and x.shape[1]==1:
            self.ndim=1
        self.labelFunc = labelFunc
        self.labelLUT = labelLUT
        if self.labelFunc is None:
            lab1 = self.labelLUT==0
        else:
            lab1 = labelFunc(x)
        lab2 = np.logical_not(lab1)
        
        # ugly Numpy behaviour
        x1 = x[lab1,]
        x2 = x[lab2,]
        y1 = y[lab1,]
        y2 = y[lab2,]
        
        if len(x1.shape)==1:
            x1 = np.expand_dims(x1, axis=1)
        if len(x2.shape)==1:
            x2 = np.expand_dims(x2, axis=1)
        if len(y1.shape)==1:
            y1 = np.expand_dims(y1, axis=1)
        if len(y2.shape)==1:
            y2 = np.expand_dims(y2, axis=1)
            
        model1 = ContinuousModel(x1, y1, kernel, lik=lik)
        model2 = ContinuousModel(x2, y2, kernel, lik=lik)
            
        self.models = list()
        self.models.append(model1)
        self.models.append(model2)
        self.BICscore = None
    #    
    def train(self, num_restarts, share_hyp=True):
        """Train the hyperparameters of the dichotomous model. By default, 
        these are then averaged to ensure they do not depend on a discontinuity
        that is not at the threshold. 
        
        :param share_hyp: If true, hyperparameters are shared between the pre-
        and post model.
        
        """
        param_list = list()
        for model in self.models:
            model.train(num_restarts=num_restarts)
            param_list.append(model.m.param_array)
        
        if share_hyp:
            mean_hyp = np.mean(np.array(param_list), axis=0)
            for model in self.models:
                model.m[:] = mean_hyp
        
        self.isOptimized = True        
    #    
    def predict(self, x_test, mask=True):
        """Predict the values for the test predictors for each of the two sub-
        models. 
        
        :param mask: If true, only the predictions for the corresponding model
        are shown; otherwise the predictions for the full range of x_test are
        given.
        
        :return: GP predictions, means and variances, per model.
        """
        if np.isscalar(x_test):
            return (self.models[0].predict(np.atleast_2d(np.array(x_test))), 
                    self.models[1].predict(np.atleast_2d(np.array(x_test))))
        else:                
            if mask:
                lab1 = self.labelFunc(x_test)
                lab2 = np.logical_not(lab1) 
                return (self.models[0].predict(x_test[lab1,]), 
                        self.models[1].predict(x_test[lab2,]))
            else:
                return (self.models[0].predict(x_test), 
                        self.models[1].predict(x_test))
    #
    def get_log_marginal_likelihood(self, mode='BIC'):
        """Computes the log marginal likelihood for the dichotomous model. 
        Since this is intractable, we instead approximate it.
        
        :param mode: Selects how to approximate the evidence. Currently, only
        BIC is implemented, which is a crude approximation, but works well in
        our examples and simulations. 
        
        :return: Returns log p(D|M).
        """
        if mode == 'BIC':
            if not self.isOptimized:
                print('Parameters have not been optimized; training now')
                self.train()
            if self.BICscore is None:
                BIC = 0
                for i, model in enumerate(self.models):
                    n = model.n                    
                    k = model.m.num_params
                    L = model.m.log_likelihood()
                    BIC += L - k/2*np.log(n)
                self.BICscore = BIC
            return self.BICscore
        elif mode in ['laplace', 'Laplace']:
            raise NotImplementedError('Laplace approximation is not yet implemented')
        elif mode == 'AIS':
            raise NotImplementedError('Annealed importance sampling is not yet implemented')
        else:
            raise NotImplementedError('Unrecognized marginal likelihood approximation {:s}'.format(mode))
    #
    def plot(self, x_test, axis=None, plotOptions=None, b=0.0, 
             plotEffectSize=False, scaleFunction=None, scaleData=None,
             plotFullRange=False):
        if axis is None:
            axis = plt.gca()
        
        def add_boundary(x, b):
            if not np.isin(b, x):
                tmp = list(x)
                bisect.insort(tmp, b)
                return np.array(tmp)
            else:
                return x
        
        if plotOptions is None:
            plotOptions = [dict(), dict()]
        ms1 = plotOptions[0].get('markersize', 10)
        ms2 = plotOptions[1].get('markersize', 10)
        
        if not plotFullRange:
            if self.labelFunc is None:
                lab1 = self.labelLUT==0
            else:
                lab1 = np.array([self.labelFunc(i) for i in x_test])
#            lab1 = x_test < b
            lab2 = np.logical_not(lab1)
            x1 = x_test[lab1,]
            x2 = x_test[lab2,]
        else:
            x1 = x_test
            x2 = x_test
            
        # for printing purposes mainly
        x1 = add_boundary(x1, b)
        x2 = add_boundary(x2, b)
            
        if self.ndim==1:
            self.models[0].plot(x1, axis=axis, plotOptions=plotOptions[0], 
                       scaleFunction=scaleFunction, scaleData=scaleData)
            self.models[1].plot(x2, axis=axis, plotOptions=plotOptions[1], 
                       scaleFunction=scaleFunction, scaleData=scaleData)
            m0b, v0b = self.models[0].predict(np.array([b]))
            
            m1b, v1b = self.models[1].predict(np.array([b]))
            
            if scaleFunction is not None:
                m0b = scaleFunction(m0b, scaleData)
                v0b = scaleFunction(v0b, scaleData)
                m1b = scaleFunction(m1b, scaleData)
                v1b = scaleFunction(v1b, scaleData)
            
            if plotEffectSize:
                axis.plot([b,b], [np.squeeze(m0b), np.squeeze(m1b)], c='k', 
                          linestyle='-', marker=None, linewidth=3.0, zorder=10)
                axis.plot(b, m0b, c='k', marker='o', markeredgecolor='k', 
                          markerfacecolor='lightgrey', ms=ms1, zorder=10)
                axis.plot(b, m1b, c='k', marker='o', markeredgecolor='k',
                          markerfacecolor='lightgrey', ms=ms2, zorder=10)
            
            return (m0b, v0b), (m1b, v1b)
            
        elif self.ndim==2:
            
            mu1, _ = self.models[0].predict(x1)
            mu2, _ = self.models[1].predict(x2)
            
            p = int(np.sqrt(x_test.shape[0]))
            mu1_aug = np.zeros((p*p,1))
            mu1_aug.fill(np.nan)
            mu1_aug[lab1,] = mu1
            mu1_aug = np.reshape(mu1_aug, newshape=(p,p))
            
            mu2_aug = np.zeros((p*p,1))
            mu2_aug.fill(np.nan)
            mu2_aug[lab2,] = mu2
            mu2_aug = np.reshape(mu2_aug, newshape=(p,p))
            
            x0 = np.reshape(x_test[:,0], newshape=(p,p))
            x1 = np.reshape(x_test[:,1], newshape=(p,p))
            axis.plot_surface(X=x0, Y=x1, Z=mu1_aug, 
                              color=plotOptions[0]['color'], antialiased=True, 
                              alpha=0.5, linewidth=0)
            axis.plot_surface(X=x0, Y=x1, Z=mu2_aug, 
                              color=plotOptions[1]['color'], antialiased=True, 
                              alpha=0.5, linewidth=0)
            
            axis.grid(False)
            axis.xaxis.pane.set_edgecolor('black')
            axis.yaxis.pane.set_edgecolor('black')
            axis.xaxis.pane.fill = False
            axis.yaxis.pane.fill = False
            axis.zaxis.pane.fill = False
        else:
            raise('Dimensionality not implemented')
    #
    
            
class BnpQedModel():
    """The analysis object for one single kernel. Includes a continuous and a
    discontinuous model.
    """
    
    
    isOptimized = False
    log_BF_10 = None
    summary_object = None
    BFmode = ''
    
    def __init__(self, x, y, kernel, labelFunc=None, labelLUT=None, mode='BIC', design='Generic'):
        
        self.x = x
        self.y = y
        self.ndim = np.ndim(x)
        self.labelFunc = labelFunc
        self.labelLUT = labelLUT
        
        if np.ndim(x) == 1 and len(x.shape) == 1:
            x = np.atleast_2d(x).T
        
        if len(y.shape) == 1:
            y = np.atleast_2d(y).T  
            
        self.CModel = ContinuousModel(x, y, kernel.copy())
        self.DModel = DiscontinuousModel(x, y, kernel.copy(), labelFunc, labelLUT)
        self.BFmode = mode
        self.design = design
    #    
    def train(self, num_restarts=10, b=0.0):
        """Train both the continuous and the discontinuous model using GPy.
        
        We use the default implementation of GPy, which is L-BFGS.  
        Multiple restarts are used to avoid settling for a local optimum.

        :param num_restarts: scalar
        :return:    The BNP-QED summary object containing statistics of the 
                    model comparison.
        """
        
        self.CModel.train(num_restarts=num_restarts)
        self.DModel.train(num_restarts=num_restarts)
        self.isOptimized = True
        return self.summary(mode=self.BFmode, b=b)
    #   
    def predict(self, x_test):
        """ Predicting the responses of either model for unseen X.
        
        :param x_test: Range of predictor values, used for either interpolating
                       or extrapolating.
        :return:    The posterior predictive mean and variance for every point 
                    in x_test, for 
                        1) the continuous model and 
                        2) BOTH discontinuous models.
        """
        
        return self.CModel.predict(x_test), self.DModel.predict(x_test)
    #
    def get_log_Bayes_factor(self, mode='BIC'):
        """ The Bayes factor given the specified data and kernel.
        
        The Bayes factor is defined as
            BF_DC = p(D | M_D) / p(D | M_C),
        where M_D, M_C represent the discontinuous and continuous models, 
        respectively.
        
        For numerical stability, we compute the logarithm of the Bayes factor.
        
        :param mode: The approximation strategy for computing model evidence.
        :return: The log Bayes factor
        """
        
        if not self.isOptimized:
            self.train()      
        if self.log_BF_10 is None:  
            self.log_BF_10 = self.DModel.get_log_marginal_likelihood(mode=mode) \
            - self.CModel.get_log_marginal_likelihood(mode=mode)
        return self.log_BF_10
    #
    def discEstimate(self, b=0.0):
        """The predictions of the discontinuous model at the boundary value b.
        
        :param b: The boundary value. Can be called repeatedly for multi-
        dimensional discontinuity estimates.
        :return: Returns a tuple of means and a typle of variances for the two
        Gaussian distributions that correspond to the predictions by the
        discontinuous model at b.
        
        """
        m0b, v0b = self.DModel.models[0].predict(np.array([b]))
        m1b, v1b = self.DModel.models[1].predict(np.array([b]))
        return (m0b, m1b), (v0b, v1b)
    #
    def get_posterior_model_probabilities(self, mode='BIC'):
        """Computes the posterior model probabilities p(M|D)
        
        We have:
            p(M_D|D) / P(M_C|D) = p(D|M_D) / p(D|M_C) p(M_D) / M(M_C)
        and
            p(M_D|D) + p(M_C|D) = 1
        so if we assume the prior model probabilities are equal, we have:
            p(M_D|D) / (1 - p(M_D|D)) = p(D|M_D) / p(D|M_C)
            and
            BF_DC = p(D|M_D) / p(D|M_C)
            -> p(M_D|D) = BF_DC / (1 + BF_DC)
           
        :param mode: The approximation method for the marginal likelihood.
        :return: A dict with of posterior model probability per model.
        """
        # Note: assumes uniform prior!
        bf = np.exp(self.get_log_Bayes_factor(mode))
        if np.isinf(bf):
            return {'pmc': 0.0, 'pmd': 1.0}
        else:
            pmd = bf / (1+bf)
            pmc = 1 - pmd
            return {'pmc': pmc, 'pmd': pmd}
    #   
    def get_effect_size(self, summ, b, nmc=5000):
        """Computes the effect size at the boundary b. The BMA of the effect 
        size is approximated using Monte Carlo sampling and Gaussian kernel 
        density estimation.
        
        Note that this measure of effect size applies only to zeroth-order
        discontinuities, i.e. regression discontinuity.
        
        :param summ: the summary object of the analysis.
        :param b: the boundary at which to compute the effect size
        :param nmc: number of Monte Carlo samples for the BMA density estimate.
        
        :return: Returns a dictionary containing 
        - the effect size estimate by the discontinuous model as a pdf
        - the effect size estimate by the discontinuous model as summary 
        statistics 
        - The BMA effect size estimate.
        - The two-step p-value (i.e. frequentist p-value given the effect size
        distribution by the discontinuous model).
        - The range over which the effect size distribution is given, used for
        plotting.
        - The mean predictions at b.
        - The normalization from standardized effect size to the scale of the 
        data.
        """
        m0b, v0b = self.DModel.models[0].predict(np.array([b]))            
        m1b, v1b = self.DModel.models[1].predict(np.array([b]))
                    
        d_mean_D = np.squeeze(m1b - m0b) # TODO: why was this swapped around?
        d_var_D = np.squeeze(v0b + v1b)
        d_std_D = np.sqrt(d_var_D)
        
        if d_mean_D < 0:
            pval = 1 - stats.norm.cdf(x=0, loc=d_mean_D, scale=d_std_D)
        else:
            pval = stats.norm.cdf(x=0, loc=d_mean_D, scale=d_std_D)
        
        xmin, xmax = (np.min([d_mean_D - 4*d_std_D, -0.1*d_std_D]), 
                      np.max([d_mean_D + 4*d_std_D, 0.1*d_std_D]))
        
        n = 300
        xrange = np.linspace(xmin, xmax, n)
        y = stats.norm.pdf(xrange, d_mean_D, d_std_D)   
        
        samples = np.zeros((nmc))
        nspike = int(np.round(summ['pmp']['pmc']*nmc))
        samples[nspike:] = np.random.normal(loc=d_mean_D, 
                                            scale=np.sqrt(d_var_D), 
                                            size=(nmc-nspike))
        
        if not np.isscalar(b):
            d_bma = None
        else:
        
            if nspike==nmc:
                # BMA dominated by continuous model
                # Put all mass at xrange closest to b
                d_bma = np.zeros((n))
                xdelta = xrange[1] - xrange[0]
                ix = np.argmin((xrange-b)**2)
                d_bma[ix] = 1.0 / xdelta
            elif nspike==0:
                # BMA dominated by discontinuous model
                d_bma = y
            else:
                # BMA is a mixture
                kde_fit = stats.gaussian_kde(samples, 
                                             bw_method='silverman')
                d_bma = kde_fit(xrange)
               
        return {'es_BMA': d_bma,
                'es_Disc': y,
                'es_disc_stats': (d_mean_D, d_std_D),
                'pval': pval,
                'es_range': xrange,
                'f(b)': (m0b, m1b),
                'es_transform': lambda z: z*d_std_D + d_mean_D}
    #
        
        
    def summary(self, mode='BIC', b=0.0):
        """A function aggregating all derived statistics after the models have
        been trained.
        
        :param mode: The approximation method for the marginal likelihood.
        :param b: The boundary value; assumes a threshold label function. Will
                    be made generic later.
        :return: A dictionary containing:
            - logbayesfactor
            - evidence (marginal likelihoods)
            - pmp (posterior model probabilities)
            And for ndim == 1:
            - es_BMA (effect size across M_D and M_C)
            - es_Disc (effect size for M_D)
            - pval (p-value of discontinuity; uses a z-test)
            - es_range (the range over which we estimate a density of the
                           effect size)
            - f(b) (the predictions at b for the two models in M_D)
            - es_transform (the transformation from standardized units to the 
                            actual effect size)
        """
        if self.summary_object is None:
            if mode is None:
                mode = self.BFmode
            summ = dict()
            summ['logbayesfactor'] = self.get_log_Bayes_factor(mode)
            summ['evidence'] = \
            {'mc': self.CModel.get_log_marginal_likelihood(mode), 
             'md': self.DModel.get_log_marginal_likelihood(mode)}
            summ['pmp'] = self.get_posterior_model_probabilities(mode)
                        
            if self.ndim == 1 and self.design != 'DiD':
                # compute effect size  
                    
                es = self.get_effect_size(summ, b)
                for k, v in es.items():
                    summ[k] = v
                    
            elif self.ndim == 2 and self.design != 'DiD':       
                warnings.warn('Computing 2D effect size with Monte Carlo may take a while.')
                
                m = len(b)
                es = {i: self.get_effect_size(summ, b[i]) for i in range(m)}
                    
                for k, v in es[0].items():
                    summ[k] = [es[i][k] for i in range(m)]
                
            else:
                 warnings.warn('Effect size analysis for D = {:d} not implemented.'.format(self.ndim))
            self.summary_object = summ
        return self.summary_object    
    #
    def plot(self, x_test, axis=None, b=0.0, plotEffectSize=False, mode='BIC'):           
        summary = self.summary(mode=mode, b=b)        
        pmc, pmd = summary['pmp']['pmc'], summary['pmp']['pmd']        
        LBF = summary['logbayesfactor']
                
        if self.ndim == 1:
            if plotEffectSize:
                fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, 
                                                    figsize=(16,6))
            else:
                fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, 
                                               figsize=(16,6), 
                                               sharex=True, sharey=True)
            if self.labelFunc is None:
                lab1 = self.labelLUT==0
            else:
                lab1 = self.labelFunc(self.x)
            lab2 = np.logical_not(lab1)
            ax1.plot(self.x[lab1], self.y[lab1], linestyle='', marker='o', 
                     color='k')
            ax1.plot(self.x[lab2], self.y[lab2], linestyle='', marker='x', 
                     color='k')
            self.CModel.plot(x_test, ax1)
            ax1.axvline(x = b, color='black', linestyle=':')
            ax1.set_title(r'Continuous model, $p(M_C \mid x)$ = {:0.2f}'.format(pmc))
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            ax1.set_xlim([self.x[0], self.x[-1]])
            
            ax2.plot(self.x[lab1], self.y[lab1], linestyle='', marker='o', 
                     color='k')
            ax2.plot(self.x[lab2], self.y[lab2], linestyle='', marker='x', 
                     color='k')
            ax2.axvline(x = b, color='black', linestyle=':')
            m0stats, m1stats = self.DModel.plot(x_test, 
                                                ax2, 
                                                [{'colors': ('firebrick', 'firebrick')},
                                                  {'colors': ('firebrick', 'firebrick')}], 
                                                b=b, 
                                                plotEffectSize=plotEffectSize)            
            ax2.set_title(r'Discontinuous model, $p(M_D \mid x)$ = {:0.2f}'.format(pmd))
            ax2.set_xlabel('x')
            ax2.set_ylabel('y')    
            ax2.set_xlim([self.x[0], self.x[-1]])
            
            
            if plotEffectSize:  
                # create ES plot                
                xmin, xmax = summary['es_interval']                
                n = 100
                xrange = np.linspace(xmin, xmax, n)
                y = summary['es_Disc']
                pval = summary['pval']
                d_bma = summary['es_BMA']
                ax3.plot(xrange, y, c='firebrick', label=r'$M_D$', 
                         linewidth=2.0, linestyle='--')
                ax3.fill_between(xrange, y, np.zeros((n)), alpha=0.1, 
                                 color='firebrick')
                ax3.axvline(x=0, linewidth=2.0, label=r'$M_C$', 
                            color='darkgreen', linestyle='--')
                ax3.plot(xrange, d_bma, c='k', label=r'BMA', linewidth=2.0)
                ax3.fill_between(xrange, d_bma, np.zeros((n)), alpha=0.1, 
                                 color='k')
                ax3.legend(loc='best')
                ax3.set_xlabel(r'$\delta$')
                ax3.set_ylabel('Density')
                ax3.set_title(r'Size of discontinuity ($p$ = {:0.3f})'.format(pval))
                ax3.set_ylim(bottom=0)
                ax3.set_xlim([xmin, xmax])
            
            fig.suptitle(r'GP RDD analysis, log BF10 = {:0.4f}'.format(LBF))
            if plotEffectSize:
                return fig, (ax1, ax2, ax3)
            else:
                return fig, (ax1, ax2)
        elif self.ndim == 2:
            fig = plt.figure(figsize=(14,6))
            
            ax1 = fig.add_subplot(1, 2, 1, projection='3d')
            if self.labelFunc is None:
                lab1 = self.labelLUT==0
            else:
                lab1 = self.labelFunc(self.x)
            lab2 = np.logical_not(lab1)
            ax1.scatter(self.x[lab1,0], self.x[lab1,1], self.y[lab1,], 
                        marker='o', c='black')
            ax1.scatter(self.x[lab2,0], self.x[lab2,1], self.y[lab2,], 
                        marker='x', c='black')
            self.CModel.plot(x_test, ax1)
            ax1.set_title('Continuous model, p(M|x) = {:0.2f}'.format(pmc))
            ax1.set_xlabel(r'$x_1$')
            ax1.set_ylabel(r'$x_2$')
            ax1.set_zlabel('y')
            
            ax2 = fig.add_subplot(1, 2, 2, projection='3d')
            ax2.scatter(self.x[lab1,0], self.x[lab1,1], self.y[lab1,], 
                        marker='o', c='black')
            ax2.scatter(self.x[lab2,0], self.x[lab2,1], self.y[lab2,], 
                        marker='x', c='black')
            ax2.set_xlabel(r'$x_1$')
            ax2.set_ylabel(r'$x_2$')
            ax2.set_zlabel('y')
            self.DModel.plot(x_test, ax2, colors=('firebrick', 'coral'))
            ax2.set_title('Continuous model, p(M|x) = {:0.2f}'.format(pmd))
            fig.suptitle('GP RDD analysis, log BF10 = {:0.4f}'.format(LBF))
            return fig, (ax1, ax2)
        else:
            raise NotImplementedError('Dimensionality not implemented')
    #
    

class BnpQedAnalysis():
    
    
    def __init__(self, x, y, kernel_dict, labelFunc=None, labelLUT=None, b=0.0, opts=dict()):
        
        self.x = x
        self.y = y
        self.kernel_dict = kernel_dict
        self.K = len(kernel_dict)
        self.ndim = np.ndim(x)
        self.labelFunc = labelFunc  # function to label any point
        self.labelLUT = labelLUT    # look-up-table for provided points
        
        assert labelFunc is not None or labelLUT is not None, 'Provide either a label function or look-up-table'
        self.b = b
        self.num_restarts = opts.get('num_restarts', 10)
        self.mode = opts.get('mode', 'BIC')
        self.verbose = opts.get('verbose', True)
        self.trained = False
        self.results = dict()
        self.total_disc_es = None
        self.total_disc_pdf = None
        self.total_bma_es = None
        self.total_bma_pdf = None
        self.rdd_p_values = None
        self.design = opts.get('Design', 'Generic')
    #       
    def train(self):
        """Trains the different models.
        
        Trains the continuous and discontinuous models for each of the provided
        kernels.
        """
        
        for kernel_name, kernel in self.kernel_dict.items():
            if self.verbose: print('Training with {:s} kernel'.format(kernel_name))
            model = BnpQedModel(self.x, self.y, kernel, self.labelFunc, 
                                self.labelLUT, self.mode, self.design)
            model.train(num_restarts=self.num_restarts, b=self.b) 
            if self.verbose:
                print('Log Bayes factor in favor of discontinuity = {:0.2f}'.format(model.summary(b=self.b)['logbayesfactor']))
                print('Evidence: M_C = {:0.3f}, M_D = {:0.3f}'.format(model.summary(b=self.b)['evidence']['mc'], 
                                                                      model.summary(b=self.b)['evidence']['md']))
                print('Posterior model probabilities: p(M_C|D) = {:0.3f}, p(M_D|D) = {:0.3f}'.format(model.summary(b=self.b)['pmp']['pmc'], 
                                                                      model.summary(b=self.b)['pmp']['pmd']))
                print('')                
            self.results[kernel_name] = model                
        self.trained = True  
        return self.results        
    #    
    def get_total_log_Bayes_factor(self, verbose=False):
        """Computes the Bayes factor of the model comparison, across all 
        provided kernels.
        
        The total Bayes factor is computed as:
            BF_total = (\sum_k p(D|k) p(k|M_D)) / (\sum_k p(D|k) p(k|M_C))
            
        In practice, we have no a priori knowledge so p(k|M_i) == p(k|M_j) for 
        all i and j.
        
        :return: The total log Bayes factor.
        """
        
        if not self.trained:
            raise NotTrainedException('You must train the model first')          
        evidences_c = np.asarray(
                [self.results[k].summary(b=self.b)['evidence']['mc'] 
                for k in self.kernel_dict.keys()])
        evidences_d = np.asarray(
                [self.results[k].summary(b=self.b)['evidence']['md'] 
                for k in self.kernel_dict.keys()])
        # prior is uniform so this doesn't do anything - but for later generalizations
        p_k_M = 1.0 / self.K
        BF = logSumExp(evidences_d - np.log(p_k_M)) \
                - logSumExp(evidences_c - np.log(p_k_M))
        if verbose:
            print('log p(D|M_C) = {:0.2f}, log p(D|M_D) = {:0.2f}'.format(logSumExp(evidences_c - np.log(p_k_M)), 
                         logSumExp(evidences_d - np.log(p_k_M))))
        return BF
    #
    def get_total_disc_effect_size(self, nmontecarlo=20000):
        """Returns the model averaged effect size across all the discontinuous 
        models.
        
        The different kernels consistute a mixture distribution over the effect 
        size. While we can get some analytical results if we assume 
        Gaussianity, the robust approach is just to Monte Carlo sample from the
        mixture distrbution.
        
        The computation is performed only once; subsequent calls of the 
        function return the stored result.
        
        :return: A tuple containing the effect size grid and corresponding pdf.        
        """
        
        if self.total_disc_es is None:
            disc_log_evidences = [self.results[kernel].summary(b=self.b)['evidence']['md'] 
                                  for kernel in self.kernel_dict.keys()]
            M = len(disc_log_evidences)
            Z = logSumExp(disc_log_evidences)
            disc_evidences = np.exp(disc_log_evidences - Z)
            disc_stats = [self.results[kernel].summary(b=self.b)['es_disc_stats'] 
                          for kernel in self.kernel_dict.keys()]
            samples = list()            
            for i in range(M):
                samples += list(np.random.normal(loc=disc_stats[i][0], 
                                                 scale=disc_stats[i][1], 
                                                 size=int(nmontecarlo*disc_evidences[i])))
                
            kde_fit = stats.gaussian_kde(samples, bw_method='silverman')
            xrange = np.linspace(np.min(samples), np.max(samples), 500)
            es_bma = kde_fit(xrange)
            self.total_disc_es = np.sum(xrange*es_bma) * (xrange[1]-xrange[0])
            self.total_disc_pdf = (xrange, es_bma)
        return self.total_disc_es
    #
    def get_total_BMA_effect_size(self):
        """Returns the model averaged effect size across all models (continuous
        and discontinuous).
        
        The different kernels consistute a mixture distribution over the effect 
        size, of which the elements corresponding to dichotomous models are 
        Gaussian and the elements corresponding to continuous models are spike
        densities. The total BMA posterior of the effect size is computed using
        Monte Carlo.
        
        The computation is performed only once; subsequent calls of the 
        function return the stored result.
        
        :return: A tuple containing the effect size grid and corresponding pdf. 
        """
        
        if self.total_bma_es is None:
            # clean up these long expressions on Isle 2
            log_evidences = [self.results[kernel].summary(b=self.b)['evidence']['md'] 
                             for kernel in self.kernel_dict.keys()] + \
                            [self.results[kernel].summary(b=self.b)['evidence']['mc'] 
                            for kernel in self.kernel_dict.keys()]
            
            M = len(log_evidences)
            Z = logSumExp(log_evidences)
            evidences = np.exp(log_evidences - Z)
            disc_stats = [self.results[kernel].summary(b=self.b)['es_disc_stats'] 
                          for kernel in self.kernel_dict.keys()]
            nsamples = 50000
            samples = list()            
            for i in range(int(M/2)):
                samples += list(np.random.normal(loc=disc_stats[i][0], 
                                                 scale=disc_stats[i][1], 
                                                 size=int(nsamples*evidences[i])))
            samples += list(np.zeros(nsamples - len(samples)))
            
            if np.sum(np.abs(samples))==0:
                xrange = np.linspace(-2, 2, 500)
                ix = np.argmin((xrange-self.b)**2)
                es_bma = np.zeros((500))
                es_bma[ix] = 1.0/ (xrange[1] - xrange[0])
            else:            
                kde_fit = stats.gaussian_kde(samples, bw_method='silverman')
                xrange = np.linspace(np.min(samples), np.max(samples), 500)
                es_bma = kde_fit(xrange)
            self.total_bma_es = np.sum(xrange*es_bma) * (xrange[1]-xrange[0])
            self.total_bma_pdf = (xrange, es_bma)
        return self.total_bma_es
    #
    def get_rdd_p_values(self):
        """Computes the p value of the discontinuity.
        
        This approach is used by Branson et al. (2019). It combines GP 
        regression to estimate the discontinuous model only, then computes a p 
        value based on the difference between the two Gaussians.
        """
        
        if self.rdd_p_values is None:
            self.rdd_p_values = dict()
            for kernel in self.kernel_dict.keys():
                es_mean, es_var = self.results[kernel].summary(b=self.b)['es_disc_stats']
                es_dist = stats.norm(loc=np.abs(es_mean), scale=np.sqrt(es_var))
                pval = es_dist.cdf(0.0)
                self.rdd_p_values[kernel] = pval
        return self.rdd_p_values
        
        
    def get_model_results(self):
        """Simple getter function for the results for each kernel.
        """
        
        if not self.trained:
            raise NotTrainedException('You must train the model first')
        return self.results
    #
    def pretty_print(self, verbose=True):
        """Print and return summary report.
        """
        M = len(self.kernel_dict)
        if self.ndim == 1:
            df = pd.DataFrame(index=range(M+1), 
                              columns=['Kernel', 
                                       'Log BF', 
                                       'Effect size', 
                                       'BMA effect size'])
            for i, kernel_name in enumerate(self.kernel_dict.keys()):            
                summary     = self.results[kernel_name].summary(b=self.b)
                logBF       = summary['logbayesfactor']
                es_range    = summary['es_range']
                delta       = es_range[1] - es_range[0]
                es_BMA      = np.sum(es_range*summary['es_BMA']) * delta
                es_disc     = np.asscalar(summary['f(b)'][0] - summary['f(b)'][1])
                df.loc[i]   = [kernel_name, logBF, es_disc, es_BMA]
                
            total_log_bf = self.get_total_log_Bayes_factor()
            df.loc[M] = ['Bayesian model average', 
                         total_log_bf, 
                         self.get_total_disc_effect_size(), 
                         self.get_total_BMA_effect_size()]
        else:
            df = pd.DataFrame(index=range(M+1), 
                          columns=['Kernel', 
                                   'Log BF'])
            for i, kernel_name in enumerate(self.kernel_dict.keys()):            
                summary     = self.results[kernel_name].summary(b=self.b)
                logBF       = summary['logbayesfactor']
                df.loc[i]   = [kernel_name, logBF]
                
            total_log_bf = self.get_total_log_Bayes_factor()
            df.loc[M] = ['Bayesian model average', 
                         total_log_bf]
            
        if verbose:
            print(df)
        return df
        
    def plot_model_fits(self, x_test, plot_opts=dict()):
        """Basic plotting functionality.
        
        Details of the plots, such as ticklabels, can be added to the figures
        later via the fig and axes objects. Left column shows all continuous
        models, the right column shows all discontinuous models.
        
        :param x_test: The range over which the models are interpolated for
                       visualization.
        :param plot_opts: Visualization options.
        :return: The figure and axes handles.
        """
                
        cmodel_color        = plot_opts.get('cmodel_color',         'black')
        dmodel_pre_color    = plot_opts.get('dmodel_pre_color',     '#cc7d21')
        dmodel_post_color   = plot_opts.get('dmodel_post_color',    '#0e2b4d')
        color_data          = plot_opts.get('color_data',           '#334431')
        marker_pre          = plot_opts.get('marker_pre',           'x')
        marker_post         = plot_opts.get('marker_post',          'o')
        marker_size         = plot_opts.get('marker_size',          5)
        marker_alpha        = plot_opts.get('marker_alpha',         1.0)
        plot_effect_size    = plot_opts.get('plot_effect_size',     True)   
        plot_title          = plot_opts.get('plot_title',           'Model fits')  
        plot_samewindow     = plot_opts.get('plot_same_window',     False)
        axes                = plot_opts.get('axes',                 None)
        plot_full_range     = plot_opts.get('plot_full_range',      
                                            self.labelFunc is None) 
        plot_xlim           = plot_opts.get('plot_xlim',           
                                            [np.min(self.x), np.max(self.x)])
        plot_ylim           = plot_opts.get('plot_ylim',           
                                            [np.min(self.y), np.max(self.y)])
        
        if not plot_samewindow:
            if axes is None:
                fig, axes = plt.subplots(nrows=self.K, ncols=2, sharex=True, 
                                         sharey=True, figsize=(12, 6*self.K))
            else:
                fig = plt.gcf()
            
            for i, kernel_name in enumerate(self.kernel_dict.keys()):
                self.results[kernel_name].CModel.plot(x_test, axes[i, 0], 
                             plotOptions={'color': cmodel_color})
                self.results[kernel_name].DModel.plot(x_test, axes[i, 1], 
                            b=self.b, 
                            plotOptions=({'color': dmodel_pre_color}, 
                                         {'color': dmodel_post_color}), 
                                         plotEffectSize=plot_effect_size,
                                         plotFullRange=plot_full_range)
                axes[i, 0].set_ylabel(kernel_name)
                summary = self.results[kernel_name].summary(b=self.b)  
                pmc, pmd = summary['pmp']['pmc'], summary['pmp']['pmd']
                axes[i, 0].set_title('p(M_C | x, y) = {:0.3f}'.format(pmc))
                axes[i, 1].set_title('p(M_D | x, y) = {:0.3f}'.format(pmd))
        else:
            if axes is None:
                fig, axes = plt.subplots(nrows=self.K, ncols=1, sharex=True, 
                                     sharey=True, figsize=(6, 6*self.K))
            else:
                fig = plt.gcf()
            
            for i, kernel_name in enumerate(self.kernel_dict.keys()):
                self.results[kernel_name].CModel.plot(x_test, axes[i], 
                             plotOptions={'color': cmodel_color})
                self.results[kernel_name].DModel.plot(x_test, axes[i], 
                            b=self.b, 
                            plotOptions=({'color': dmodel_pre_color}, 
                                         {'color': dmodel_post_color}), 
                                         plotEffectSize=plot_effect_size,
                                         plotFullRange=plot_full_range)
                axes[i].set_ylabel(kernel_name)
                summary = self.results[kernel_name].summary(b=self.b)  
                pmc, pmd = summary['pmp']['pmc'], summary['pmp']['pmd']
                axes[i].set_title('p(M_C | x, y) = {:0.3f}, p(M_D | x, y) = {:0.3f}'.format(pmc, pmd))
                         
        for ax in axes.flatten():
            ax.axvline(x=self.b, color='black', linestyle='--')
            if self.labelFunc is None:
                lab1 = self.labelLUT==0
            else:
                lab1 = self.labelFunc(self.x)
            lab2 = np.logical_not(lab1)
            ax.plot(self.x[lab1], self.y[lab1], linestyle='None', 
                    marker=marker_pre, color=color_data, alpha=marker_alpha, 
                    ms=marker_size)
            ax.plot(self.x[lab2], self.y[lab2], linestyle='None', 
                    marker=marker_post, color=color_data, alpha=marker_alpha, 
                    ms=marker_size)
            ax.set_xlim(plot_xlim)
            ax.set_ylim(plot_ylim)
        plt.suptitle(plot_title)
        return fig, axes
    #
    def plot_effect_sizes(self, plot_opts=dict()):
        """Plot the effect sizes.
        
        Details of the plots, such as ticklabels, can be added to the figures
        later via the fig and axes objects.
        
        :param plot_opts: Visualization options.
        :return: The figure and axes handles.
        """
        
        if self.ndim == 1:
            fig, axes = plt.subplots(nrows=self.K, ncols=1, 
                                     sharex=True, 
                                     figsize=(6, 6*self.K))
            
            dmodel_color = plot_opts.get('dmodel_color', '#cc7d21')
            for i, kernel_name in enumerate(self.kernel_dict.keys()):
                summary = self.results[kernel_name].summary(b=self.b)
#                xmin, xmax  = summary['es_interval']                
                pdf         = summary['es_Disc']
                d_bma       = summary['es_BMA']                
                xrange      = summary['es_range']  
                n_interp    = len(pdf)
                
                axes[i].plot(xrange, pdf, c=dmodel_color, label=r'$M_D$', 
                    linewidth=1.0, linestyle='--')
                axes[i].fill_between(xrange, pdf, np.zeros((n_interp)), 
                    alpha=0.3, color=dmodel_color)
                axes[i].axvline(x=0, linewidth=1.0, label=r'$M_C$', color='k', 
                    linestyle='--')    
                axes[i].plot(xrange, d_bma, c='k', label=r'BMA', linewidth=1.0)
                axes[i].fill_between(xrange, d_bma, np.zeros((n_interp)), 
                    alpha=0.3, color='k')
                axes[i].set_xlim([np.min(xrange), np.max(xrange)])
                axes[i].set_ylabel('Probability density')
                axes[i].set_xlim([xrange[0], xrange[-1]])
                axes[i].set_title('{:s} kernel'.format(kernel_name))
            
            axes[-1].legend(loc='best')
            
            return fig, axes
        else:
            raise NotImplementedError('Effect size plot for D>1 is not implemented')
            # TODO: Part of this code is available in the Dutch elections example.
    #    
    def plot_model_average_effect_sizes(self, plot_opts=dict()):
        """Plot effect sizes model averaged over kernels.
        
        Visualizes the Monte Carlo densities of the effect sizes that have been
        computed in self.get_total_BMA_effect_size() and 
        self.get_total_disc_effect_size().
        
        :param plot_opts: Visualization options.
        :return: The figure handle.
        """
        
        if self.total_bma_es is None:
            self.get_total_BMA_effect_size()
        if self.total_disc_es is None:
            self.get_total_disc_effect_size()
                
        if self.ndim == 1:
            fig = plt.figure()
            
            dmodel_color = plot_opts.get('dmodel_color', '#cc7d21')
            disc_x, disc_pdf = self.total_disc_pdf
            bma_x, bma_pdf = self.total_bma_pdf           
                             
            plt.plot(disc_x, disc_pdf, color=dmodel_color, label=r'M_D', 
                     linewidth=1.0)  
            plt.plot(bma_x, bma_pdf, color='k', label=r'BMA', linewidth=1.0)  
            plt.title('Effect size across kernels')
            plt.ylabel('Probability density')            
            plt.legend(loc='best')
            
            return fig
        else:
            raise NotImplementedError('Effect size plot for D>1 is not implemented')
        
        
    def plot_posterior_model_probabilities(self, plot_opts=dict()):
        """Plot the posterior model probabilities as a pie chart.
        
        Creates a pie chart of the posterior model probabilities per kernel.
        Only provided for completeness; is probably more suited to be embedded
        into one of the other figures, as the information density of a pie 
        chart is quite low.
        
        :param plot_opts: Visualization options.        
        """
        
        fig, axes = plt.subplots(nrows=self.K, ncols=1, 
                                 sharex=True, sharey=True, 
                                 figsize=(6, 6*self.K))
            
        cmodel_color = plot_opts.get('cmodel_color', 'black')
        dmodel_color = plot_opts.get('dmodel_color', '#cc7d21')
        for i, kernel_name in enumerate(self.kernel_dict.keys()):
            summary = self.results[kernel_name].summary(b=self.b)  
            pmc, pmd = summary['pmp']['pmc'], summary['pmp']['pmd']
            
            axes[i].pie([pmc, pmd], 
                        colors=[cmodel_color, dmodel_color], 
                        labels=[np.round(pmc, 2), np.round(pmd, 2)], 
                        explode=[0.05, 0.05])   
            axes[i].set_title('{:s} kernel'.format(kernel_name))
        axes[-1].legend(loc='best')
        for ax in axes:
            ax.set_aspect(1.0)
        
        return fig, axes
        
        
# Helpers
        
#def get_kernel(kerneltype, D):
#
#    if kerneltype == 'Matern32':
#        kernel = GPy.kern.Matern32(D) + GPy.kern.White(D)
#    elif kerneltype == 'Linear':
#        kernel = GPy.kern.Linear(D) + GPy.kern.Bias(D) + GPy.kern.White(D)
#    elif kerneltype in ('ttest', 'Constant'):
#        kernel = GPy.kern.Bias(D) + GPy.kern.White(D)
#    elif kerneltype == 'RBF':
#        kernel = GPy.kern.RBF(D) + GPy.kern.White(D)
#    elif kerneltype == 'Periodic':
#        kernel = GPy.kern.PeriodicMatern32(D) 
#        + GPy.kern.Linear(D) 
#        + GPy.kern.White(D) #+ GPy.kern.Bias(D)?
#    else:
#        raise('Unsupported kernel type')
#    return kernel.copy()
#    
def logSumExp(ns):
    """Log-sum-exp trick to get normalization of evidences.
    """
    mx = np.max(ns)
    ds = ns - mx
    sumOfExp = np.exp(ds).sum()
    return mx + np.log(sumOfExp)       
    
class NotTrainedException(Exception):
    """Raised when a function is called that required training first.
    """
    pass