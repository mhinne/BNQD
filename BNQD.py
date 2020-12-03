import gpflow
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from gpflow.models import GPModel, GPR
from gpflow.mean_functions import MeanFunction, Zero, Constant
from gpflow.likelihoods import Gaussian
from gpflow.logdensities import multivariate_normal
from gpflow.utilities import deepcopy, print_summary, positive, to_default_float
from gpflow.ci_utils import ci_niter
from gpflow.kernels import Matern32, Matern52, SquaredExponential, Linear, Bias, Exponential, Polynomial, Cosine, Periodic, RationalQuadratic

from gpflow.conditionals.util import sample_mvn
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import scipy.stats as st
from gpflow import Parameter

gpflow.config.set_default_float(np.float64)
gpflow.config.set_default_jitter(1e-4)
# gpflow.config.set_default_summary_fmt("notebook")
# convert to float64 for tfp to play nicely with gpflow in 64
f64 = gpflow.utilities.to_default_float

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

version = '2.0.0'

#plt.rc('font', size=SMALL_SIZE)    # controls default text sizes
plt.rc('axes', titlesize=30)        # fontsize of the axes title
plt.rc('axes', labelsize=24)        # fontsize of the x and y labels
plt.rc('xtick', labelsize=18)       # fontsize of the tick labels
plt.rc('ytick', labelsize=18)       # fontsize of the tick labels
plt.rc('legend', fontsize=20)       # legend fontsize
plt.rc('figure', titlesize=36)      # fontsize of the figure title

import matplotlib.cm as cm
color_cycle = cm.magma(np.linspace(0, 1, 5))


def log_sum_exp(ns):
  """Log-sum-exp trick to get normalization of evidences.
  """
  mx = np.max(ns)
  ds = ns - mx
  sumOfExp = np.exp(ds).sum()
  return mx + np.log(sumOfExp) 

class ContinuousModel(GPR):
  """
  The continuous model is simply a Gaussian process regression. We add this 
  wrapper for consistency and convenience.
  """

  def __init__(self, data, kernel, mean_function=None):
    super().__init__(data, kernel, mean_function)
    self.title = 'Continuous model'
  #
  def objective(self):
    return -1.0*self.log_marginal_likelihood()
  #
  def log_hyper_marginal_likelihood(self, mode='BIC'):
    if mode is None:
      return self.log_marginal_likelihood().numpy()
    elif mode is 'BIC':
      L = self.log_marginal_likelihood().numpy()
      n = self.data[1].shape[0]
      k = len(self.trainable_parameters) + len(self.likelihood.trainable_parameters)
      BIC = L - k/2.0 * np.log(n)
      return BIC
    elif mode is 'Laplace':
      raise NotImplementedError('Laplace approximation is not implemented.')
    elif mode is 'HMC':
      raise NotImplementedError('HMC approximation is not implemented.')
    else:
      raise NotImplementedError('{:s} is not implemented.'.format(mode))
  #
#
class DiscontinuousModel(GPModel):
  """
  The discontinuous model is a Gaussian process model with a Gaussian 
  likelihood, just as the continuous model. However, the marginal likelihood is 
  computed as the product of the marginal likelihoods for the pre- and post 
  intervention data points. Consequently, points in either condition do not 
  influence the alternative condition.
  """

  def __init__(self, data, kernel, likelihood=None, mean_function=None, 
               num_latent_gps=None, noise_variance=1.0):
    if likelihood is None:
      likelihood = Gaussian(noise_variance)
    _, Y_data = data
    super().__init__(kernel, likelihood, mean_function, 
                     num_latent_gps=data[0][1].shape[-1])
    self.data = data
    self.title = 'Discontinuous model'

  #
  def log_marginal_likelihood(self):
    """
    Computes p(D|m1) = p(D_C|m1)p(D_I|m1), where D_C are the data points 
    corresponding to absence of treatment, while D_I are the data points in the
    treated group.
    """
    
    log_prob = 0
    for dataset in self.data:
        x, y = dataset
        K = self.kernel(x)            
        num_data = x.shape[0]
        k_diag = tf.linalg.diag_part(K)
        s_diag = tf.fill([num_data], self.likelihood.variance)
        ks = tf.linalg.set_diag(K, k_diag + s_diag)
        L = tf.linalg.cholesky(ks)
        m = self.mean_function(x)

        # [R,] log-likelihoods for each independent dimension of Y
        log_prob += multivariate_normal(y, m, L)
    return tf.reduce_sum(log_prob)
  #
  def log_hyper_marginal_likelihood(self, mode='BIC'):
    if mode is None:
      return self.log_marginal_likelihood().numpy()
    elif mode is 'BIC':
      L = self.log_marginal_likelihood().numpy()
      n = self.data[0][1].shape[0] + self.data[1][1].shape[0]
      k = len(self.trainable_parameters) + len(self.likelihood.trainable_parameters)
      BIC = L - k/2.0 * np.log(n)
      return BIC
    elif mode is 'Laplace':
      raise NotImplementedError('Laplace approximation is not implemented.')
    elif mode is 'HMC':
      raise NotImplementedError('HMC approximation is not implemented.')
    else:
      raise NotImplementedError('{:s} is not implemented.'.format(mode))
  #
  def objective(self):
    return -1.0*self.log_marginal_likelihood()
  #
  def predict_f(self, Xnew_list, full_cov=False, full_output_cov=False):
    """
    This is standard GPR prediction, except that effectively the covariance
    between points before and after treatment is set to zero.
    """
    res = list()
        
    for i, Xnew in enumerate(Xnew_list):
    
      x_data, y_data = self.data[i]
      err = y_data - self.mean_function(x_data)

      kmm = self.kernel(x_data)
      knn = self.kernel(Xnew, full_cov=full_cov)
      kmn = self.kernel(x_data, Xnew)

      num_data = x_data.shape[0]
      s = tf.linalg.diag(tf.fill([num_data], self.likelihood.variance))

      conditional = gpflow.conditionals.base_conditional
      f_mean_zero, f_var = conditional(kmn, kmm + s, knn, err, full_cov=full_cov,
                                        white=False)  # [N, P], [N, P] or [P, N, N]
      f_mean = f_mean_zero + self.mean_function(Xnew) 
      res.append((f_mean, f_var))
    return res
  #  
  def predict_f_samples(self, Xnew_list, num_samples=None, full_cov=True, 
                        full_output_cov=False):

    if full_cov and full_output_cov:
      raise NotImplementedError(
        "The combination of both `full_cov` and `full_output_cov` is not supported."
      )
    res = list()

    pred_f_results = self.predict_f(Xnew_list, full_cov=full_cov, 
                                   full_output_cov=full_output_cov)

    for i, Xnew in enumerate(Xnew_list):        
      if full_cov:
        mean_for_sample = tf.linalg.adjoint(pred_f_results[i][0])  
        samples = sample_mvn(
            mean_for_sample, pred_f_results[i][1], 'full', 
            num_samples=num_samples
        )  
        samples = tf.linalg.adjoint(samples)  
      else:
        samples = sample_mvn(
            pred_f_results[i][0], pred_f_results[i][1], 'diag', 
            num_samples=num_samples
        )  
      res.append(samples)
    return res

  def predict_y(self, Xnew_list, full_cov=False, full_output_cov=False):
    preds = self.predict_f(Xnew_list, full_cov=full_cov, 
                           full_output_cov=full_output_cov)

    res = list()
    for pred in preds:
      f_mean, f_var = pred
      res.append(self.likelihood.predict_mean_and_var(f_mean, f_var))
    return res
  #
  def maximum_log_likelihood_objective(self):
      return self.log_marginal_likelihood()
  #
  def effect_size(self, threshold=0.0):
    """
    Let \tau be the effect size, then \tau = f_I(b) - f_C(b). See Branson et al. 
    (2019), equation (17) for the posterior of \tau. TODO, also incorporate GP output variance!
    """ 
    if not isinstance(self.likelihood, Gaussian):
      raise NotImplementedError('Effect size estimates for non-Gaussian likelihoods is not implemented.')
    pred_c, pred_i = self.predict_y(2*[np.atleast_2d(threshold)])
    pred_c_mean, pred_c_var = pred_c
    pred_i_mean, pred_i_var = pred_i

    if np.isscalar(threshold):
      return (pred_i_mean - pred_c_mean).numpy().item(), (pred_i_var + pred_c_var).numpy().item()
    else:
      return (pred_i_mean - pred_c_mean).numpy(), (pred_i_var + pred_c_var).numpy()
  #
#

class BNQD():

  def __init__(self, data, kernels, ip):
    """
    For every type of kernel we construct a continuous and a discontinuous 
    model. Prior distributions for kernel hyperparameters are set via a look-up
    table. 
    """
    x, y = data
    mean_function = Constant(np.mean(y, axis=0))
    gpflow.set_trainable(mean_function.c, False)

    # This is not foolproof, what if ndim(x) = (n, 1)?
    self.ndim = np.ndim(x)

    if np.ndim(x) == 1:
      x = x[:, None]
    
    if np.ndim(y) == 1:
      y = y[:, None]

    self.x = x
    self.y = y

    if np.isscalar(ip):
      self.f_i = lambda w: w[:, 0] >= ip
      self.f_c = lambda w: w[:, 0] <= ip
    else:
      self.f_i = ip
      self.f_c = lambda w: np.logical_not(self.f_i(w))

    x1 = x[self.f_c(self.x), :]
    y1 = y[self.f_c(self.x), :] 
    x2 = x[self.f_i(self.x), :]
    y2 = y[self.f_i(self.x), :]

    self.ip = ip

    if not isinstance(kernels, list): 
      kernels = [kernels]

    self.kernels = kernels
    self.m0 = list()
    self.m1 = list()

    self.hmc_helpers = None
    self.hmc_samples = None

    for kernel in kernels:

      K0 = deepcopy(kernel)
      K1 = deepcopy(kernel)

      m0 = ContinuousModel(data=(x, y), kernel=K0, mean_function=mean_function)
      m1 = DiscontinuousModel(data=((x1, y1), (x2, y2)), kernel=K1, 
                              mean_function=mean_function)
      
      if hasattr(kernel, 'lengthscales'):
        for m in [m0, m1]:
          m.kernel.lengthscales.prior = tfd.Gamma(np.float64(0.01), 
                                                  np.float64(0.01))

      if hasattr(kernel, 'offset'):
        for m in [m0, m1]:
          m.kernel.offset.prior = tfd.Normal(np.float64(0.0), np.float64(1.0))

      # place the same priors on the hyperparameters of the two models
      if hasattr(kernel, 'variance'):
        for m in [m0, m1]:
          m.kernel.variance.prior = tfd.Gamma(np.float64(0.01), 
                                              np.float64(0.01))
      
      # likelihood variance
      if hasattr(m.likelihood, 'variance'):
        for m in [m0, m1]:
          m.likelihood.variance.prior = tfd.Gamma(np.float64(0.01), 
                                                  np.float64(0.01))

      self.m0.append(m0)
      self.m1.append(m1)
  #
  def get_hyperparameters(self):
    """
    Returns list of two dictionaries per kernel each containing the (optimized)
    hyperparameters.
    """
    hypers = list()
    for k, kernel in enumerate(self.kernels):
      m0_param_dict = {name: param.numpy() for name, param in gpflow.utilities.select_dict_parameters_with_prior(self.m0[k]).items()}  
      m1_param_dict = {name: param.numpy() for name, param in gpflow.utilities.select_dict_parameters_with_prior(self.m1[k]).items()} 
      hypers.append({'m0': m0_param_dict, 'm1': m1_param_dict})
    return hypers
  #
  def plot_hmc_params(self):
    for i, kernel in enumerate(self.kernels):
      fig, axes = plt.subplots(2, len(self.m0[0].trainable_parameters), 
                               figsize=(15, 5), sharex='col', 
                               constrained_layout=True)
      for j, m in enumerate([self.m0[i], self.m1[i]]):
        samples = self.hmc_constrained_samples[i][j]
        param_to_name = {param: name for name, param in gpflow.utilities.select_dict_parameters_with_prior(m).items()}            
        # print(param_to_name)
        for ax, val, param in zip(axes[j,:], samples, m.trainable_parameters):
          ax.hist(np.stack(val).flatten(), bins=30, density=True, 
                  color=color_cycle[j,:], alpha=0.6)
          ax.axvline(x=param.numpy(), ls=':', c='k', lw=2.0)
          ax.set_title(param_to_name[param], fontsize=14)
        axes[j,0].set_ylabel('Continuous' if j==0 else 'Discontinuous', 
                             fontsize=14)
      fig.suptitle('HMC results, {:s} kernel'.format(kernel.name.replace('_', '\_')), 
                   fontsize=20)
      plt.show()
  #
  def train(self, opt=None, verbose=False, posterior=False, train_opts=None):
    """
    If opt is none, we default to the Scipy optimizer.

    If verbose==True, we print the results of the optimizer (mostly for debugging).

    If posterior==True, we proceed with HMC of the hyperparameters after 
    initializing with the optimiser values. Not yet implemented.
    """
    if opt is None:
      opt = gpflow.optimizers.Scipy()

    if train_opts is None:
      train_opts = dict()

    for i in range(len(self.kernels)):

      for m in [self.m0[i], self.m1[i]]:
        opt_log = opt.minimize(m.objective, m.trainable_variables, 
                               options=dict(maxiter=train_opts.get('max_iter', 
                                                                   1000)))
        if verbose:
          print(opt_log)
          print_summary(m)

    if posterior:  
      # See https://gpflow.readthedocs.io/en/master/notebooks/advanced/mcmc.html    
      if verbose:
        print('Approximating p(\theta | X, Y) using HMC')

      num_burnin_steps = ci_niter(train_opts.get('hmc_burnin', 5000))
      self.num_samples = ci_niter(train_opts.get('hmc_samples', 5000))

      self.hmc_helpers = list()
      self.hmc_samples = list()
      self.hmc_constrained_samples = list()

      for i in range(len(self.kernels)):
        hmc_helpers_kernel = list()
        hmc_samples_kernel = list()
        hmc_samples_constrained_kernel = list()

        for m in [self.m0[i], self.m1[i]]:
          # Note that here we need model.trainable_parameters, not trainable_variables - only parameters can have priors!
          hmc_helper = gpflow.optimizers.SamplingHelper(
              m.log_posterior_density, m.trainable_parameters
          )

          hmc = tfp.mcmc.HamiltonianMonteCarlo(
              target_log_prob_fn=hmc_helper.target_log_prob_fn, 
              num_leapfrog_steps=train_opts.get('hmc_num_leapfrog', 10), 
              step_size=train_opts.get('hmc_stepsize', 0.01)
          )
          adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
              hmc, num_adaptation_steps=train_opts.get('hmc_num_adapt', 10), 
              target_accept_prob=f64(train_opts.get('hmc_target_accept', 0.75)), 
              adaptation_rate=train_opts.get('hmc_adapt_rate', 0.1)
          )

          @tf.function
          def run_chain_fn():
              return tfp.mcmc.sample_chain(
                  num_results=self.num_samples,
                  num_burnin_steps=num_burnin_steps,
                  current_state=hmc_helper.current_state,
                  kernel=adaptive_hmc,
                  trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
              )

          samples, traces = run_chain_fn()
          parameter_samples = hmc_helper.convert_to_constrained_values(samples)

          hmc_helpers_kernel.append(hmc_helper)
          hmc_samples_kernel.append(samples)
          hmc_samples_constrained_kernel.append(parameter_samples)
        self.hmc_helpers.append(hmc_helpers_kernel)
        self.hmc_samples.append(hmc_samples_kernel)          
        self.hmc_constrained_samples.append(hmc_samples_constrained_kernel)
  #
  def plot(self, xpred, plot_samples=False, fig=None, plot_opts=dict()):
    """
    Plot the estimated latent functions and one standard deviation around it, 
    for both models. Data are superimposed.
    """

    if plot_samples and self.hmc_samples is None:
      print('Cannot plot samples; train with posterior=True.')
      plot_samples = False

    width = plot_opts.get('width', 15)
    height = plot_opts.get('height', 6)
    markersize = plot_opts.get('markersize', 14)

    if self.ndim == 1:
      if fig is None:
        fig, axes = plt.subplots(nrows=len(self.kernels), ncols=1, 
                                figsize=(width, height*len(self.kernels)), 
                                 sharex=True, sharey=True)
      if np.ndim(xpred) == 1:
        xpred = xpred[:, None]      

      xpred1 = xpred[self.f_c(xpred)] # control
      xpred2 = xpred[self.f_i(xpred)] # intervention

      log_bfs = self.get_bayes_factor()
      es_bma = self.bma_effect_size()
      es_m1 = self.discontinuous_effect_size_mean_var()

      for i, kernel in enumerate(self.kernels):
        ax = axes[i]

        mu0, var0 = self.m0[i].predict_y(xpred)
        preds1 = self.m1[i].predict_y(Xnew_list=[xpred1, xpred2])

        ax.plot(np.squeeze(xpred), mu0, color=color_cycle[0,:], lw=2.0, 
                label='$M_0$', ls='--')
        if self.hmc_samples is None or not plot_samples:
          ax.fill_between(np.squeeze(xpred), np.squeeze(mu0 + 0.5*np.sqrt(var0)), 
                          np.squeeze(mu0 - 0.5*np.sqrt(var0)), alpha=0.3, 
                          color=color_cycle[0,:])
          
        else:
          for j in np.random.choice(np.arange(self.num_samples), size=20, 
                                    replace=False):
            for var, var_samples in zip(self.hmc_helpers[i][0].current_state, 
                                        self.hmc_samples[i][0]):
                var.assign(var_samples[j])
            f = self.m0[i].predict_f_samples(xpred, 1)
            ax.plot(np.squeeze(xpred), f[0, :, :], color=color_cycle[0,:], lw=1, 
                    alpha=0.3)

        add_label = True

        if self.hmc_samples is not None and plot_samples:
          hmc_plot_samples = list()
          for j in np.random.choice(np.arange(self.num_samples), size=20, 
                                    replace=False):
            for var, var_samples in zip(self.hmc_helpers[i][1].current_state, 
                                        self.hmc_samples[i][1]):
                var.assign(var_samples[j])
            f = self.m1[i].predict_f_samples(Xnew_list=[xpred1, xpred2], 
                                             num_samples=1)
            ax.plot(np.squeeze(xpred1), f[0][0,:,:], color=color_cycle[1,:], 
                    lw=1, alpha=0.3)
            ax.plot(np.squeeze(xpred2), f[1][0,:,:], color=color_cycle[1,:], 
                    lw=1, alpha=0.3)

        for xp, pred1 in zip([xpred1, xpred2], preds1):
          mu1, var1 = pred1
          if add_label:
            ax.plot(np.squeeze(xp), mu1, color=color_cycle[1,:], lw=2.0, 
                    label='$M_1$')
            add_label = False
          else:
            ax.plot(np.squeeze(xp), mu1, color=color_cycle[1,:], lw=2.0)
          if self.hmc_samples is None or not plot_samples:
            ax.fill_between(np.squeeze(xp), np.squeeze(mu1 + 0.5*np.sqrt(var1)), 
                            np.squeeze(mu1 - 0.5*np.sqrt(var1)), alpha=0.3, 
                            color=color_cycle[1,:])
          
        ax.set_xlim([xpred[0,0], xpred[-1,0]])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.axvline(x=self.ip, ls=':', lw=1.0, c='k', 
                   label='Intervention threshold')
        ax.plot(np.squeeze(self.x), np.squeeze(self.y), ls='none', marker='o', 
                c='k', fillstyle='full', mfc=color_cycle[2,:], markersize=markersize, 
                label='Data')

        pts_preds = self.m1[i].predict_f(2*[np.atleast_2d(self.ip)])
        for pt in pts_preds:
          pt_mean, _ = pt
          ax.plot(self.ip, np.squeeze(pt_mean), marker='o', ls='none', 
                  fillstyle='full', mfc='w', color='k', markersize=12)
          
        ax.set_title('{:s} kernel, log BF = {:0.3f}, $E[p(d|M_1)] = {:0.3f}$, $E[p(d)] = {:0.3f}$'.format(kernel.name.replace('_', '\_').capitalize(), 
                                                                                                          log_bfs[i], es_m1[i][0], es_bma[i]))

      handles, labels = ax.get_legend_handles_labels()
      plt.subplots_adjust(hspace=0.2, left=0, right=1.0, top=0.9, bottom=0.05)
      plt.figlegend(handles, labels, frameon=False, ncol=4, fontsize=18,
                    loc='lower center')
      return fig
    elif self.ndim==2:
              
      K = len(self.kernels)
      fig = plt.figure(figsize=(width, 6*height))
            
      f_c_label = self.f_c(xpred)
      f_i_label = self.f_i(xpred)      
      
      nt = int(np.sqrt(xpred.shape[0]))

      f_c_label_res = np.reshape(f_c_label, (nt, nt))
      f_i_label_res = np.reshape(f_i_label, (nt, nt))

      X = np.reshape(xpred[:,0], (nt, nt))
      Y = np.reshape(xpred[:,1], (nt, nt))

      for i, kernel in enumerate(self.kernels):
        ax1 = fig.add_subplot(K, 2, 2*i+1, projection='3d')
        mu0, _ = self.m0[i].predict_y(xpred)        
        Z0 = np.reshape(mu0, (nt, nt))

        ax1.plot_wireframe(X, Y, Z0, color=color_cycle[0,:], 
                        antialiased=True, alpha=0.5)

        ax2 = fig.add_subplot(K, 2, 2*(i+1), projection='3d')
        preds1 = self.m1[i].predict_y(Xnew_list=[xpred, xpred])

        for pred1, label in zip(preds1, [f_c_label_res, f_i_label_res]):
          mu1, _ = pred1
          Z1 = np.reshape(mu1, (nt, nt)) 

          ax2.plot_wireframe(X, Y, np.where(label, Z1, np.nan), 
                           color=color_cycle[1,:], antialiased=True,  
                           alpha=0.5)

        for ax in [ax1, ax2]:
          ax.scatter3D(self.x[self.f_c(self.x),0], 
              self.x[self.f_c(self.x),1], 
              self.y[self.f_c(x)], 
              s=50, color=color_cycle[2,:], alpha=1.0, depthshade=False, edgecolor='k')
          ax.scatter3D(self.x[self.f_i(self.x),0], 
                self.x[self.f_i(self.x),1], 
                self.y[self.f_i(x)], 
                s=50, color=color_cycle[2,:], alpha=1.0, depthshade=False, edgecolor='k')
          ax.set_xlabel('$x_1$')
          ax.set_ylabel('$x_2$')
          ax.set_zlabel('$y$')
          ax.set_xlim([np.min(xpred[:,0]), np.max(xpred[:,0])])
          ax.set_ylim([np.min(xpred[:,1]), np.max(xpred[:,1])])

          ax.view_init(20,-60)
        ax1.set_title('Continuous, {:s} kernel'.format(kernel.name.replace('_', '\_')))
        ax2.set_title('Discontinuous, {:s} kernel'.format(kernel.name.replace('_', '\_')))
  #
  def plot_effect_size(self, fig=None, plot_opts=dict()):

    # log_bfs = self.get_bayes_factor()
    es_bma = self.bma_effect_size()
    es_m1 = self.discontinuous_effect_size_mean_var()
    pmp = self.get_model_posterior()

    if self.ndim == 1:
      if fig is None:
        fig, axes = plt.subplots(nrows=len(self.kernels), ncols=1, 
                                figsize=(15, 6*len(self.kernels)), sharex=True, sharey=True)

        width_factor = plot_opts.get('width_factor', 4.0)

        dmins = np.array([es_m1[k][0] - width_factor*np.sqrt(es_m1[k][1]) for k in range(len(self.kernels))])
        dmaxs = np.array([es_m1[k][0] + width_factor*np.sqrt(es_m1[k][1]) for k in range(len(self.kernels))])

        dmin = np.min(dmins)
        dmax = np.max(dmaxs)
        d_range = np.linspace(dmin, dmax, num=100)

        for k, kernel in enumerate(self.kernels):
          ax = axes[k]

          d_mean, d_var = es_m1[k]
          
          d_pdf = st.norm.pdf(d_range, loc=d_mean, scale=np.sqrt(d_var))
          ax.plot(d_range, d_pdf, color=color_cycle[1,:], lw=2, ls='-', label=r'$p(d | D, m_1)$')
          ax.fill_between(d_range, d_pdf, color=color_cycle[1,:], alpha=0.5)
          ax.axvline(x=0, ls='--', lw=2.0, c='k', label=r'$p(d | D, m_0)$')

          pmc, pmd = pmp[k]

          nsamples = plot_opts.get('nsamples', 25000)
          samples = np.zeros((nsamples))
          dsamples = int(pmd*nsamples)
          if dsamples > 0:
            samples[0:dsamples] = np.random.normal(loc=d_mean, scale=np.sqrt(d_var), size=dsamples)
          spike_and_slab = st.kde.gaussian_kde(samples, bw_method='silverman')
          ax.plot(d_range, spike_and_slab(d_range), label=r'$p(d | D)$', lw=2, ls=':', color=color_cycle[0,:])
          ax.fill_between(d_range, spike_and_slab(d_range), color=color_cycle[0,:], alpha=0.5)

          inset = ax.inset_axes([0.6, 0.4, 0.4, 0.4])

          wedges = inset.pie([pmc, pmd], explode=[0.05, 0.05], 
                 labels=[np.round(pmc, 2), np.round(pmd,2)], 
                 colors=[color_cycle[0,:], color_cycle[1,:]],
                 labeldistance=1.2, 
                 wedgeprops = {'linewidth': 2, 
                               'alpha': 0.5, 
                               'linestyle': '-'},
                 textprops = {'fontsize': 20})

          ax.set_ylabel(kernel.name.replace('_', '\_').capitalize())
          ax.set_ylim(bottom=0)

        axes[-1].set_xlim([dmin, dmax])
        axes[-1].set_xlabel('$d$')
        handles, labels = axes[-1].get_legend_handles_labels()        
        plt.subplots_adjust(hspace=0.2, left=0, right=1.0, top=0.9, bottom=0.05)
        plt.figlegend(handles, labels, ncol=3, frameon=False, 
                      loc='lower center', fontsize=18)

    elif self.ndim == 2:
      raise NotImplementedError('Effect size plot for D={:d} is not available.'.format(self.ndim))
    else:
      raise NotImplementedError('Effect size plot for D={:d} is not available.'.format(self.ndim))
  #
  def get_bayes_factor(self):
    """
    Right now, we approximate p(y|x) with p(y|theta*, x), where 
    
      theta* = argmax_theta p(y|theta, x) p(theta).

    This can be improved by e.g. bridge sampling, Bayesian quadrature, HMC, etc.
    """

    log_bfs = list()
    for i in range(len(self.kernels)):
      log_bfs.append(self.m1[i].log_hyper_marginal_likelihood() - self.m0[i].log_hyper_marginal_likelihood())

    return log_bfs
  #
  def discontinuous_effect_size_mean_var(self):
    """
    Given m1 and Gaussian likelihood, the estimated effect size follows a 
    Gaussian distribution for which we have an analytic form. With different 
    likelihoods, this becomes more complex, and is considered future work.
    """

    es = list()
    for i in range(len(self.kernels)):
      es.append([self.m1[i].effect_size(self.ip)[0], self.m1[i].effect_size(self.ip)[1]])
    return es
  #
  def get_bma_effect_size_monte_carlo(self, n_mc=5000):

    es_m1 = self.discontinuous_effect_size_mean_var()
    pmp = self.get_model_posterior()
    bma_es_mc = list()

    for k, kernel in enumerate(self.kernels):
      pmc, pmd = pmp[k]
      d_mean, d_var = es_m1[k]
      samples = np.zeros((n_mc))
      dsamples = int(pmd*n_mc)
      if dsamples > 0:
        samples[0:dsamples] = np.random.normal(loc=d_mean, scale=np.sqrt(d_var), size=dsamples)
      bma_es_mc.append(samples)
    return bma_es_mc

  #
  def bma_effect_size(self):
    """

    p(d) = \sum_m p(d|m)p(m)

    p(d|m0) = dirac_d==0, so p(d) = p(d|m1)p(m1)

    """
    es_bma = list()
    es = self.discontinuous_effect_size_mean_var()
    model_posteriors = self.get_model_posterior()
    for i in range(len(self.kernels)):
      p0, p1 = model_posteriors[i]
      d_1 = es[i][0]
      es_bma.append(d_1*p1)
    return es_bma
  #
  def get_model_posterior(self):
    model_posteriors = list()
    log_bfs = self.get_bayes_factor()
    for i in range(len(self.kernels)):
      logbf10 = log_bfs[i]
      p_m1 = np.exp(logbf10) / (1 + np.exp(logbf10))
      p_m0 = 1 - p_m1
      model_posteriors.append([p_m0, p_m1])
    return model_posteriors
  #
  def print_hyperparameters(self):
    for i, kernel in enumerate(self.kernels):
      print('{:s} kernel'.format(kernel.name))
      print_summary(self.m0[i])
      print_summary(self.m1[i])
  #
  def get_marginal_likelihoods(self):
    ml0_all = list()
    ml1_all = list()
    for i in range(len(self.kernels)):
      ml0_all.append(self.m0[i].log_hyper_marginal_likelihood())
      ml1_all.append(self.m1[i].log_hyper_marginal_likelihood())
    return ml0_all, ml1_all

  def get_bma_bayes_factor(self):
    ml0_all, ml1_all = self.get_marginal_likelihoods()
    return log_sum_exp(np.asarray(ml1_all) - np.log(len(self.kernels))) - log_sum_exp(np.asarray(ml0_all - np.log(len(self.kernels))))
  #
  def get_marginal_bma_results(self):
    """
    TODO: the math might be off here, double check!
    """
    es_m1             = self.discontinuous_effect_size_mean_var()
    ml0_all, ml1_all  = self.get_marginal_likelihoods()

    bma_bf = self.get_bma_bayes_factor()
    bma_post_m1 = np.exp(bma_bf) / (1+np.exp(bma_bf))
    bma_post_m0 = 1 - bma_post_m1
    
    ml1_all_prob = np.exp(ml1_all - log_sum_exp(ml1_all))
    ml_all = ml1_all + ml0_all
    all_prob = np.exp(ml_all - log_sum_exp(ml_all))

    bma_disc_es = np.sum([ml1_all_prob[i]*es_m1[i][0] for i in range(len(self.kernels))])
    bma_disc_var = np.sum([ml1_all_prob[i]*es_m1[i][1] for i in range(len(self.kernels))])
    # first M all_prob are Discontinuous models
    bma_bma_es =  np.sum([all_prob[i]*es_m1[i][0] for i in range(len(self.kernels))])

    bma_marginal_m0 = log_sum_exp(ml0_all + np.log(1.0 / len(self.kernels) ))
    bma_marginal_m1 = log_sum_exp(ml1_all + np.log(1.0 / len(self.kernels) ))

    return {'BMA BF': bma_bf, 
            'BMA p(m0)': bma_post_m0, 
            'BMA p(m1)': bma_post_m1, 
            'BMA marginal m0': bma_marginal_m0, 
            'BMA marginal m1': bma_marginal_m1, 
            'BMA m1 es': bma_disc_es, 
            'BMA m1 es var': bma_disc_var, 
            'BMA marginal es': bma_bma_es}
  # 
  def get_marginal_bma_results_ND(self):
    """
    """
    ml0_all, ml1_all  = self.get_marginal_likelihoods()

    bma_bf = self.get_bma_bayes_factor()
    bma_post_m1 = np.exp(bma_bf) / (1+np.exp(bma_bf))
    bma_post_m0 = 1 - bma_post_m1
    
    ml1_all_prob = np.exp(ml1_all - log_sum_exp(ml1_all))
    ml_all = ml1_all + ml0_all
    all_prob = np.exp(ml_all - log_sum_exp(ml_all))

    bma_marginal_m0 = log_sum_exp(ml0_all + np.log(1.0 / len(self.kernels) ))
    bma_marginal_m1 = log_sum_exp(ml1_all + np.log(1.0 / len(self.kernels) ))

    return {'BMA BF': bma_bf, 
            'BMA p(m0)': bma_post_m0, 
            'BMA p(m1)': bma_post_m1, 
            'BMA marginal m0': bma_marginal_m0, 
            'BMA marginal m1': bma_marginal_m1}
  #
  def tabular_results(self):
    if self.ndim == 1:

      log_bfs           = self.get_bayes_factor()
      es_bma            = self.bma_effect_size()
      es_m1             = self.discontinuous_effect_size_mean_var()
      model_posteriors  = self.get_model_posterior()
      ml0_all, ml1_all  = self.get_marginal_likelihoods()

      colnames = ['log BF_10', 'p(m_0|D)', 'p(m_1|D)', 'p(D|m_0)', 'p(D|m_1)', 
                  'E[p(d|m_1)]', 'V[p(d|m1)]', 'E[p(d)]']

      t = PrettyTable(['Kernel'] + colnames)
      t.title = 'BNQD report'

      to_table_string = lambda n: '{:0.4f}'.format(n)

      # results per kernel
      for i, kernel in enumerate(self.kernels):
        name = kernel.name
        if name == 'polynomial':
          name += ' (d={:d})'.format(kernel.degree)
        t.add_row([name.capitalize(), 
                  to_table_string(log_bfs[i]),
                  to_table_string(model_posteriors[i][0]),
                  to_table_string(model_posteriors[i][1]),
                  to_table_string(ml0_all[i]),
                  to_table_string(ml1_all[i]),
                  to_table_string(es_m1[i][0]),
                  to_table_string(es_m1[i][1]),
                  to_table_string(es_bma[i])])

      bma_results = self.get_marginal_bma_results()

      t.add_row(['Bayesian model average', 
                to_table_string(bma_results['BMA BF']), 
                to_table_string(bma_results['BMA p(m0)']),
                to_table_string(bma_results['BMA p(m1)']),
                to_table_string(bma_results['BMA marginal m0']),
                to_table_string(bma_results['BMA marginal m1']),
                to_table_string(bma_results['BMA m1 es']),
                to_table_string(bma_results['BMA m1 es var']),
                to_table_string(bma_results['BMA marginal es'])])
      
      for col in colnames:
        t.align[col] = 'r'      
      t.align['Kernel'] = 'l'
    else:
      log_bfs           = self.get_bayes_factor()
      model_posteriors  = self.get_model_posterior()
      ml0_all, ml1_all  = self.get_marginal_likelihoods()

      colnames = ['log BF_10', 'p(m_0|D)', 'p(m_1|D)', 'p(D|m_0)', 'p(D|m_1)']

      t = PrettyTable(['Kernel'] + colnames)
      t.title = 'BNQD report'

      to_table_string = lambda n: '{:0.4f}'.format(n)

      # results per kernel
      for i, kernel in enumerate(self.kernels):
        name = kernel.name
        if name == 'polynomial':
          name += ' (d={:d})'.format(kernel.degree)
        t.add_row([name.capitalize(), 
                  to_table_string(log_bfs[i]),
                  to_table_string(model_posteriors[i][0]),
                  to_table_string(model_posteriors[i][1]),
                  to_table_string(ml0_all[i]),
                  to_table_string(ml1_all[i])])

      bma_results = self.get_marginal_bma_results_ND()

      t.add_row(['Bayesian model average', 
                to_table_string(bma_results['BMA BF']), 
                to_table_string(bma_results['BMA p(m0)']),
                to_table_string(bma_results['BMA p(m1)']),
                to_table_string(bma_results['BMA marginal m0']),
                to_table_string(bma_results['BMA marginal m1'])])
      
      for col in colnames:
        t.align[col] = 'r'      
      t.align['Kernel'] = 'l'
    
    return t
  #
#
