# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 14:43:11 2020

@author: Max
"""

import GPy
import numpy as np
import matplotlib.pyplot as plt
import csv
import warnings; warnings.simplefilter('ignore')

from rdd import rdd
import pandas as pd
import importlib
import sys
sys.path.append('../')
import BNQD
importlib.reload(BNQD)

print("GPy version:      {}".format(GPy.__version__))
print("BNQD version:     {}".format(BNQD.__version__))

plt.close('all')

print('Demo script for basic analysis with BNQD')

plot_data = False

# for different true effect sizes:
for d_true in [0.2, 0.5, 1.5]:
    
    # generate data
    D = 1                                       # dimensionality
    n = 50                                      # number of observations
    
    x = np.linspace(-3, 3, n)                   # predictor
    x0 = 0                                      # point of intervention
    
    b0 = 0.2                                    # bias
    b1 = 0.3                                    # slope
    
    sigma = 0.5                                 # noise level
    
    mu = b0 + b1*x + d_true*(x>x0)              # true latent
    y = np.random.normal(loc=mu, scale=sigma)   # observations
    
    if plot_data:
        # show data
        plt.figure()
        plt.plot(x, mu, color='blue', linewidth=2, label='True function')
        plt.plot(x, y, linestyle='none', ms=6, marker='x', color='black', label='Observations')
        plt.axvline(x=x0, linestyle='--', linewidth=1, color='black', label='Intervention point')
        plt.title('True function and noisy observations')
        plt.legend()
    
    # make a dictionary of kernels and their names
    linear_kernel   = GPy.kern.Linear(D) + GPy.kern.Bias(D)
    exp_kernel      = GPy.kern.Exponential(D)
    RBF_kernel      = GPy.kern.RBF(D)
    kernel_names    = ['Linear', 'exponential', 'Gaussian']
    kernels         = [linear_kernel, exp_kernel, RBF_kernel]
    
    kernel_dict = dict(zip(kernel_names, kernels))
    
    # options to pass to BNQD
    opts = dict()
    opts['num_restarts']    = 50        # restarts for parameter optimization
    opts['mode']            = 'BIC'     # method for approximating evidence
    opts['verbose']         = False     # switch for more output
    
    label_func = lambda x: x < x0       # point x < x0 are 'controls'
    
    # do BNQD
    qed = BNQD.BnpQedAnalysis(x, y, kernel_dict, label_func, b=x0, opts=opts)
    qed.train()
    
    # plot output
    x_test = np.linspace(-3, 3, 100) # for smoother plots
    
    qed.plot_model_fits(x_test=x_test)
    qed.plot_effect_sizes()
    qed.plot_posterior_model_probabilities()