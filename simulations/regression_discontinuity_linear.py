# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 13:11:10 2019

@author: Max Hinne
"""

import GPy
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings; warnings.simplefilter('ignore')
from rdd import rdd
import pandas as pd
import sys
from tqdm import tqdm

import importlib
sys.path.append('../')

import BNQD
importlib.reload(BNQD)

print("GPy version:      {}".format(GPy.__version__))
print("BNQD version:     {}".format(BNQD.__version__))

plt.close('all')


    #
def get_predictors(n):
#    return random_predictors(n)
    return even_predictors(n)
    #
def even_predictors(n):
    return np.linspace(-3, 3, n)    
    #
def random_predictors(n, xmin=-3, xmax=3):
    return np.sort(np.random.uniform(low=xmin, high=xmax, size=n))
    #
def linear_data_same_slope(n=50, slope=0.3, bias=0.0, b=0.0, disc=10.0, noise_sd=1.0):
    x = get_predictors(n)  
    f = bias + slope*x + disc*(x>b)
    y = np.random.normal(loc=f, scale=noise_sd)
    return x, y
#


D = 1
kernel_names    = ['Linear', 'Exponential', 'RBF', 'Matern32']
linear_kernel   = GPy.kern.Linear(D) + GPy.kern.Bias(D)
exp_kernel      = GPy.kern.Exponential(D)
Matern32_kernel = GPy.kern.Matern32(D)
RBF_kernel      = GPy.kern.RBF(D)
kernels         = [linear_kernel, exp_kernel, RBF_kernel, Matern32_kernel]

kernel_dict = dict(zip(kernel_names, kernels))

opts = dict()
opts['num_restarts'] = 50
opts['mode'] = 'BIC'
opts['verbose'] = False

b = 0.0
labelFunc = lambda x: x < b

# sim RDD
datafun = linear_data_same_slope


gaps = 2**np.linspace(-2, 2, 5)
num_sims = 100
n = 100
noise = 1.0 # note this is used as sd; but S/R = var(signal)/var(noise)

fig, axes = plt.subplots(nrows=1, ncols=len(gaps), figsize=(18,3), sharex=True, sharey=True)
for i, gap in enumerate(gaps):
    x, y = datafun(n=n, b=b, disc=gap, noise_sd=noise)
    axes[i].plot(x, y, linestyle='None', marker='o', color='k', ms=3)
    axes[i].axvline(x=b, linestyle=':', color='k')
    axes[i].set_title('b = {:0.2f}'.format(gap))
    axes[i].set_xlabel('x')
    axes[i].set_ylabel('y')
plt.suptitle('Example data sets')


results = list()

tic = time.time()
for run in tqdm(range(num_sims)):
    run = list()
    for disc in gaps:
        x, y = datafun(n=n, b=b, disc=disc, noise_sd=noise)
        
        qed = BNQD.BnpQedAnalysis(x, y, kernel_dict, labelFunc, b=b, opts=opts)
        _ = qed.train()
        results_df = qed.pretty_print(verbose=False)

        data = pd.DataFrame({'y':y, 'x': x})
        rddmodel = rdd.rdd(data, 'x', 'y', cut=b, verbose=False)
        fit = rddmodel.fit()
        rddpval = fit.pvalues['TREATED']
        gp_pvals = qed.get_rdd_p_values()
        
        simulation = dict()
        simulation['bnpqed'] = results_df
        simulation['freq'] = rddpval
        simulation['gp_pval'] = gp_pvals
        run.append(simulation)
        del qed
    results.append(run)

print('Done in {:f} seconds'.format(time.time() - tic))
            

            
bma_bfs = np.zeros((num_sims, len(gaps)))
pvals = np.zeros((num_sims, len(gaps)))

for i in range(num_sims):
    for j in range(len(gaps)):
        bma_bfs[i,j] = results[i][j]['bnpqed'].iloc[4,3]
        pvals[i,j] = results[i][j]['freq']



K = len(kernel_names)
M = len(gaps)

effect_sizes = np.zeros((M, (K+1)*2, num_sims))
bayes_factors = np.zeros((M, K+1, num_sims))
p_values = np.zeros((M, K+1, num_sims))

for j in range(M):    
    for i in range(num_sims):
        p_values[j,K,i] = results[i][j]['freq']
        for k in range(K):
            effect_sizes[j, k, i] = results[i][j]['bnpqed'].iloc[k,2] # M_D
            effect_sizes[j, K+k+1, i] = results[i][j]['bnpqed'].iloc[k,3] # BMA
            bayes_factors[j, k, i] = results[i][j]['bnpqed'].iloc[k, 1]
            p_values[j,k,i] = results[i][j]['gp_pval'][kernel_names[k]]
        bayes_factors[j, K, i] = results[i][j]['bnpqed'].iloc[K, 1]
        effect_sizes[j, K, i] = results[i][j]['bnpqed'].iloc[K,2]
        effect_sizes[j, 2*K+1, i] = results[i][j]['bnpqed'].iloc[K,3]

colors = ['#44C5CB', '#FCE315', '#F53D52', '#FF9200']
markers = ['o', 's', '^', 'x', 'v']

lw = 2
cs = 6
          
# Plot sim 01
fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, figsize=(18,6))

for k in range(K):
    axes[0].errorbar(x=np.arange(M),
                 y=np.mean(bayes_factors[:, k, :], axis=1),
                 yerr=np.std(bayes_factors[:, k, :], axis=1),
                 linestyle='-',
                 linewidth=lw, 
                 marker=markers[k],
                 capsize=cs,
                 color=colors[k],
                 label=kernel_names[k])
    axes[1].errorbar(x=np.arange(M),
                 y=np.mean(p_values[:, k, :], axis=1),
                 yerr=np.std(p_values[:, k, :], axis=1),
                 linestyle='-',
                 linewidth=lw, 
                 marker=markers[k],
                 capsize=cs,
                 color=colors[k],
                 label=kernel_names[k])
    axes[2].errorbar(x=np.arange(M), 
                 y=np.mean(effect_sizes[:, k, :], axis=1), 
                 yerr=0.5*np.std(effect_sizes[:, k, :], axis=1), 
                 linestyle='-',
                 linewidth=lw, 
                 marker=markers[k],
                 capsize=cs,
                 color=colors[k],
                 label='{:s}, M_D'.format(kernel_names[k]))    
    axes[2].errorbar(x=np.arange(M), 
                 y=np.mean(effect_sizes[:, K+k+1, :], axis=1), 
                 yerr=0.5*np.std(effect_sizes[:, K+k+1, :], axis=1), 
                 linestyle='--', 
                 linewidth=lw,
                 marker=markers[k],
                 capsize=cs,
                 color=colors[k],
                 label='{:s}, BMA'.format(kernel_names[k]))
axes[0].errorbar(x=np.arange(M),
             y=np.mean(bayes_factors[:, K, :], axis=1),
             yerr=np.std(bayes_factors[:, K, :], axis=1),
             linestyle='-',
             linewidth=lw, 
             marker=markers[K],
             capsize=cs,
             color='k',
             label='BMA')    
axes[1].errorbar(x=np.arange(M),
             y=np.mean(p_values[:, K, :], axis=1),
             yerr=np.std(p_values[:, K, :], axis=1),
             linestyle='-',
             linewidth=lw, 
             marker='*',
             capsize=cs,
             color='k',
             label='RDD p-value')
axes[2].errorbar(x=np.arange(M), 
             y=np.mean(effect_sizes[:, K, :], axis=1), 
             yerr=0.5*np.std(effect_sizes[:, K, :], axis=1), 
             linestyle='-',
             linewidth=lw,
             marker=markers[K],
             capsize=cs,
             color='k',
             label='M_D BMA')
axes[2].errorbar(x=np.arange(M), 
             y=np.mean(effect_sizes[:, 2*K+1, :], axis=1), 
             yerr=0.5*np.std(effect_sizes[:, 2*K+1, :], axis=1), 
             linestyle='--',
             linewidth=lw,
             marker=markers[K],
             capsize=cs,
             color='k',
             label='Total BMA')

for ax in axes:
    ax.axvline(x=gaps.index(noise), color='k', linestyle=':', linewidth=1)
    ax.set_xticks(range(M))
    ax.set_xticklabels(gaps)
    ax.set_xlabel('Discontinuity size')
axes[0].set_ylabel('Log Bayes factor')
axes[0].set_title('Bayesian')  
axes[1].set_ylabel('RDD p-value')
axes[1].set_title('Frequentist')  
axes[2].set_ylabel('Effect size')
axes[2].set_title('Effect size regularization via BMA')
plt.suptitle('Linear true process')
handles, labels = axes[2].get_legend_handles_labels()
plt.figlegend(handles, labels, loc='lower center', ncol=5)