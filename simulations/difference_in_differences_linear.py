# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 13:11:10 2019

@author: u341138
"""

import GPy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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


def rad2deg(theta):
    return theta/(2*np.pi)*360
def deg2rad(r):
    return r / 360*(2*np.pi)
#
def get_predictors(n):
    return random_predictors(n)
#    return even_predictors(n)
    #
def even_predictors(n):
    return np.linspace(-3, 3, n)    
    #
def random_predictors(n, xmin=-3, xmax=3):
    return np.sort(np.random.uniform(low=xmin, high=xmax, size=n))
    #
def linear_data_different_slope(n=50, slope=1.0, angle=15, bias=1.2, b=0.0, noise_sd=1.0):
    x = get_predictors(n)   
    theta = np.arctan(slope) # angle of pre-intervention with horizontal
    alpha = deg2rad(angle)    
    slope2 = np.tan(theta-alpha)
    f = bias + slope*x*(x<=b) + slope2*x*(x>b)
    y = np.random.normal(loc=f, scale=noise_sd)
    return x, y, f
    #


D = 1
num_restarts    = 50 # 50
kernel_names    = ['Linear', 'Exponential', 'RBF', 'Matern32']
linear_kernel   = GPy.kern.Linear(D) + GPy.kern.Bias(D)
exp_kernel      = GPy.kern.Exponential(D)
Matern32_kernel = GPy.kern.Matern32(D)
RBF_kernel      = GPy.kern.RBF(D)
kernels         = [linear_kernel, exp_kernel, RBF_kernel, Matern32_kernel]

kernel_dict     = dict(zip(kernel_names, kernels))

opts = dict()
opts['num_restarts'] = num_restarts
opts['mode'] = 'BIC'
opts['verbose'] = False
opts['design'] = 'DiD'

b = 0.0
labelFunc = lambda x: x < b


datafun = linear_data_different_slope


angles = [30, 60, 90] # degrees, 90
num_sims = 100
n = 100
noise_levels = [0.01, 0.1, 1.0] 


# only plot on Windows
fig, axes = plt.subplots(nrows=len(noise_levels), ncols=len(angles), figsize=(18,3), sharex=True, sharey=True)

for i, theta in enumerate(angles):
    for j, noise_sd in enumerate(noise_levels):
        x, y, f = datafun(n=n, b=b, angle=theta, noise_sd=noise_sd)
        axes[j,i].plot(x, y, linestyle='None', marker='o', color='k', ms=2)
        axes[j,i].axvline(x=b, linestyle=':', color='k')
        axes[j,i].set_title(r'$\theta$ = {:0.2f}'.format(theta))
        axes[j,i].set_xlabel('x')
        axes[j,i].set_ylabel('y')
plt.draw()
plt.suptitle('Example data sets')


results = list()

for run in tqdm(range(num_sims)):
    run = list()
    for noise in noise_levels:
        run_noise = list()
        for angle in angles:
            x, y, _ = datafun(n=n, b=b, angle=angle, noise_sd=noise)
            
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
            run_noise.append(simulation)
            
            del qed
        run.append(run_noise)
    results.append(run)
        

     
num_kernels = len(kernel_names)
num_angles = len(angles)
num_noise = len(noise_levels)

bayes_factors = np.zeros((num_sims, num_noise, num_angles, num_kernels))
bma_bayes_factors = np.zeros((num_sims, num_noise, num_angles))

for i in range(num_sims):
    for j in range(num_noise):
        for k in range(num_angles):
            bma_bayes_factors[i,j,k] = results[i][j][k]['bnpqed'].iloc[num_kernels, 1]
            for l in range(num_kernels):
                bayes_factors[i,j,k,l] = results[i][j][k]['bnpqed'].iloc[l,1]
                
colors = ['#44C5CB', '#FCE315', '#F53D52', '#FF9200']
markers = ['o', 's', '^', 'x', 'v']
lw = 2
cs = 6

fig, axes = plt.subplots(nrows=1, ncols=num_noise, sharex=True, figsize=(18,6))                    
for j in range(num_noise):
    axes[j].set_title(r'$\sigma$ = {:0.2f}'.format(noise_levels[j]))
    inset = inset_axes(axes[j], width="60%", height="30%", loc=8)
    inset.errorbar(x=np.arange(num_angles),
                         y=np.mean(bayes_factors[:, j, :, 0], axis=0),
                         yerr=np.std(bayes_factors[:, j, :, 0], axis=0),
                         linestyle='-',
                         linewidth=lw, 
                         marker=markers[0],
                         capsize=cs,
                         color=colors[0],
                         label=kernel_names[0])
    inset.set_xticklabels([])
    inset.patch.set_alpha(0.7)
    for k in range(1, num_kernels):
        axes[j].errorbar(x=np.arange(num_angles),
                         y=np.mean(bayes_factors[:, j, :, k], axis=0),
                         yerr=np.std(bayes_factors[:, j, :, k], axis=0),
                         linestyle='-',
                         linewidth=lw, 
                         marker=markers[k],
                         capsize=cs,
                         color=colors[k],
                         label=kernel_names[k])
    axes[j].errorbar(x=np.arange(num_angles),
                     y=np.mean(bma_bayes_factors[:, j, :], axis=0),
                     yerr=np.std(bma_bayes_factors[:, j, :], axis=0),
                     linestyle='-',
                     linewidth=lw, 
                     marker=markers[num_kernels],
                     capsize=cs,
                     color='k',
                     label='BMA')  
    axes[j].set_xticks(range(num_angles))
    axes[j].set_xticklabels(angles)
    axes[j].set_xlabel('Difference in slope ($^{\circ}$)')
    axes[j].axhline(y=0, linestyle=':', color='k')
axes[0].set_ylabel('Log Bayes factor')
        
handles, labels = inset.get_legend_handles_labels()    
handles2, labels2 = axes[2].get_legend_handles_labels()
handles += handles2
labels += labels2
plt.figlegend(handles, labels, loc='lower center', ncol=5)

    