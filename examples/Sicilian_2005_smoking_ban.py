# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 12:30:28 2019

@author: Max Hinne
"""


import GPy
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import csv
import warnings; warnings.simplefilter('ignore')
import bisect
import pandas as pd

from rdd import rdd
import importlib
import sys
sys.path.append('../')
import BNQD
importlib.reload(BNQD)

print("GPy version:      {}".format(GPy.__version__))
print("BNQD version:     {}".format(BNQD.__version__))

plt.close('all')

print('Analysis of the instatement of the 2005 smoking ban in Sicily')

datafile = '..\\datasets\\Sicily smoking ban\\sicily.csv'
data = dict()

with open(datafile, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        for k, v in dict(row).items():
            if k in data:
                data[k].append(v)
            else:
                data[k] = [v]

data['year']    = np.array([int(y) for y in data['year']])
data['month']   = np.array([int(m) for m in data['month']])
data['aces']    = np.array([int(a) for a in data['aces']]) # acute coronary events
data['time']    = np.array([int(t) for t in data['time']]) # predictor
data['smokban'] = np.array([s=='1' for s in data['smokban']])
data['pop']     = np.array([float(t) for t in data['pop']])
data['stdpop']  = np.array([float(st) for st in data['stdpop']]) # age-standardized population numbers

# Some transformations, needed to keep kernel matrices positive definite with reasonable parameter estimates.
# We simply transform back and forth when necessary.
zscorex = lambda x: (x - np.mean(data['time'])) / np.std(data['time'])
zscorey = lambda x: (x - np.mean(aces_per_agestd_pop)) / np.std(aces_per_agestd_pop)
inv_zscorey = lambda z: z*np.std(aces_per_agestd_pop) + np.mean(aces_per_agestd_pop)

n = len(data['time'])
b = data['time'][data['smokban']][0] - 1 # the ban instantiated *before* this
bz = zscorex(b)

labelFunc = lambda x: x < bz

aces_per_agestd_pop = data['aces'] / data['stdpop']*10**5

x = zscorex(data['time'])
y = stats.zscore(aces_per_agestd_pop) # ACE / age-standardized population
D = 1 # this is a 1D time-series



linear_kernel           = GPy.kern.Linear(D) + GPy.kern.Bias(D)
exp_kernel              = GPy.kern.Exponential(D)
std_periodic_kernel     = GPy.kern.StdPeriodic(D, period=12)
_                       = std_periodic_kernel['period'].constrain_fixed()
matern32_kernel         = GPy.kern.Matern32(D)
RBF_kernel              = GPy.kern.RBF(D)


x_test = np.linspace(x[0], x[-1], num=100)  # interpolate between actual observations
tmp = list(x_test)
bisect.insort(tmp, bz)
x_test = np.asarray(tmp)


kernel_names    = ['Linear', 'Periodic', 'RBF']
kernels         = [linear_kernel, std_periodic_kernel, RBF_kernel]

kernel_dict = dict(zip(kernel_names, kernels))

opts = dict()
opts['num_restarts'] = 50
opts['mode'] = 'BIC'
opts['verbose'] = False

qed = BNQD.BnpQedAnalysis(x, y, kernel_dict, labelFunc, b=bz, opts=opts)
qed.train()

results_df  = qed.pretty_print()
gp_pvals    = qed.get_rdd_p_values()
rdddata     = pd.DataFrame({'y':y, 'x': x})
rddmodel    = rdd.rdd(rdddata, 'x', 'y', cut=bz, verbose=False)
fit         = rddmodel.fit()
rddpval     = fit.pvalues['TREATED']        



es_fig, es_axes = qed.plot_effect_sizes()
mf_fig, mf_axes = qed.plot_model_fits(x_test)
# do for bottom row
for ax in mf_axes[len(kernel_dict)-1, :]:
    ax.set_xticks(zscorex(np.arange(1, n, step=12)))
    ax.set_xticklabels(np.unique(data['year']))
    ax.set_xlabel('Time')
    
# do for left column
for ax in mf_axes.flatten():        
    ax.set_yticks(zscorey(np.arange(150, 276, step=25)))
    ax.set_yticklabels(np.arange(150, 276, step=25))        
    ax.set_ylabel('Std ACE rate x 10,000')
#mf_fig.suptitle('Total Bayes factor = {:0.2f}'.format(np.exp(total_log_bf)))

pmp_fig, pmp_axes = qed.plot_posterior_model_probabilities()

for kernel in kernel_names:
    summ = qed.results[kernel].summary(b=bz)
    zpred_at_b = summ['f(b)']
    pred_at_b = [np.squeeze(inv_zscorey(p)) for p in zpred_at_b]
    risk_reduction = 1 - pred_at_b[0] / pred_at_b[1]
    pmd = summ['pmp']['pmd']
    print(summ['pmp'])
    print('{:s}, risk reduction = {:0.2f}'.format(kernel, risk_reduction*100))
    print('{:s} BMA, risk reduction = {:0.2f}'.format(kernel, pmd*risk_reduction*100))
