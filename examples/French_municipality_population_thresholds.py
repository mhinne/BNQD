# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 12:30:28 2019

@author: Max Hinne
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

print('Analysis of French population thresholds on municipal policies')

markers     = ['o', 'd', '^', 's']
colors      = ['#c70d3a', '#ed5107', '#230338', '#02383c']
          
def read_data(datafile='..\\datasets\\French municipality sizes\\france_pops_long_including_2006_2011.csv'): 
    print('Reading data')
    data = dict()
    
    with open(datafile, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            for k, v in dict(row).items():            
                if k in data:
                    if v != 'NA':
                        data[k].append(v)
                else:
                    data[k] = [v]
    
    data['id']    = np.array([i for i in data['id']])
    data['year']   = np.array([int(y) for y in data['year']])
    data['pop']    = np.array([int(p) for p in data['pop']]) 
    return data

def processed_data(data, thresholds, intv=200):
    print('Formatting data structure')
    datasets = list()
    
    for i, t in enumerate(thresholds):
        min_pop = np.max([0, t - intv])
        max_pop = np.min([t + intv, 2000])
        pop_select = np.logical_and(data['pop'] > min_pop, data['pop'] < max_pop)
        y = data['pop'][pop_select] - t
        x = np.linspace(-intv, intv, num=2*intv+1)
        counts, bins = np.histogram(y, bins=x)
        populated = counts > 0
        counts = counts[populated]  
        centers = (bins[:-1] + bins[1:]) / 2
        centers = centers[populated] 
        datasets.append((t, centers, counts))        
    return datasets

def plot_data(datasets, intv=200):
    print('Visualize data')
    plt.figure()
    
    for i, dataset in enumerate(datasets):
        t, centers, counts = dataset              
        plt.plot(centers, counts, color=colors[i], 
                 label='Threshold = {:d}'.format(t), linestyle='None', 
                 marker=markers[i], alpha=0.4)  
        
    plt.xlim([-intv, intv])
    plt.title('Municipality population sizes')
    plt.axvline(x=0, color='k', linestyle='--')  
    plt.xlabel('Distance from threshold')
    plt.ylabel('Frequency')
    plt.ylim([0, 650])
    plt.legend(loc='best')
    plt.show()
    


data = read_data()

zscore = lambda x: (x - np.mean(x)) / np.std(x)

thresholds = [100, 500, 1000, 1500]
intv = 200

datasets = processed_data(data, thresholds, intv=intv)

M               = len(datasets)
D               = 1
labelFunc       = lambda x: x >= 0.0

linear_kernel   = GPy.kern.Linear(D) + GPy.kern.Bias(D)
matern32_kernel = GPy.kern.Matern32(D)
sqexp_kernel    = GPy.kern.RBF(D)
kernels         = [linear_kernel, sqexp_kernel]
kernel_names    = ['Linear', 'RBF']
    

ztransform_invert = lambda z, data: z*np.std(data) + np.mean(data)

kernel_dict = dict(zip(kernel_names, kernels))

opts = dict()
opts['num_restarts']    = 50
opts['mode']            = 'BIC'
opts['verbose']         = False


b = 0.0


results = dict()    
for i, dataset in enumerate(datasets):    
    t, x_raw, y_raw = dataset
    x_test = np.linspace(np.min(x_raw), np.max(x_raw), num=100)
    print('Threshold = {:d}'.format(t))
    
    bnpqed = BNQD.BnpQedAnalysis(x=x_raw, y=y_raw, kernel_dict=kernel_dict, 
                                 labelFunc=labelFunc, b=b, opts=opts)
    bnpqed.train()
    result_df = bnpqed.pretty_print()
    gp_pvals = bnpqed.get_rdd_p_values()
    data = pd.DataFrame({'y':y_raw, 'x': x_raw})
    rddmodel = rdd.rdd(data, 'x', 'y', cut=b, verbose=False)
    fit = rddmodel.fit()
    rddpval = fit.pvalues['TREATED']        
    res = dict()
    res['bnpqed'] = bnpqed
    res['freq'] = rddpval
    res['gp_pval'] = gp_pvals    
    results[dataset[0]] = res     
    
    
# plotting    
f, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(18, 8))
for ax in axes:    
    for i, dataset in enumerate(datasets):
        t, centers, counts = dataset              
        ax.plot(centers, counts, color=colors[i], 
                 label='Threshold = {:d}'.format(t), linestyle='None', 
                 marker=markers[i], alpha=0.4)  
        
    ax.set_xlim([-intv, intv])
    ax.set_title('Municipality population sizes')
    ax.axvline(x=0, color='k', linestyle='--')  
    ax.set_xlabel('Distance from threshold')
    ax.set_ylabel('Frequency')
    ax.set_ylim([0, 650])
    
for i, dataset in enumerate(datasets):
    plot_opts = dict()
    plot_opts['marker_pre']         = markers[i]
    plot_opts['marker_post']        = markers[i]
    plot_opts['color_data']         = colors[i]
    plot_opts['marker_alpha']       = 0.1
    plot_opts['marker_size']        = 6
    plot_opts['plot_xlim']          = [-intv, intv]
    plot_opts['plot_ylim']          = [0, 650]
    plot_opts['plot_same_window']   = True
    plot_opts['axes']               = axes
    t, x_raw, y_raw = dataset
    x_test = np.linspace(np.min(x_raw), np.max(x_raw), num=100)
    _ = results[t]['bnpqed'].plot_model_fits(x_test, plot_opts)

