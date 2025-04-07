# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 11:47:37 2025

@author: pkollepara
"""

import elfi
import time
import matplotlib.pyplot as plt
import numpy as np
import scipy
import logging

import primary_numba_test as primary
#%%
seed = 42
rng = np.random.default_rng(seed)

#%% True params
beta_true = 0.29
InfD_true = 14
ImmD_true = 42

y_obs = primary.calibrate2(beta_true, InfD_true, ImmD_true)

def extract_summary_stats(x):
    return x[1]

def extract_summary_stats_age(x):
    return np.array([x[1].flatten()])

def extract_pathway_probs(status):
    

#%%
beta = elfi.Prior(scipy.stats.uniform, 0, 0+1)
InfD = elfi.Prior(scipy.stats.uniform, 7, 56-7)


#%% Net prevalence calibration

Y = elfi.Simulator(primary.calibrate2, beta, InfD, ImmD_true, observed = y_obs)

S = elfi.Summary(extract_summary_stats, Y)

d = elfi.Distance('euclidean', S)

smc = elfi.SMC(d, batch_size = 1, seed = seed)

# #%%
# schedule = [0.8, 0.6, 0.4, 0.2, 0.1]

# start = time.time()
# result_smc = smc.sample(n_samples = 100, thresholds = schedule)
# end = time.time()

# print(end-start, 'seconds')

#%% calibrate using age prevalence
y_obs_age = primary.calibrate_age_prev(beta_true, InfD_true, ImmD_true)

Y_age = elfi.Simulator(primary.calibrate_age_prev, beta, InfD, ImmD_true, observed = y_obs_age)

S_age = elfi.Summary(extract_summary_stats_age, Y_age)

d_age = elfi.Distance('euclidean', S_age)

#smc_age = elfi.SMC(d_age, batch_size = 1, seed = seed)

# #%%
#schedule1 = [0.7, 0.35, 0.2, 0.1]

# start = time.time()
# result_smc1 = smc_age.sample(n_samples = 100, quantiles = [0.5, 0.25, 0.1])
# end = time.time()

# print(end-start, 'seconds')

# #%%
# schedule2 = [0.1, 0.068]
# start = time.time()
# result_smc2 = smc_age.sample(n_samples = 100, thresholds = schedule2)
# end = time.time()

# print(end-start, 'seconds')

#%%
rej_age = elfi.Rejection(d_age, batch_size = 1, seed=seed)

start = time.time()
result_rej = rej_age.sample(n_samples = 50, quantile = 0.05)
end = time.time()

#%%
rej_age2 = elfi.Rejection(d_age, batch_size = 1, seed=seed)

start = time.time()
result_rej2 = rej_age2.sample(n_samples = 50, quantile = 0.05)
end = time.time()

#%% calibrate using vectorized age prevalence 
y_obs_age = primary.calibrate_age_prev(beta_true, InfD_true, ImmD_true)

Y_age = elfi.Simulator(primary.calibrate_age_prev, beta, InfD, ImmD_true, observed = y_obs_age)

S_age = elfi.Summary(extract_summary_stats_age, Y_age)

d_age = elfi.Distance('euclidean', S_age)
