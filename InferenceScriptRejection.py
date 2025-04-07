# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 11:35:46 2025

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

print(f'obs_prev = {extract_summary_stats(y_obs)}')
#%%
beta = elfi.Prior(scipy.stats.uniform, 0, 0+1)

# Y = elfi.Simulator(primary.calibrate2, beta, InfD_true, ImmD_true, observed = y_obs)

# S = elfi.Summary(extract_summary_stats, Y)

# d = elfi.Distance('euclidean', S)

# #%%
# rej = elfi.Rejection(d, batch_size = 1)

# start = time.time()
# result = rej.sample(n_samples = 100, quantile = .1)
# end = time.time()

# print(end-start, 'seconds')

#%%
InfD = elfi.Prior(scipy.stats.uniform, 7, 56-7)

Y2 = elfi.Simulator(primary.calibrate2, beta, InfD, ImmD_true, observed = y_obs)

S2 = elfi.Summary(extract_summary_stats, Y2)

d2 = elfi.Distance('euclidean', S2)

rej2 = elfi.Rejection(d2, batch_size = 1)

start = time.time()
result2 = rej2.sample(n_samples = 100, quantile = .05)
end = time.time()

print((end-start)/60, 'minutes to completion')

result2.plot_pairs()
plt.savefig('Sample_serial.png')

#%%