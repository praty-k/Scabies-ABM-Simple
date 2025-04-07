# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 15:29:02 2025

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

def extract_pathway_probs(x):
    return x[1]
    

#%%
beta = elfi.Prior(scipy.stats.uniform, 0, 0+1)
InfD = elfi.Prior(scipy.stats.uniform, 7, 56-7)


#%% Pathway calibration

y_obs = primary.calibrate_status(beta_true, InfD_true, ImmD_true)
print(y_obs[1])
#%%
Y = elfi.Simulator(primary.calibrate_status, beta, InfD, ImmD_true, observed = y_obs)

S = elfi.Summary(extract_pathway_probs, Y)

d = elfi.Distance('euclidean', S)

rej = elfi.Rejection(d, batch_size = 1, seed=seed)

#%%
start = time.time()
result = rej.sample(n_samples = 100, quantile = 0.05)
end = time.time()

