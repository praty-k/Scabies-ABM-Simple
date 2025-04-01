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

import primary
#%%
seed = 42
rng = np.random.default_rng(seed)

#%% True params
beta_true = 0.29
InfD_true = 14
ImmD_true = 42

y_obs = primary.calibrate2(beta_true, InfD_true, ImmD_true)[1]

def extract_summary_stats(x):
    return x[1]

print(f'obs_prev = extract_summary_stats(y_obs)')

#%%
beta = elfi.Prior(scipy.stats.uniform, 0, 0+1)

Y = elfi.Simulator(primary.calibrate2, beta, InfD_true, ImmD_true, observed = y_obs)

S = elfi.Summary(extract_summary_stats, Y)

d = elfi.Distance('euclidean', S)

#%%
rej = elfi.Rejection(d, batch_size = 100)