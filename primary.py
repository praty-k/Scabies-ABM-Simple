# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 21:25:18 2025

@author: PKollepara
"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(seed = 42)

RndNums = rng.random(10000)

''' Scheme = SIIRRS '''

# Input params
beta = 1.3
InfD = 2
ImmD = 1
# Operational params
beta = 1.3
gamma = 2/InfD
mu = 2/ImmD

TargetPopSize = 1000
GrpSizes = np.array([int(0.3*TargetPopSize), 
                        int(0.25*TargetPopSize), 
                        int(0.25*TargetPopSize), 
                        int(0.1*TargetPopSize), 
                        int(0.1*TargetPopSize)])
NumGrps = len(GrpSizes)
PopSize = np.sum(GrpSizes)

ContactMatrix = rng.random((NumGrps, NumGrps))
State = np.zeros(TargetPopSize)



def init_arrays(NumGrps):
    S = GrpSizes-1 #Needs to be an array containing number of people in each S compartment
    Ia = np.ones(NumGrps)
    Ib = np.zeros(NumGrps)
    Ra = np.zeros(NumGrps)
    Rb = np.zeros(NumGrps)
    return S, Ia, Ib, Ra, Rb    

# Computing Rates
def compute_rates_probs(beta, gamma, mu, PopSize, S, Ia, Ib, Ra, Rb):
    RatesLeaveS = beta/PopSize * S * (np.sum(ContactMatrix*Ia, 1) + np.sum(ContactMatrix*Ib, 1))
    RatesLeaveIa = gamma*Ia
    RatesLeaveIb = gamma*Ib
    RatesLeaveRa = mu*Ra
    RatesLeaveRb = mu*Rb
    #return RatesLeaveS, RatesLeaveIa, RatesLeaveIb, RatesLeaveRa, RatesLeaveRb    
    AllRates = np.concatenate((RatesLeaveS, RatesLeaveIa, RatesLeaveIb, RatesLeaveRa, RatesLeaveRb))
    RateSum = np.sum(AllRates)
    AllPs = AllRates/RateSum
    return RateSum, AllPs, AllRates

Events = np.arange(0, 5*NumGrps, 1)


## A function that takes Event as input and does the operation on state variables
def step(Event, S, Ia, Ib, Ra, Rb):
    if Event<NumGrps:
        S[Event]-=1
        Ia[Event]+=1
    elif Event<2*NumGrps:
        Ia[Event - NumGrps]-=1
        Ib[Event - NumGrps]+=1
    elif Event<3*NumGrps:
        Ib[Event - 2*NumGrps]-=1
        Ra[Event - 2*NumGrps]+=1
    elif Event<4*NumGrps:
        Ra[Event - 3*NumGrps]-=1
        Rb[Event - 3*NumGrps]+=1
    elif Event<5*NumGrps:
        Rb[Event - 4*NumGrps]-=1
        S[Event - 4*NumGrps]+=1
    return S, Ia, Ib, Ra, Rb

NumSteps = 100000
ts = np.zeros(NumSteps)
Ss, Ias, Ibs, Ras, Rbs = np.zeros((NumSteps, NumGrps)), np.zeros((NumSteps, NumGrps)), np.zeros((NumSteps, NumGrps)), np.zeros((NumSteps, NumGrps)), np.zeros((NumSteps, NumGrps))
Ss[0, :], Ias[0, :], Ibs[0, :], Ras[0, :], Rbs[0, :] = S, Ia, Ib, Ra, Rb

for counter in range(1, NumSteps):
    if np.any(S<0) or np.any(Ia<0) or np.any(Ib<0) or np.any(Ra<0) or np.any(Rb<0):
        print('Error, negative values')
        break
    elif np.sum(Ia + Ib)!=0:
        RateSum, AllPs, AllRates = compute_rates_probs(beta, gamma, mu, PopSize, S, Ia, Ib, Ra, Rb)
        Event = rng.choice(Events, size = 1, p = AllPs)
        EventTimeStep = rng.exponential(scale = 1/RateSum)
        ts[counter] = ts[counter-1]+EventTimeStep
        S, Ia, Ib, Ra, Rb = step(Event, S, Ia, Ib, Ra, Rb)
    Ss[counter, :], Ias[counter, :], Ibs[counter, :], Ras[counter, :], Rbs[counter, :] = S, Ia, Ib, Ra, Rb

plt.plot(ts, (np.sum(Ias, 1) + np.sum(Ibs, 1))/PopSize, label = 'Infected')
plt.plot(ts, np.sum(Ss, 1)/PopSize, label = 'Susceptible')