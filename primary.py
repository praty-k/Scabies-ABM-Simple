# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 21:25:18 2025

@author: PKollepara
"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(seed = 42)

#####


def init_state_vars(NumGrps, GrpSizes):
    S = GrpSizes-1 #Needs to be an array containing number of people in each S compartment
    Ia = np.ones(NumGrps)
    Ib = np.zeros(NumGrps)
    Ra = np.zeros(NumGrps)
    Rb = np.zeros(NumGrps)
    return S, Ia, Ib, Ra, Rb    

# Computing Rates
def compute_rates_probs(beta, gamma, mu, PopSize, S, Ia, Ib, Ra, Rb, ContactMatrix):
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

## A function that takes Event as input and does the operation on state variables
def step(Event, S, Ia, Ib, Ra, Rb, NumGrps):
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


def simulate(beta, InfD, ImmD, NumSteps, TargetPopSize):
    gamma = 2/InfD
    mu = 2/ImmD
    
    GrpSizes = np.array([int(0.3*TargetPopSize), 
                            int(0.25*TargetPopSize), 
                            int(0.25*TargetPopSize), 
                            int(0.1*TargetPopSize), 
                            int(0.1*TargetPopSize)])
    NumGrps = len(GrpSizes)
    PopSize = np.sum(GrpSizes)
    #ContactMatrix = rng.random((NumGrps, NumGrps))
    
    ContactMatrix = np.array([[0.72072839, 0.71123776, 0.20269503, 0.0366554 , 0.30379952],
           [0.1571363 , 0.39578848, 0.97934612, 0.18107137, 0.31887394],
           [0.72511432, 0.50918278, 0.04392814, 0.16169002, 0.93524955],
           [0.10316499, 0.63509279, 0.79232565, 0.26543906, 0.725078  ],
           [0.16989518, 0.28475027, 0.31182203, 0.99643499, 0.2145723 ]])
    State = np.zeros(TargetPopSize)
    Events = np.arange(0, 5*NumGrps, 1) # 5 because there are five states, S, Ia, Ib, Ra, Rb
    S, Ia, Ib, Ra, Rb = init_state_vars(NumGrps, GrpSizes)
    ts = np.zeros(NumSteps)
    Ss, Ias, Ibs, Ras, Rbs = np.zeros((NumSteps, NumGrps)), np.zeros((NumSteps, NumGrps)), np.zeros((NumSteps, NumGrps)), np.zeros((NumSteps, NumGrps)), np.zeros((NumSteps, NumGrps))
    Ss[0, :], Ias[0, :], Ibs[0, :], Ras[0, :], Rbs[0, :] = S, Ia, Ib, Ra, Rb
    for counter in range(1, NumSteps):
        if np.any(S<0) or np.any(Ia<0) or np.any(Ib<0) or np.any(Ra<0) or np.any(Rb<0):
            print('Error, negative values')
            break
        elif np.sum(Ia + Ib)!=0:
            RateSum, AllPs, AllRates = compute_rates_probs(beta, gamma, mu, PopSize, S, Ia, Ib, Ra, Rb, ContactMatrix)
            Event = rng.choice(Events, size = 1, p = AllPs)
            EventTimeStep = rng.exponential(scale = 1/RateSum)
            ts[counter] = ts[counter-1]+EventTimeStep
            S, Ia, Ib, Ra, Rb = step(Event, S, Ia, Ib, Ra, Rb, NumGrps)
        Ss[counter, :], Ias[counter, :], Ibs[counter, :], Ras[counter, :], Rbs[counter, :] = S, Ia, Ib, Ra, Rb
    
    return ts, Ss, Ias, Ibs, Ras, Rbs
    
####
''' Scheme = SIIRRS '''

# Input params
beta = 1.5
InfD = 2
ImmD = 1
NumSteps = 50000
PopSize = 1000

ts, Ss, Ias, Ibs, Ras, Rbs = simulate(beta, InfD, ImmD, NumSteps, PopSize)
plt.plot(ts, (np.sum(Ias, 1) + np.sum(Ibs, 1))/PopSize, label = 'Infected')
plt.plot(ts, np.sum(Ss, 1)/PopSize, label = 'Susceptible')
print(ts[20], Ias[20], Ibs[20])
