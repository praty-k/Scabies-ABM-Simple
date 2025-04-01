# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 13:43:22 2025

@author: pkollepara
"""


import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns
from numba import jit
np.random.seed(seed = 42)

#####

@jit
def create_status_vars(PopSize, GrpSizes, NumGrps):
    ID = np.empty((NumGrps, np.max(GrpSizes)))
    Status = np.empty((NumGrps, np.max(GrpSizes)))
    ID[:]=np.nan
    Status[:]=np.nan
    for ng in range(NumGrps):
        ID[ng, 0:GrpSizes[ng]] = np.arange(0, GrpSizes[ng], 1)
        Status[ng, 0:GrpSizes[ng]] = 0
    return Status   

@jit
def init_vars(PopSize, NumGrps, GrpSizes): #Initialises with one infection in each group
    S = GrpSizes-1 #Needs to be an array containing number of people in each S compartment
    Ia = np.ones(NumGrps)
    Ib = np.zeros(NumGrps)
    Ra = np.zeros(NumGrps)
    Rb = np.zeros(NumGrps)
    Status = create_status_vars(PopSize, GrpSizes, NumGrps)
    Status[:, 0] = 1 #Assign Infected (a) status for the first agent of each group
    return S, Ia, Ib, Ra, Rb, Status    

# Computing Rates
@jit
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
@jit
def step(Event, S, Ia, Ib, Ra, Rb, Status, GrpSizes, NumGrps, rnd):
    if Event<NumGrps:
        S[Event]-=1
        Ia[Event]+=1
        #Select random susceptible
        GrpNum = Event
        GrpSize = GrpSizes[GrpNum]
        Agents = np.where(Status[GrpNum] == 0)[0]
        SelectedAgent = Agents[int(rnd*len(Agents))]
        Status[GrpNum, SelectedAgent] = 1
    elif Event<2*NumGrps:
        Ia[Event - NumGrps]-=1
        Ib[Event - NumGrps]+=1
        #Select random Infected (a)
        GrpNum = Event - NumGrps
        GrpSize = GrpSizes[GrpNum]
        Agents = np.where(Status[GrpNum] == 1)[0]
        SelectedAgent = Agents[int(rnd*len(Agents))]
        Status[GrpNum, SelectedAgent] = 2
    elif Event<3*NumGrps:
        Ib[Event - 2*NumGrps]-=1
        Ra[Event - 2*NumGrps]+=1
        #Select random Infected (b)
        GrpNum = Event - 2*NumGrps
        GrpSize = GrpSizes[GrpNum]
        Agents = np.where(Status[GrpNum] == 2)[0]
        SelectedAgent = Agents[int(rnd*len(Agents))]
        Status[GrpNum, SelectedAgent] = 3
    elif Event<4*NumGrps:
        Ra[Event - 3*NumGrps]-=1
        Rb[Event - 3*NumGrps]+=1
        #Select random Recovered (a)
        GrpNum = Event - 3*NumGrps
        GrpSize = GrpSizes[GrpNum]
        Agents = np.where(Status[GrpNum] == 3)[0]
        SelectedAgent = Agents[int(rnd*len(Agents))]
        Status[GrpNum, SelectedAgent] = 4
    elif Event<5*NumGrps:
        Rb[Event - 4*NumGrps]-=1
        S[Event - 4*NumGrps]+=1
        #Select random Recovered (b)
        GrpNum = Event - 4*NumGrps
        GrpSize = GrpSizes[GrpNum]
        Agents = np.where(Status[GrpNum] == 4)[0]
        SelectedAgent = Agents[int(rnd*len(Agents))]
        Status[GrpNum, SelectedAgent] = 0
    return S, Ia, Ib, Ra, Rb, Status

@jit
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
    
    # The matrix below was randomly generated.
    ContactMatrix = np.array([[0.72072839, 0.71123776, 0.20269503, 0.0366554 , 0.30379952],
           [0.1571363 , 0.39578848, 0.97934612, 0.18107137, 0.31887394],
           [0.72511432, 0.50918278, 0.04392814, 0.16169002, 0.93524955],
           [0.10316499, 0.63509279, 0.79232565, 0.26543906, 0.725078  ],
           [0.16989518, 0.28475027, 0.31182203, 0.99643499, 0.2145723 ]])
    
    #ContactMatrix = np.ones((NumGrps, NumGrps))
    
    State = np.zeros(TargetPopSize)
    Events = np.arange(0, 5*NumGrps, 1) # 5 because there are five states, S, Ia, Ib, Ra, Rb
    S, Ia, Ib, Ra, Rb, StatusFull = init_vars(PopSize, NumGrps, GrpSizes)
    ts = np.zeros(NumSteps)
    StatusBrief = np.empty((NumSteps, PopSize))
    Ss, Ias, Ibs, Ras, Rbs = np.zeros((NumSteps, NumGrps)), np.zeros((NumSteps, NumGrps)), np.zeros((NumSteps, NumGrps)), np.zeros((NumSteps, NumGrps)), np.zeros((NumSteps, NumGrps))
    Ss[0, :], Ias[0, :], Ibs[0, :], Ras[0, :], Rbs[0, :] = S, Ia, Ib, Ra, Rb
    
    #Pre-generated random numbers
    rnds = np.random.uniform(0, 1, size = NumSteps)
    for counter in range(1, NumSteps):
        if np.any(S<0) or np.any(Ia<0) or np.any(Ib<0) or np.any(Ra<0) or np.any(Rb<0):
            print('Error, negative values')
            break
        elif np.sum(Ia + Ib)!=0:
            RateSum, AllPs, AllRates = compute_rates_probs(beta, gamma, mu, PopSize, S, Ia, Ib, Ra, Rb, ContactMatrix)
            #Event = np.random.choice(Events, size = 1, p = AllPs)[0]
            Event = Events[np.searchsorted(np.cumsum(AllPs), np.random.rand(1)[0])]
            EventTimeStep = np.random.exponential(scale = 1/RateSum)
            ts[counter] = ts[counter-1]+EventTimeStep
            S, Ia, Ib, Ra, Rb, StatusFull = step(Event, S, Ia, Ib, Ra, Rb, StatusFull, GrpSizes, NumGrps, rnds[counter])
            StatusFlat = StatusFull.flatten()
            StatusFlat = StatusFlat[~np.isnan(StatusFlat)]
            
        Ss[counter, :], Ias[counter, :], Ibs[counter, :], Ras[counter, :], Rbs[counter, :] = S, Ia, Ib, Ra, Rb
        StatusBrief[counter, :] = (StatusFlat == 1) | (StatusFlat == 2)
    return ts, Ss, Ias, Ibs, Ras, Rbs, StatusBrief, PopSize

def calibrate(beta, InfD, ImmD):
    NumSteps = 50000
    TargetPopSize = 1000
    ts, Ss, Ias, Ibs, Ras, Rbs, Status, PopSize = simulate(beta, InfD, ImmD, NumSteps, TargetPopSize)
    return ts, Ss/PopSize, Ias/PopSize, Ibs/PopSize, Ras/PopSize, Rbs/PopSize, Status

def sampler(SamplingTimes, ts, xs):
    Sample = np.empty(SamplingTimes.shape)
    Sample[:] = np.nan
    for counter1, SamplingTime in enumerate(SamplingTimes):
        for counter2, t in enumerate(ts):
            if t>SamplingTime:
                Sample[counter1] = xs[counter2-1]
                break
    return Sample

def sampler_for_status(SamplingTimes, ts, status):
    Sample = np.empty((SamplingTimes.shape[0], status.shape[1]))
    Sample[:] = np.nan
    for counter1, SamplingTime in enumerate(SamplingTimes):
        for counter2, t in enumerate(ts):
            if t>SamplingTime:
                Sample[counter1, :] = status[counter2-1, :]
                break
    return Sample

def calibrate2(beta, InfD, ImmD, batch_size = 1, random_state = None):
    NumSteps = 50000
    TargetPopSize = 1000
    ts, Ss, Ias, Ibs, Ras, Rbs, Status, PopSize = simulate(beta, InfD, ImmD, NumSteps, TargetPopSize)
    SamplingTimes = np.arange(0, ts[-1], 30)
    s_sample = sampler(SamplingTimes, ts, np.sum(Ss, 1))
    i_sample = sampler(SamplingTimes, ts, np.sum(Ias, 1) + np.sum(Ibs, 1))
    r_sample = sampler(SamplingTimes, ts, np.sum(Ras, 1) + np.sum(Rbs, 1))
    NumObs = 4
    summary_stat = sampler(SamplingTimes, ts, np.sum(Ias, 1) + np.sum(Ibs, 1))[-NumObs:]/PopSize
    # If an outbreak does not take off then return prevalence sample with zeros
    if len(summary_stat) == 0:
        summary_stat = np.zeros(NumObs)
    # If outbreak dies out then replace NaNs in the prevalence sample with zeros
    for jj in range(len(summary_stat)):
        if np.isnan(summary_stat[jj]):
            summary_stat[jj] = 0.0
    return (SamplingTimes, s_sample, i_sample, r_sample), np.array([summary_stat])

#%%
if __name__ == '__main__':
    ''' Scheme = SIIRRS '''

    # Input params
    beta = 5/7 # in 1/days
    InfD = 2*7 # in days
    ImmD = 1*7 # in days
    
    
    ts, ss, ias, ibs, ras, rbs, status = calibrate(beta, InfD, ImmD)
    plt.plot(ts, np.sum(ss, 1), label = 'Susceptible')
    plt.plot(ts, (np.sum(ias, 1)), label = 'Infected (a)')
    plt.plot(ts, (np.sum(ibs, 1)), label = 'Infected (b)')
    plt.plot(ts, np.sum(ras, 1), label = 'Recovered (a)')
    plt.plot(ts, np.sum(rbs, 1), label = 'Recovered (b)')
    plt.legend()
    
    print(ts[20], ias[20], ibs[20])
    
    
    SamplingTimes = np.arange(0, 400, 15)
    
    s_sample = sampler(SamplingTimes, ts, np.sum(ss, 1))
    i_sample = sampler(SamplingTimes, ts, np.sum(ias, 1) + np.sum(ibs, 1))
    r_sample = sampler(SamplingTimes, ts, np.sum(ras, 1) + np.sum(rbs, 1))
    
    
    status_sample = sampler_for_status(SamplingTimes, ts, status)
        
    #%%
    total_start = time.time()
    InfectiousDurations = np.arange(1, 9, 1)*7
    ImmuneDurations = np.arange(1, 9, 1)*7
    betas = np.arange(1, 15, 1)/7
    
    Prevalence = np.zeros((len(betas), len(InfectiousDurations), len(ImmuneDurations)))
    SimTime = np.zeros((len(betas), len(InfectiousDurations), len(ImmuneDurations)))
    for ii, beta in enumerate(betas):
        for jj, InfD in enumerate(InfectiousDurations):
            for kk, ImmD in enumerate(ImmuneDurations):
                start = time.time()
                ts, ss, ias, ibs, ras, rbs, status = calibrate(beta, InfD, ImmD)
                end = time.time()
                plt.plot(ts, np.sum(ss, 1), label = 'Susceptible')
                plt.plot(ts, (np.sum(ias, 1)), label = 'Infected (a)')
                plt.plot(ts, (np.sum(ibs, 1)), label = 'Infected (b)')
                plt.plot(ts, np.sum(ras, 1), label = 'Recovered (a)')
                plt.plot(ts, np.sum(rbs, 1), label = 'Recovered (b)')
                plt.legend()
                plt.savefig(f'SimTimeExploreNumba/{beta=:.2f}_{InfD=:.2f}_{ImmD=:.2f}.png', dpi = 300)
                plt.close()
                Prev = (np.sum(ias, 1) + np.sum(ibs, 1))[-1]
                Prevalence[ii, jj, kk] = Prev
                SimTime[ii, jj, kk] = end-start
    total_end = time.time()         
    print('Total time: ', total_end - total_start)
    
    #%%
    np.savez('SimTimeExploreNumba/Prevalence', Prevalence)
    np.savez('SimTimeExploreNumba/SimTime', SimTime)
    
    #%%
    for ii, beta in enumerate(betas):
        ax=sns.heatmap(Prevalence[ii], annot = True, xticklabels=ImmuneDurations, yticklabels=InfectiousDurations, cmap = 'Blues')
        ax.invert_yaxis()
        ax.set(xlabel = 'Immune Duration', ylabel='Infectious Duration')
        ax.set_title(f'Transmissibility = {beta:.2f}')
        plt.savefig(f'SimTimeExploreNumba/Heatmap_beta_{beta:.2f}.png')
        plt.close()