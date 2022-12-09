### Utility functions for PD BGM Model ========================================

### Imports ===================================================================
import os
import numpy as np

### Globus Pallidus (Both GPi and GPe): =======================================

def gpe_ainf(v):
    return 1 / (1 + np.exp(-(v+57) / 2))

def gpe_hinf(v):
    return 1 / (1 + np.exp((v+58) / 12))

def gpe_minf(v):
    return 1 / (1 + np.exp(-(v+37) / 10))

def gpe_ninf(v):
    return 1 / (1 + np.exp(-(v+50) / 14))

## NOTE: not provided in the Lu implementation.
def gpe_rinf(v):
    return 1 / (1 + np.exp((v+70) / 2))

def gpe_sinf(v):
    return 1 / (1 + np.exp(-(v+35) / 2))

def gpe_tauh(v):
    return 0.05 + 0.27 / (1 + np.exp(-(v+40) / -12))

def gpe_taun(v):
    ## yes, same formula as tauh
    return 0.05 + 0.27 / (1 + np.exp(-(v+40) / -12))



### Subthalamic Nucleus (STN) =================================================

def stn_ainf(v):
    return 1 / (1 + np.exp(-(v+63) / 7.8))


def stn_binf(r):
    ## Note: the input here is r, not v
    return 1 / (1 + np.exp(-(r-0.4)/0.1)) - 1 / (1 + np.exp(0.4 / 0.1))

def stn_cinf(v):
    return 1 / (1 + np.exp(-(v+20) / 8))

def stn_hinf(v):
    return 1 / (1 + np.exp((v+39) / 3.1))

def stn_minf(v):
    return 1 / (1 + np.exp(-(v+30) / 15))

def stn_ninf(v):
    return 1 / (1 + np.exp(-(v+32) / 8.0))

def stn_rinf(v):
    return 1 / (1 + np.exp((v+67) / 2))

def stn_sinf(v):
    return 1 / (1 + np.exp(-(v+39) / 8))

def stn_tauc(v):
    return 1 + 10 / (1 + np.exp((v+80) / 26))

def stn_tauh(v):
    return 1 + 500 / (1 + np.exp(-(v+57) / -3))

def stn_taun(v):
    return 1 + 100 / (1 + np.exp(-(v+80) / -26))

def stn_taur(v):
    return 7.1 + 17.5 / (1 + np.exp(-(v-68) / -2.2))


### Thalamus Neurons (Th) =====================================================

def th_hinf(v):
    return 1 / (1 + np.exp((v + 41) / 4))

def th_minf(v):
    return 1 / (1 + np.exp(-(v+37) / 7))

def th_pinf(v):
    return 1 / (1 + np.exp(-(v+60) / 6.2))

def th_rinf(v):
    return 1 / (1 + np.exp((v+84) / 4))

### different functions for these in the two implementations
def th_tauh_naumann(v):
    return 1 / (ah(v) + bh(v))

def ah(v):
    return 0.128 * np.exp(-(v+46) / 18)

def bh(v):
    return 4 / (1 + np.exp(-(v+23) / 5))

def th_tauh_lu(v):
    return 1 / (1 + np.exp((v+41) / 4))

def th_taur(v):
    return 0.15 * (28 + np.exp(-(v+25) / 10.5))

## GPe synaptic outflow function in high-level equations.
def Hinf(v):
    return 1 / (1 + np.exp(-(v+57) / 2))
