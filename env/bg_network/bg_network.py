### High-Level PD BGM Model Equations =========================================

### Imports ===================================================================
import os, sys, warnings
import glob
import time
import math, random
import numpy as np
import pandas as pd
import pickle as pkl
from itertools import combinations
import scipy.stats as st
import scipy.signal as sp_sg
from scipy.integrate import simps
import gym
from gym import spaces
from matplotlib import pyplot as plt
import seaborn as sns
import rl_modules.envs.bg_network.utils as utils
import gc
from pyNN import space

np.seterr(divide='ignore', invalid='ignore')
## FIXED CONSTANTS:
Cm=1 #membrane capacitance, micro-Farad

### Membrane parameters, in order of Th, STN, GP or Th, STN, GPe, GPi

## Consistent
gl = {'Th': 0.05, 'STN': 2.25, 'GP': 0.1}
El = {'Th': -70, 'STN': -60, 'GP': -65}

## Consistent
gna = {'Th': 3, 'STN': 37, 'GP': 120}
Ena = {'Th': 50, 'STN': 55, 'GP': 55}

## Consistent
gk = {'Th': 5, 'STN': 45, 'GP': 30}
Ek = {'Th': -75, 'STN': -80, 'GP': -80}

## Consistent
gt = {'Th': 5, 'STN': 0.5, 'GP': 0.5}
Et = 0


## Consistent
gca = {'Th': 0, 'STN': 2, 'GP': 0.15}
Eca = {'Th': 0, 'STN': 140, 'GP': 120}

## Consistent
gahp = {'Th': 0, 'STN': 20, 'GP': 10}

## Consistent
k1 = {'Th': 0, 'STN': 15, 'GP': 10}

## Consistent
kca = {'Th': 0, 'STN': 22.5, 'GP': 15}



A = {'Th': 0, 'STN': 3, 'GPe': 2, 'GPi': 2}
B = {'Th': 0, 'STN': 0.1, 'GPe': 0.04, 'GPi': 0.04}

the = {'Th': 0, 'STN': 30, 'GPe': 20, 'GPi': 20}


## See sim_output/healthy_beta_rel_power_means.csv for the numbers used here.
## TODO: note these could change as analysis is finalized.
healthy_vgi_b_mean = 0.017037733592451
healthy_vge_b_mean = 0.0206326633836837
healthy_vsn_b_mean = 0.130433121350334



# Synapse parameters
# In order of Igesn,Isnge,Igege,Isngi,Igegi,Igith

## NOTE: Naumann: 0.17; Meili Lu: .112 (probably produces harsher PD symptoms.)
gsyn = {
    'Igesn': 0.5, 'Isnge': 0.15, 'Igege': 0.5, 
    'Isngi': 0.15, 'Igegi': 0.5, 'Igith': 0.17}

## consistent
Esyn = {
    'Igesn': -85, 'Isnge': 0, 'Igege': -85, 
    'Isngi': 0, 'Igegi': -85, 'Igith': -85}  ## MiliVolts (Inhibitory vs excitatory)

tau = 5 
gpeak = 0.43 
gpeak1 = 0.3  

## averaged values of S_GI from https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8895773
bar_S = [
    0.3220, 0.2985, 0.3211, 0.3334, 0.3230,
    0.3209, 0.3290, 0.3131, 0.3060, 0.3122]

sigma_conductivity = 0.27 ## S/m

### A couple warnings we choose to ignore
warnings.filterwarnings('ignore', message='X does not have valid feature name*')

class BGNetwork(gym.Env):
    def __init__(
        self,
        n = 10,
        tmax = 4000,
        dt = 0.01,
        has_pd = True,
        predictor_loc = "predictors/bgn_rf_model.pkl",
        naumann = True,
        downsample_fs = 2000, 
        seed = None,
        fig_loc = "",
        interactive_plotting = False):

        if seed:
            self.seed(seed)
        
        self.fig_loc = os.getcwd() + "/" + fig_loc
        self.interactive_plotting = interactive_plotting

        self.env_name = 'bg_network'
        self.is_gym = True

        self.regions = ['vth', 'vsn', 'vge', 'vgi']

        self.radius = 3000 # in nanometers
        self.neuron_positions = {}
        self.neuron_dists_electrode_1 = {}
        self.neuron_dists_electrode_2 = {}
        self.recording_electrode_1_pos = np.array([0,-1500,250])
        self.recording_electrode_2_pos = np.array([0,1500,250])
        
        ## time period for which we compute PSD signals
        # None: one-sided PSD from beginning of the episode.
        self.psd_window = 2000 ## ms
        self.psd={}
        self.neuron_psd={}

        self.freq_band_dict = {
            'gamma': (30, 100),
            'beta': (12, 30),
            'alpha': (8, 12),
            'theta': (4, 8),
        }

        self.min_freq = min(self.freq_band_dict[d][0] for d in self.freq_band_dict)
        self.max_freq = max(self.freq_band_dict[d][1] for d in self.freq_band_dict)
        ## desired frequency resolution.
        self.power_freq_res = 1
        freq_bands = list(self.freq_band_dict.keys())
        self.n_reg = len(self.regions)
        self.n_bands = len(self.freq_band_dict)

        self.region_combos = list(combinations(self.regions, 2))
        self.reg_bands = [(x, y) for x in self.regions for y in freq_bands]
        self.reg_combos_bands = [
            (x, y, z) for (x, y) in self.region_combos for z in freq_bands]

        self.coherence = pd.DataFrame(data={
            'region0': [i[0] for i in self.region_combos], 
            'region1': [i[1] for i in self.region_combos], 
            'coherence':np.repeat(0, len(self.region_combos))
            })

        self.coherence_bands = pd.DataFrame(data={
            'region0': [i[0] for i in self.reg_combos_bands], 
            'region1': [i[1] for i in self.reg_combos_bands], 
            'band': [i[2] for i in self.reg_combos_bands], 
            'avg_coherence': np.repeat(0, len(self.reg_combos_bands))
            })

        self.coh_reg0_vec = [i[0] for i in self.region_combos]
        self.coh_reg1_vec = [i[1] for i in self.region_combos]
        self.coh_band_vec = self.coherence_bands['band'].to_numpy()

        self.rel_power = pd.DataFrame(data={
            'region': [i[0] for i in self.reg_bands], 
            'band': [i[1] for i in self.reg_bands], 
            'power': np.repeat(0, len(self.reg_bands)),
            'rel_power': np.repeat(0, len(self.reg_bands))
        })

        self.vgi_b_idx = np.where(np.logical_and(
            self.rel_power['band'] == "beta",
            self.rel_power['region'] == 'vgi'))[0][0]
        self.vge_b_idx = np.where(np.logical_and(
            self.rel_power['band'] == "beta",
            self.rel_power['region'] == 'vge'))[0][0]
        self.vsn_b_idx = np.where(np.logical_and(
            self.rel_power['band'] == "beta",
            self.rel_power['region'] == 'vsn'))[0][0]

        self.pow_reg_vec = self.rel_power['region'].to_numpy()

        self.pow_freq_band_low = list(map(
            lambda x: self.freq_band_dict.get(x)[0], 
            self.rel_power['band'].to_numpy()))
        self.pow_freq_band_high = list(map(
            lambda x: self.freq_band_dict.get(x)[1], 
            self.rel_power['band'].to_numpy()))

        self.coh_freq_band_low = list(map(
            lambda x: self.freq_band_dict.get(x)[0], 
            self.coherence_bands['band'].to_numpy()))
        self.coh_freq_band_high = list(map(
            lambda x: self.freq_band_dict.get(x)[1], 
            self.coherence_bands['band'].to_numpy()))


        ##  Primary adjustable constants:
        self.n = n # number of neurons in each region.
        self.tmax = tmax
        self.dt = dt #timestep (ms)
        self.has_pd = has_pd #patient has Parksinson's Disease
        self.naumann = naumann #whether to use naumann's utility functions or lu's
        self.downsample_fs = downsample_fs
        self.predictor_loc = predictor_loc

        self.time = 0 #current time of sim (after initialization).

        ## SMC input parameters:
        self.ism = 3.5 ## SMC pulse amplitude (microamp)
        self.deltasm = 5 ## SMC pulse time length
        self.smc_scale = 1785.71 ## gamma dist param for SMC pulses
        self.smc_shape = 25 ## gamma dist param for SMC pulses

        ## DBS input parameters:
        self.dbs_pulse_len = 0.3 ## Pulse length (ms)

        self.ac_high = np.asarray([300])

        ## BiPhasic DBS: https://www.nature.com/articles/s41582-020-00426-z#Sec8
        ## This range is [-3, 3] mA.
        self.action_space = spaces.Box(
            -1 * self.ac_high, self.ac_high, dtype=np.float32)
        self.observation_space = spaces.Box(
            np.zeros((math.comb(self.n_reg, 2) + self.n_reg) * self.n_bands), 
            np.ones((math.comb(self.n_reg, 2) + self.n_reg) * self.n_bands), 
            dtype=np.float32)

        self.state_names = np.concatenate((
            self.rel_power['region'] + "." + 
            self.rel_power['band'] + "." + 
            "rel_power",
            self.coherence_bands['region0'] + "." +
            self.coherence_bands['region1'] + "." +
            self.coherence_bands['band'] + "." +
            'avg_coherence'))

        ### Reward Hyperparameters:
        self.lambda_RI = 0 ## Neuron-level reward
        self.lambda_GPi_beta_power = 1 ## EEG-level reward
        self.lambda_S_GPi_sq_error = 0 ## Neuron-level reward
        self.lambda_DBS_amp = 0.001 ## Actor choice. (Other two EEG rewards are basically (-1, 0) interval)
        self.lambda_PD_prob = 1 ## EEG-level reward.

        ## Generate other parameters.
        self.initial_values()

        self.ep_len = int((self.tmax/self.dt - self.psd_dt_window) // int(np.round(self.dbs_pulse_len/self.dt, 0)))

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

        
    def step(self, action):
        ## add dbs pulse
        self.addDbsPulse(dbs_amp = action)

        ## project state variables over the pulse length.
        for i in range(self.time, int(self.time + np.round(self.dbs_pulse_len/self.dt, 0))):
            ## simple progress bar
            ## TODO: cleanup in final version.
            if i % 10000 == 0:
                print("Sim at dt step: " + str(i))

            if self.do_compute_state and i % 5000 == 0:
                gc.collect()
            if self.do_compute_state and i % 1000 == 0:
                print("Sim at dt step: " + str(i))
            #     self.end_time = time.time()
            #     print("Time since last print: %s seconds" % (self.end_time - self.start_time))
            #     self.start_time = self.end_time

            V1 = self.n_t_states['vth'][:,i]
            V2 = self.n_t_states['vsn'][:,i]
            V3 = self.n_t_states['vge'][:,i]
            V4 = self.n_t_states['vgi'][:,i]
            
            # Synapse parameter updates.
            self.n_states['S21'][1:self.n] = self.n_states['S2'][0:self.n-1]
            self.n_states['S21'][0] = self.n_states['S2'][self.n-1]
            self.n_states['S31'][0:self.n-1] = self.n_states['S3'][1:self.n]
            self.n_states['S31'][self.n-1] = self.n_states['S3'][0]
            self.n_states['S32'][2:self.n] = self.n_states['S3'][0:self.n-2]
            self.n_states['S32'][0:2] = self.n_states['S3'][self.n-2:self.n]
            
            # Membrane parameters =============================================
            self.membrane_pars = {}
            self.membrane_pars['m1'] = utils.th_minf(V1)
            self.membrane_pars['m2'] = utils.stn_minf(V2) 
            self.membrane_pars['m3'] = utils.gpe_minf(V3) 
            self.membrane_pars['m4'] = utils.gpe_minf(V4)
            self.membrane_pars['n2'] = utils.stn_ninf(V2)
            self.membrane_pars['n3'] = utils.gpe_ninf(V3)
            self.membrane_pars['n4'] = utils.gpe_ninf(V4)
            self.membrane_pars['h1'] = utils.th_hinf(V1)
            self.membrane_pars['h2'] = utils.stn_hinf(V2)
            self.membrane_pars['h3'] = utils.gpe_hinf(V3)
            self.membrane_pars['h4'] = utils.gpe_hinf(V4)
            self.membrane_pars['p1'] = utils.th_pinf(V1)
            self.membrane_pars['a2'] = utils.stn_ainf(V2) 
            self.membrane_pars['a3'] = utils.gpe_ainf(V3)
            self.membrane_pars['a4'] = utils.gpe_ainf(V4)
            self.membrane_pars['b2'] = utils.stn_binf(self.gate_states["R2"])
            self.membrane_pars['s3'] = utils.gpe_sinf(V3)
            self.membrane_pars['s4'] = utils.gpe_sinf(V4)
            self.membrane_pars['r1'] = utils.th_rinf(V1)
            self.membrane_pars['r2'] = utils.stn_rinf(V2)
            self.membrane_pars['r3'] = utils.gpe_rinf(V3)
            self.membrane_pars['r4'] = utils.gpe_rinf(V4)
            self.membrane_pars['c2'] = utils.stn_cinf(V2)

            self.membrane_pars['tn2'] = utils.stn_taun(V2)
            self.membrane_pars['tn3'] = utils.gpe_taun(V3)
            self.membrane_pars['tn4'] = utils.gpe_taun(V4)
            
            if self.naumann:
                self.membrane_pars["th1"] = utils.th_tauh_naumann(V1)
            else:
                self.membrane_pars["th1"] = utils.th_tauh_lu(V1)
            self.membrane_pars['th2'] = utils.stn_tauh(V2)
            self.membrane_pars['th3'] = utils.gpe_tauh(V3)
            self.membrane_pars['th4'] = utils.gpe_tauh(V4)
            self.membrane_pars['tr1'] = utils.th_taur(V1)
            self.membrane_pars['tr2'] = utils.stn_taur(V2)
            
            self.membrane_pars['tr3'] = 30 
            self.membrane_pars['tr4'] = 30
            self.membrane_pars['tc2'] = utils.stn_tauc(V2)
            
            ### Cell Currents =================================================
            # Thalamic cell currents
            Il1 = gl['Th'] * (V1 - El['Th'])
            Ina1 = (gna['Th'] * (self.membrane_pars['m1']**3) * 
                self.gate_states["H1"] * (V1 - Ena['Th']))
            Ik1 = (gk['Th'] * ((0.75 * (1 - self.gate_states["H1"]))**4) * 
                (V1 - Ek['Th']))
            It1  = (gt['Th'] * (self.membrane_pars['p1']**2) * 
                self.gate_states["R1"] * (V1 - Et))
            Igith = (gsyn['Igith'] * 
                (V1 - Esyn['Igith']) * (self.n_states['S4']))  ## miliVolts
            
            # STN cell currents
            Il2 = (gl['STN'] * (V2 - El['STN']))
            Ik2 = (gk['STN'] * (self.gate_states["N2"]**4) * 
                (V2 - Ek['STN']))
            Ina2 = (gna['STN'] * (self.membrane_pars['m2']**3) * 
                self.gate_states["H2"] * (V2 - Ena['STN']))
            It2 = (gt['STN'] * (self.membrane_pars['a2']**3) * 
                (self.membrane_pars['b2']**2) * (V2 - Eca['STN']))
            Ica2 = (gca['STN'] * (self.gate_states["C2"]**2) * 
                (V2 - Eca['STN']))
            Iahp2 = (gahp['STN'] * (V2 - Ek['STN']) * 
                (self.gate_states["CA2"] / (self.gate_states["CA2"] + k1['STN'])))
            Igesn = (gsyn['Igesn'] * (V2 - Esyn['Igesn']) * (
                self.n_states['S3'] + self.n_states['S31']))
            
            Iappstn = 33 - self.has_pd * 10
            
            # GPe cell currents
            Il3 = gl['GP'] * (V3 - El['GP'])
            Ik3 = gk['GP'] * (self.gate_states["N3"]**4) * (V3 - Ek['GP'])
            Ina3 = (gna['GP'] * (self.membrane_pars['m3']**3) * 
                self.gate_states["H3"] * (V3 - Ena['GP']))
            It3 = (gt['GP'] * (self.membrane_pars['a3']**3) * 
                self.gate_states["R3"] * (V3 - Eca['GP']))
            Ica3 = (gca['GP'] * (self.membrane_pars['s3']**2) * 
                (V3 - Eca['GP']))
            Iahp3 = (gahp['GP'] * (V3 - Ek['GP']) * 
                (self.gate_states["CA3"] / (self.gate_states["CA3"] + k1['GP'])))
            Isnge = (gsyn['Isnge'] * 
                (V3 - Esyn['Isnge']) * 
                (self.n_states['S2'] + self.n_states['S21']))
            Igege = (gsyn['Igege'] * 
                (V3 - Esyn['Igege']) * 
                (self.n_states['S31'] + self.n_states['S32']))

            ## NOTE: slightly differing from Naumann (21, 8, random adjustment)
            Iappgpe = 21 - 13 * self.has_pd# + self.r

            # GPi cell currents
            Il4 = gl['GP'] * (V4 - El['GP'])
            Ik4 = (gk['GP'] * (self.gate_states["N4"]**4) * 
                (V4 - Ek['GP']))
            Ina4 = (gna['GP'] * (self.membrane_pars['m4']**3) * 
                self.gate_states["H4"] * (V4 - Ena['GP']))
            It4 = (gt['GP'] * (self.membrane_pars['a4']**3) * 
                self.gate_states["R4"] * (V4 - Eca['GP']))
            Ica4 = (gca['GP'] * (self.membrane_pars['s4']**2) * 
                (V4 - Eca['GP']))
            Iahp4 = (gahp['GP'] * (V4 - Ek['GP']) * 
                (self.gate_states["CA4"] / 
                (self.gate_states["CA4"] + k1['GP'])))
            Isngi = (gsyn['Isngi'] * 
                (V4 - Esyn['Isngi']) * 
                (self.n_states['S2'] + self.n_states['S21']))
            Igegi = (gsyn['Igegi'] * 
                (V4 - Esyn['Igegi']) * 
                (self.n_states['S31'] + self.n_states['S32']))

            ## NOTE: slightly differing from between Naumann and Lu
            Iappgpi = 22 - self.has_pd * 6
            
            ### Differential Equations for cells ==============================
            ### Thalamic

            self.I_syn['vth'] = -Igith + self.I_smc[i]
            self.n_t_states['vth'][:,i+1] = (V1 + self.dt * 
                (1 / Cm * (-Il1-Ik1-Ina1-It1 + self.I_syn['vth'])))
            
            ## gate update:
            self.gate_states["H1"] = (self.gate_states["H1"] + 
                self.dt * ((self.membrane_pars['h1'] - self.gate_states["H1"]) / 
                self.membrane_pars['th1']))
            self.gate_states["R1"] = (self.gate_states["R1"] + 
                self.dt * ((self.membrane_pars['r1'] - self.gate_states["R1"]) / 
                self.membrane_pars['tr1']))
            
            ### STN
            self.I_syn['vsn'] = -Igesn+Iappstn
            self.n_t_states['vsn'][:,i+1] = (V2 + self.dt * 
                (1 / Cm * (-Il2-Ik2-Ina2-It2-Ica2-Iahp2 + self.I_syn['vsn'])))
            
            ## gate updates:
            self.gate_states["N2"] = (
                self.gate_states["N2"] + self.dt * 
                (0.75 * (
                    self.membrane_pars['n2'] - self.gate_states["N2"]
                ) / self.membrane_pars['tn2']))
            self.gate_states["H2"] = (
                self.gate_states["H2"] + self.dt * (0.75 * 
                (self.membrane_pars['h2'] - self.gate_states["H2"]) / 
                self.membrane_pars['th2']))
            self.gate_states["R2"] = (
                self.gate_states["R2"] + self.dt * (0.2 * 
                (self.membrane_pars['r2'] - self.gate_states["R2"]) / 
                self.membrane_pars['tr2']))

            self.gate_states["CA2"] = (
                self.gate_states["CA2"] + self.dt * (3.75 * 10**-5 * 
                (-Ica2-It2-kca['STN'] * self.gate_states["CA2"])))
            self.gate_states["C2"] = (
                self.gate_states["C2"] + self.dt * (0.08 * (
                self.membrane_pars['c2'] - self.gate_states["C2"]) / 
                self.membrane_pars['tc2']))

            a = np.where(np.logical_and(
                self.n_t_states['vsn'][:,i-1] < -10,  
                self.n_t_states['vsn'][:,i] > -10))
            u = np.zeros(self.n)
            u[a] = gpeak / (tau * np.exp(-1)) / self.dt
            self.n_states['S2'] = (self.n_states['S2'] + 
                self.dt * self.n_states['Z2'])
            zdot = (u - 2 / tau * self.n_states['Z2'] - 
                1 / (tau**2) * self.n_states['S2'])
            self.n_states['Z2'] = self.n_states['Z2'] + self.dt * zdot
            
            ### GPe
            self.I_syn['vge'] = -Isnge-Igege + Iappgpe
            self.n_t_states['vge'][:,i+1] = (V3 + self.dt * 
                (1 / Cm * (-Il3-Ik3-Ina3-It3-Ica3-Iahp3 + self.I_syn['vge'])))
            
            ## gate updates:
            self.gate_states["N3"] = (
                self.gate_states["N3"] + self.dt * (0.1 * (
                self.membrane_pars['n3'] - self.gate_states["N3"]
                ) / self.membrane_pars['tn3']))
            self.gate_states["H3"] = (
                self.gate_states["H3"]+ self.dt * (0.05 * (
                self.membrane_pars['h3'] - self.gate_states["H3"]
                ) / self.membrane_pars['th3']))
            self.gate_states["R3"] = (
                self.gate_states["R3"] + self.dt * (1 * (
                self.membrane_pars['r3'] - self.gate_states["R3"]
                ) / self.membrane_pars['tr3']))
            self.gate_states["CA3"] = (
                self.gate_states["CA3"] + self.dt * 
                (1 * 10**-4 * (-Ica3-It3-kca['GP'] * self.gate_states["CA3"])))
            self.n_states['S3'] = (self.n_states['S3'] + self.dt * 
                (A['GPe'] * (1 - self.n_states['S3']) * utils.Hinf(V3 - the['GPe']) - 
                B['GPe'] * self.n_states['S3']))
            
            ### GPi
            ## TODO: add Milisecond Time to history?
                
            self.I_syn['vgi'] = -Isngi - Igegi + Iappgpi ## microamps/cm^2
            self.n_t_states['vgi'][:,i+1] = (V4 + self.dt * 
                (1/Cm*(-Il4-Ik4-Ina4-It4-Ica4-Iahp4 + self.I_syn['vgi'] + self.I_dbs[i])))
            
            ## gate updates:
            self.gate_states["N4"] = (
                self.gate_states["N4"] + self.dt * 
                (0.1 * (self.membrane_pars['n4'] - self.gate_states["N4"]) / 
                self.membrane_pars['tn4']))

            self.gate_states["H4"] = (
                self.gate_states["H4"] + self.dt * 
                (0.05 * (self.membrane_pars['h4'] - self.gate_states["H4"]) / 
                self.membrane_pars['th4']))

            self.gate_states["R4"] = (
                self.gate_states["R4"] + self.dt * 
                (1 * (self.membrane_pars['r4'] - self.gate_states["R4"]) / 
                self.membrane_pars['tr4']))

            self.gate_states["CA4"] = (
                self.gate_states["CA4"] + self.dt * 
                (1*10**-4 * (-Ica4 - It4 - kca['GP'] * self.gate_states["CA4"])))
            a = np.where(np.logical_and(
                self.n_t_states['vgi'][:,i] < -10,  
                self.n_t_states['vgi'][:,i+1] > -10))
            u = np.zeros(self.n)
            u[a] = gpeak1 / (tau * np.exp(-1)) / self.dt

            self.n_states['S4'] = self.n_states['S4'] + self.dt * self.n_states['Z4']
            zdot = u - 2/tau * self.n_states['Z4'] - 1/(tau**2) * self.n_states['S4']
            self.n_states['Z4'] = self.n_states['Z4'] + self.dt * zdot

            self.SGPi_state_hist[i, :] = self.n_states['S4']
            
            ### Local Field Potential (LFP) as distance-based Ohm's law to recording electrode.
            for r in self.regions:
                ## Distance 1e-6: converts micrometers to meters.
                ## 1e-3 results in microVolts output.
                self.lfp_1_states[r][i+1] = 1/(4*np.pi*sigma_conductivity) * np.sum(self.I_syn[r] * (1/(self.neuron_dists_electrode_1[r] * 1e-6))) * 1e-3
                self.lfp_2_states[r][i+1] = 1/(4*np.pi*sigma_conductivity) * np.sum(self.I_syn[r] * (1/(self.neuron_dists_electrode_2[r] * 1e-6))) * 1e-3

                self.lfp_diff_states[r][i+1] = self.lfp_2_states[r][i+1] - self.lfp_1_states[r][i+1]

                ### (Full sampling frequency) EEG states
                self.eeg_states[r][i+1] = self.lfp_1_states[r][i+1]

            ## update downsample counter.
            if i % self.ds_interval == 0:
                self.i_ds += 1

        ## update time parameter: 
        self.time += int(np.round(self.dbs_pulse_len / self.dt, 0))

        ## compute the state variables and append to history
        if self.do_compute_state:
            self.compute_state(action = action)

        ## initiatize state computation on the next step.
        if self.time >= self.psd_dt_window and not self.do_compute_state:
            self.do_compute_state = True

        ## flag if done.
        self.done = self.time >= self.t.shape[0]-1
        

        ### END STEP() ========================================================
        return (
            np.array(self.state, dtype=np.float32), 
            self.reward, self.done, {}
            )

    def compute_state(self, action):
        ### Filter, downsample signal.
        ### TODO: Try Decimate Approach.
        for r in self.regions:
            self.lfp_bandpass_signal_cheby[r] = sp_sg.filtfilt(
                self.cheb_b, self.cheb_a, 
                self.lfp_1_states[r][(self.time - self.psd_dt_window):self.time])
        ### Butterworth version.
        #lfp_bandpass_signal_butter = sp_sg.filtfilt(self.butter_b, self.butter_a, self.lfp_1_states[r][(self.time - self.psd_dt_window):self.time])
        
        ### TODO: just do the neuron level power beta for sake of actual "biomarker" in the reward function?
        self.freq_band_rel_power(get_freqs = (self.psd_freq_res is None))
        self.freq_band_coherence(get_freqs = (self.coh_freq_res is None))

        ### append to state vector
        self.state = np.concatenate((
            self.rel_power_vec,
            self.coherence_vec
        ))

        ### Compute predicted probability of PD.
        if self.pd_predictor is not None:
            pd_prob = self.pd_predictor.predict_proba(X=self.state.reshape(1, -1))[0][1]
            
        else:
            pd_prob = np.nan


        ## append to history.
        self.history[self.state_hist_idx, 0] = action

        ## reward functions:
        ##   (EI, S_GPi Sq Dev, GPi Power, DBS current)

        ## Reliability Index
        self.history[self.state_hist_idx, 1] = self.calculateEI(
            beg_ignore_dt = self.time - self.psd_dt_window,
            end_ignore_dt = self.t.shape[0] - self.time
            )['RI']

        ## deviation from 'normal state' synaptic variable SGi
        self.history[self.state_hist_idx, 2] = np.sum(
            -1*(self.n_states['S4'] - bar_S)**2)

        ## Beta Band Rel Power "Error" from Healthy:
        self.history[self.state_hist_idx, 3] = -1* (
            np.clip(self.rel_power_vec[self.vgi_b_idx] - healthy_vgi_b_mean, 0, 1) + 
            np.clip(self.rel_power_vec[self.vge_b_idx] - healthy_vge_b_mean, 0, 1) + 
            np.clip(self.rel_power_vec[self.vsn_b_idx] - healthy_vsn_b_mean, 0, 1))
        ## Predicted proba addition:
        self.history[self.state_hist_idx, 4]= -pd_prob

        self.history[self.state_hist_idx, 5] = -np.abs(action)

        

        ### Overall reward
        self.reward = (
            self.lambda_RI * self.history[self.state_hist_idx, 1] + 
            self.lambda_S_GPi_sq_error * self.history[self.state_hist_idx, 2] +
            self.lambda_GPi_beta_power * self.history[self.state_hist_idx, 3] +
            self.lambda_DBS_amp * self.history[self.state_hist_idx, 5])
        
        if self.pd_predictor is not None:
            self.reward += self.lambda_PD_prob * self.history[self.state_hist_idx, 4]

        self.history[self.state_hist_idx, 6] = self.reward

        ## State space additions.
        self.history[self.state_hist_idx, 7:] = self.state

        self.state_hist_idx += 1
    
    def freq_band_rel_power(self, get_freqs = False):
        ## compute raw PSD
        self.compute_psd(get_freqs = get_freqs)
        ## integrate (and compute relative power)

        ## compute power:
        self.power_vec = list(map(
            lambda i: simps(self.psd[self.pow_reg_vec[i]][
                np.logical_and(
                    ## TODO: check if intervals are closed / open and update tex file accordingly.
                        self.psd_freqs >= self.pow_freq_band_low[i],
                        self.psd_freqs <= self.pow_freq_band_high[i])
                        ], dx=self.psd_freq_res), 
            list(range(len(self.pow_freq_band_high)))))

        ## compute relative power:
        self.rel_pow_denom = np.repeat(list(map(
            lambda r: simps(self.psd[r][
                        np.logical_and(
                            self.psd_freqs >= self.min_freq, 
                            self.psd_freqs <= self.max_freq
                            )
                    ], dx=self.psd_freq_res),
            self.regions
        )), self.n_reg)

        self.rel_power_vec = self.power_vec / self.rel_pow_denom
        
        
    
    def freq_band_coherence(self, get_freqs = False):
        ## compute raw coherences
        self.compute_coherence(get_freqs = get_freqs)

        ## integrate (and average) over bins
        self.coherence_vec = list(map(
            lambda i: simps(self.coherence_array[i,:][
                    np.logical_and(
                        self.coh_freqs >= self.coh_freq_band_low[i],
                        self.coh_freqs <= self.coh_freq_band_high[i])
                        ], dx=self.coh_freq_res) / (
                            self.coh_freq_res * np.sum(np.logical_and(
                            self.coh_freqs >= self.coh_freq_band_low[i],
                            self.coh_freqs <= self.coh_freq_band_high[i],))
                        ), 
            list(range(len(self.coh_freq_band_low)))))
        
    def compute_psd(self, get_freqs = False):
        for r in self.regions:
            self.psd[r] = sp_sg.welch(self.lfp_bandpass_signal_cheby[r], self.fs, 
                nperseg = self.nperseg ### dt multiplier
            )[1][0:int(round(self.max_freq/(self.fs/self.nperseg) + 1,0))]

        if get_freqs:
            self.psd_freqs = sp_sg.welch(
                self.lfp_bandpass_signal_cheby[self.regions[0]],
                self.fs, 
                nperseg = self.nperseg ### dt multiplier
            )[0][0:int(round(self.max_freq/(self.fs/self.nperseg) + 1,0))]
            self.psd_freq_res = self.psd_freqs[1]-self.psd_freqs[0]

    def compute_neuron_psd(self, neuron_idx = 0):
        for r in self.regions:
            self.neuron_psd_freqs, self.neuron_psd[r] = sp_sg.welch(
                self.n_t_states[r][neuron_idx][self.time - self.psd_dt_window:self.time + 1],
                self.fs, 
                nperseg = self.nperseg ### dt multiplier
            )
        self.neuron_psd_freq_res = self.neuron_psd_freqs[1]-self.neuron_psd_freqs[0]
    
    def compute_coherence(self, get_freqs = False):
        
        self.coherence_array = np.repeat(np.asarray(list(map(
            lambda i: sp_sg.coherence(
                self.lfp_bandpass_signal_cheby[self.coh_reg0_vec[i]],
                self.lfp_bandpass_signal_cheby[self.coh_reg1_vec[i]],
                #self.eeg_states[self.coh_reg0_vec[i]][self.i_ds-1 - self.psd_ds_window:self.psd_ds_window + 1],
                #self.eeg_states[self.coh_reg1_vec[i]][self.i_ds-1 - self.psd_ds_window:self.psd_ds_window + 1],
                self.fs,
                nperseg = self.nperseg)[1][0:int(round(self.max_freq/(self.fs/self.nperseg) + 1,0))], 
            list(range(len(self.coh_reg0_vec)))))), self.n_bands, axis=0)
        
        if get_freqs:
            self.coh_freqs = (sp_sg.coherence(
                self.lfp_bandpass_signal_cheby[self.regions[0]],
                self.lfp_bandpass_signal_cheby[self.regions[1]],
                self.fs,
                nperseg = self.nperseg)[0]
            )[0:int(round(self.max_freq/(self.fs/self.nperseg) + 1,0))]
            self.coh_freq_res = self.coh_freqs[1]-self.coh_freqs[0]


    
    def plot_power_bands(
        self,
        plot_name_tag = "",
        width = 10, height = 4, save = True):
        
        ## Update data frame
        self.rel_power['power'] = self.power_vec

        fig = plt.figure(figsize = (width, height), dpi=100)

        sns.barplot(
            data = self.rel_power, 
            x = "region", y = "power", hue = "band",
            hue_order = ['theta', 'alpha', 'beta', 'gamma'])

        ## axis labels and legend
        plt.ylabel('Integrated PSD [V**2/Hz]')

        if save:
            fig.savefig(self.fig_loc + plot_name_tag + 'total_power_bands.png')

        plt.close(fig)

    
    def plot_rel_power_bands(
        self,
        plot_name_tag = "",
        width = 10, height = 4, save = True):
        ## Update data frame

        self.rel_power['rel_power'] = self.rel_power_vec
        
        fig = plt.figure(figsize = (width, height), dpi=100)

        sns.barplot(
            data = self.rel_power, 
            x = "region", y = "rel_power", hue = "band",
            hue_order = ['theta', 'alpha', 'beta', 'gamma'])

        if save:
            fig.savefig(self.fig_loc + plot_name_tag + 'rel_power_bands.png')

        plt.close(fig)
        
    
    def plot_coherence_bands(
        self,
        plot_name_tag = "",
        facet_width = 3, facet_height = 3, save = True):
        ## Update data frame
        self.coherence_bands['avg_coherence'] = self.coherence_vec

        coherence_bands_same_reg_df = pd.DataFrame(
            data={
                'region0': [i[0] for i in self.reg_bands],
                'region1': [i[0] for i in self.reg_bands],
                'band': [i[1] for i in self.reg_bands],
                'avg_coherence': 1})

        coherence_rev_df = pd.DataFrame(
            data={
                'region0': self.coherence_bands['region1'],
                'region1': self.coherence_bands['region0'],
                'band': self.coherence_bands['band'],
                'avg_coherence': self.coherence_bands['avg_coherence']
            }
        )

        coherence_bands_plot_df = pd.concat((self.coherence_bands, coherence_rev_df, coherence_bands_same_reg_df), axis=0)

        ## heatmap plotting function for faceted heatmap.
        def draw_heatmap(*args, **kwargs):
            data = kwargs.pop('data')
            d = data.pivot(index=args[1], columns=args[0], values=args[2])
            if data['band'].unique()[0] == 'gamma' and data['band'].unique().shape == (1,):
                kwargs['cbar'] = True
            sns.heatmap(d, **kwargs)

        fig = sns.FacetGrid(
            coherence_bands_plot_df, 
            col='band', col_order = ['theta', 'alpha', 'beta', 'gamma'],
            height = facet_height,
            aspect = facet_width / facet_height)
            
        fig.map_dataframe(
            draw_heatmap, 'region0', 'region1', 'avg_coherence', 
            vmin=0, vmax=1, square = True, cbar = False, cmap="YlGnBu",)


        if save:
            fig.savefig(self.fig_loc + plot_name_tag + 'coherence_heatmap.png')

        plt.close(fig.fig)
    
    def plot_eeg_psd(
        self,
        plot_name_tag = "",
        width = 10, height = 4, save = True):

        fig = plt.figure(figsize = (width, height), dpi=100)

        for r in self.regions:
            plt.semilogy(self.psd_freqs, self.psd[r], label=r)
            plt.xlim([1, 100])
            plt.ylim([1e-5, 1])
        
        ## axis labels and legend
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
        plt.legend(loc='best')

        if save:
            fig.savefig(self.fig_loc + plot_name_tag + 'eeg_psd.png')

        plt.close(fig)

    
    def plot_neuron_psd(
        self, neuron_idx = 0,
        plot_name_tag = "",
        width = 10, height = 4, save = True):

        fig = plt.figure(figsize = (width, height), dpi=100)
        for r in self.regions:
            ### compute neuron PSD
            self.compute_neuron_psd(neuron_idx = neuron_idx)
            plt.semilogy(self.neuron_psd_freqs, self.neuron_psd[r], label=r)
            plt.xlim([1, 100])
            plt.ylim([0.001, 100])
        
        ## axis labels and legend
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
        plt.legend(loc='best')

        if save:
            fig.savefig(self.fig_loc + plot_name_tag + 'neuron_psd.png')

        plt.close(fig)

    def plot_thalamus(
        self, time_plot=50000,
        plot_name_tag = "",
        width = 10, height = 4, save = True):

        fig = plt.figure(figsize = (width, height), dpi=100)
        plt.plot(self.t[0:time_plot], self.n_t_states['vth'][0,0:time_plot], label="Thalamus")
        plt.plot(self.t[0:time_plot], self.I_smc[0:time_plot], color='red', label='SMC Input')

        ## axis labels and legend
        plt.xlabel('Time (ms)')
        plt.ylabel('Voltage (mV)')

        if save:
            fig.savefig(self.fig_loc + plot_name_tag + 'thalamus_potential.png')

        plt.close(fig)

    def plot_region_potentials(
        self, dt_range = None, 
        neuron_idx = 0,
        smc_color = "red", dbs_color = "purple", 
        plot_name_tag = "",
        width = 10, height = 6, save = True):
        
        ### set time range if not specified
        if dt_range is None:
            dt_range = [self.psd_dt_window, self.time+1]

        t_range = self.t[dt_range[0]:dt_range[1]]
        dt_range = np.arange(dt_range[0], dt_range[1])

        ### Constuct subplots of each region potentials, SMC input, DBS input
        fig, axs = plt.subplots(2, 2)
        axs[0,0].plot(t_range, self.n_t_states['vth'][neuron_idx, dt_range])
        ## SMC impulses:
        axs[0,0].plot(
            t_range, 
            self.I_smc[dt_range],
            color=smc_color,
            label = 'SMC Input')
        axs[0,0].set_title('Thalamic')

        axs[0,1].plot(t_range, self.n_t_states['vsn'][neuron_idx, dt_range])
        axs[0,1].set_title('STN')

        axs[1,0].plot(t_range, self.n_t_states['vge'][neuron_idx, dt_range])
        axs[1,0].set_title('GPe')

        axs[1,1].plot(t_range, self.n_t_states['vgi'][neuron_idx, dt_range])
        ## DBS impulses:
        axs[1,1].plot(
            t_range, 
            self.I_dbs[dt_range],
            color=dbs_color,
            label = 'DBS Input')
        axs[1,1].set_title('GPi')

        for ax in axs.flat:
            ax.set(xlabel = 'Time (msec)', ylabel = 'Vm (mV)')

        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axs.flat:
            ax.label_outer()  

        ## axis labels and legend
        plt.xlabel('Time (ms)')
        plt.ylabel('Voltage (mV)')

        fig.set_size_inches(w = width, h = height, forward=True)
        if save:
            fig.savefig(self.fig_loc + plot_name_tag + 'region_potentials.png', dpi=100)
        else:
            plt.show()

        plt.close(fig)



    def plot_lfp(
        self, dt_range = None, 
        plot_name_tag = "",
        width = 10, height = 6, save = True):
        
        ### set time range if not specified
        if dt_range is None:
            dt_range = [self.psd_dt_window, self.time+1]

        t_range = self.t[dt_range[0]:dt_range[1]]
        dt_range = np.arange(dt_range[0], dt_range[1])

        ### Constuct subplots of each region potentials, SMC input, DBS input
        fig, axs = plt.subplots(2, 2)
        axs[0,0].plot(t_range, self.lfp_1_states['vth'][dt_range] - np.mean(self.lfp_1_states['vth'][dt_range]), label="Electrode 1")
        axs[0,0].plot(t_range, self.lfp_2_states['vth'][dt_range] - np.mean(self.lfp_2_states['vth'][dt_range]), label="Electrode 2")
        axs[0,0].plot(t_range, self.lfp_diff_states['vth'][dt_range] - np.mean(self.lfp_diff_states['vth'][dt_range]), label="Diff (1-2)")
        axs[0,0].set_title('Thalamic')

        axs[0,1].plot(t_range, self.lfp_1_states['vsn'][dt_range] - np.mean(self.lfp_1_states['vsn'][dt_range]), label="Electrode 1")
        axs[0,1].plot(t_range, self.lfp_2_states['vsn'][dt_range] - np.mean(self.lfp_2_states['vsn'][dt_range]), label="Electrode 2")
        axs[0,1].plot(t_range, self.lfp_diff_states['vsn'][dt_range] - np.mean(self.lfp_diff_states['vsn'][dt_range]), label="Diff (1-2)")
        axs[0,1].set_title('STN')

        axs[1,0].plot(t_range, self.lfp_1_states['vge'][dt_range] - np.mean(self.lfp_1_states['vge'][dt_range]), label="Electrode 1")
        axs[1,0].plot(t_range, self.lfp_2_states['vge'][dt_range] - np.mean(self.lfp_2_states['vge'][dt_range]), label="Electrode 2")
        axs[1,0].plot(t_range, self.lfp_diff_states['vge'][dt_range] - np.mean(self.lfp_diff_states['vge'][dt_range]), label="Diff (1-2)")
        axs[1,0].set_title('GPe')

        axs[1,1].plot(t_range, self.lfp_1_states['vgi'][dt_range] - np.mean(self.lfp_1_states['vgi'][dt_range]), label="Electrode 1")
        axs[1,1].plot(t_range, self.lfp_2_states['vgi'][dt_range] - np.mean(self.lfp_2_states['vgi'][dt_range]), label="Electrode 2")
        axs[1,1].plot(t_range, self.lfp_diff_states['vgi'][dt_range] - np.mean(self.lfp_diff_states['vgi'][dt_range]), label="Diff (1-2)")
        # ## DBS impulses:
        # axs[1,1].plot(
        #     t_range, 
        #     self.I_dbs[dt_range],
        #     color=dbs_color,
        #     label = 'DBS Input')
        axs[1,1].set_title('GPi')

        axs[0,0].set(ylabel = 'Baselined Voltage (mV)')
        axs[1,0].set(ylabel = 'Baselined Voltage (mV)')

        axs[1,0].set(xlabel = 'Time (ms)')
        axs[1,1].set(xlabel = 'Time (ms)')

        ## axis labels and legend
        plt.xlabel('Time (ms)')
        plt.ylabel('Baselined Voltage (mV)')
        plt.legend(loc='best')


        fig.set_size_inches(w = width, h = height, forward=True)
        if save:
            fig.savefig(self.fig_loc + plot_name_tag + 'region_lfps.png', dpi=100)

        plt.close(fig)


    def reset(self, seed=None):
        '''
        Resets environment state variables.
        '''
        if seed:
            self.seed(seed)
        ## re-initialize.
        self.initial_values()

        ## load predictor model, if specified
        if self.predictor_loc != "":
            pred_full_loc=[file for file in glob.iglob(
                os.getcwd()+ "/**/"+self.predictor_loc, recursive=True)]
            assert(len(pred_full_loc) == 1) ## unambiguous file location?
            with open(pred_full_loc[0], 'rb') as f:
                self.pd_predictor = pkl.load(f)
        else:
            self.pd_predictor = None


        ## plot options
        if self.interactive_plotting:
            plt.ion()
        else:
            plt.ioff()
        
        self.state = None
        self.reward = None
        self.done = None

        self.do_compute_state = False
        self.state_hist_idx = 0

        ## initialize SGPi history store
        self.SGPi_state_hist = np.zeros((self.t.shape[0], self.n))
        self.lfp_bandpass_signal_cheby = {}
        

        ## run sim until PSD state values may be computed given window setting.
        while self.time < self.psd_dt_window:
            self.step(action = 0)
        
        ## Note: time resolution of information storage is not DT.
        self.df_init_dict = {
            "Time": range(self.time, self.t.shape[0], int(np.round(self.dbs_pulse_len/self.dt, 0))),
            "Action": None,
            "Reward.RI": None,
            "Reward.S_GPi_sq_error": None,
            "Reward.GPi_Beta": None,
            "Reward.PD_Pred_Prob": None,
            "Reward.DBS_Amp": None,
            "Reward_Overall": None,
            }

        self.ep_len = len(self.df_init_dict['Time'])

        self.df_init_dict.update({s: None for s in self.state_names})

        ## initialize action, reward, state, history.
        self.history = np.zeros((len(self.df_init_dict['Time']), len(self.df_init_dict)-1))

        ## compute initial state
        self.compute_state(action=0)

        return self.state

        

    def initial_values(self,):
        '''
        Repopulates initial values in case adjustable parameters have been updated.
        '''
        self.t = np.arange(
            start=0, 
            stop=int(self.tmax), 
            step=self.dt)


        ## add enough steps to include full length of last DBS pulse.
        adder = int(np.round(self.dbs_pulse_len / self.dt, 0) - (self.tmax / self.dt) % np.round(
            self.dbs_pulse_len / self.dt, 0) + 1)

        for c in range(adder):
            self.t = np.append(self.t, [self.t[-1]+self.dt])
        
        self.start_time = time.time()
        self.time = 0
        ## downsample index
        self.i_ds = 0
        self.smc_pulse = self.ism * np.ones(shape = int(self.deltasm / self.dt)) ## still in microAmps

        ### 1 / self.dt is the number of samples in a milisecond. fs is the number of samples in a second.
        self.fs = int(1000 / self.dt)

        ## initialize neuron positions
        self.generateNeuronPositionsAndDist()

        ## PSD dt window
        self.psd_dt_window = int(self.psd_window / self.dt)
        self.downsample_dt = self.dt * (self.fs / self.downsample_fs)
        self.psd_ds_window = int(self.psd_window / self.downsample_dt)

        ## will populate this on init.
        self.psd_freq_res, self.coh_freq_res = None, None
        ## set nperseg based on desired frequency resolution
        self.nperseg = int(self.fs / self.power_freq_res)

        ## confirm nperseg is valid
        if self.nperseg > 0.5*self.psd_dt_window:
            raise ValueError('nperseg needed for freq res is too large. Decrease freq res or increase window size.')

        ## Generate Filter constants.
        self.butter_b, self.butter_a = sp_sg.butter(N=3, Wn=100, btype='lowpass', fs=self.fs)
        self.cheb_b, self.cheb_a = sp_sg.cheby1(N=5, rp=0.5, Wn=self.max_freq/(0.5*self.fs))
        
        # Sensorimotor cortex input to thalamic cells:
        self.createSMC()

        # initialize empty DBS vector:
        self.I_dbs = np.zeros(shape = len(self.t))

        ## Other initial matrices:
        other_state_names = ['S2', 'S21', 'S3', 'S31', 'S32', 'S4', 'Z2', 'Z4']

        self.ds_interval = int(np.round(self.fs / self.downsample_fs, 0))
        self.dbs_interval = int(np.round(self.dbs_pulse_len/self.dt, 0))

        self.eeg_states_ds_len = int(len(self.t) / self.ds_interval)
        
        (self.n_t_states, self.n_states, self.I_syn,
        self.eeg_states, self.eeg_states_ds,
        self.lfp_1_states, self.lfp_2_states,
        self.lfp_diff_states, self.lfp_states_df,
        self.neuron_dists_to_electrode) = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}

        for r in self.regions:
            self.n_t_states[r] = np.zeros(shape = (self.n, len(self.t)))
            self.eeg_states[r] = np.zeros(shape = len(self.t))
            self.lfp_1_states[r] = np.zeros(shape = len(self.t))
            self.lfp_2_states[r] = np.zeros(shape = len(self.t))
            self.lfp_diff_states[r] = np.zeros(shape = len(self.t))
            self.eeg_states_ds[r] = np.zeros(shape = self.eeg_states_ds_len)

            self.I_syn[r] = np.zeros(self.n)
    
            ## initial membrane voltages for all neurons
            self.n_t_states[r][:,0] = -62 + np.random.normal(size = self.n)*5
            self.eeg_states[r][0] = np.mean(self.n_t_states[r][:,0])
            self.eeg_states_ds[r][0] = self.eeg_states[r][0]
            
        for k in other_state_names:
            self.n_states[k] = np.zeros(self.n)


        self.r = np.random.normal(size = self.n)*2

        ## Gate variable initials:
        self.gate_states = {}
        self.gate_states["N2"] = utils.stn_ninf(self.n_t_states['vsn'][:,0])
        self.gate_states["N3"] = utils.gpe_ninf(self.n_t_states['vge'][:,0])
        self.gate_states["N4"] = utils.gpe_ninf(self.n_t_states['vgi'][:,0])
        self.gate_states["H1"] = utils.th_hinf(self.n_t_states['vth'][:,0])
        self.gate_states["H2"] = utils.stn_hinf(self.n_t_states['vsn'][:,0])
        self.gate_states["H3"] = utils.gpe_hinf(self.n_t_states['vge'][:,0])
        self.gate_states["H4"] = utils.gpe_hinf(self.n_t_states['vgi'][:,0])
        self.gate_states["R1"] = utils.th_rinf(self.n_t_states['vth'][:,0])
        self.gate_states["R2"] = utils.stn_rinf(self.n_t_states['vsn'][:,0])
        self.gate_states["R3"] = utils.gpe_rinf(self.n_t_states['vge'][:,0])
        self.gate_states["R4"] = utils.gpe_rinf(self.n_t_states['vgi'][:,0])
        
        self.gate_states["CA2"] = 0.1
        self.gate_states["CA3"] = self.gate_states["CA2"]
        self.gate_states["CA4"] = self.gate_states["CA2"]
        self.gate_states["C2"] = utils.stn_cinf(self.n_t_states['vsn'][:,0])

    ### Error Index Calc: =========================================================
    
    def calculateEI(
        self, 
        beg_ignore_dt = 0,
        end_ignore_dt = 0
        ):
        '''
        Calculates the Error Index (EI)

        Input:
        t - time vector (msec)
        vth - Array with membrane potentials of each thalamic cell
        timespike - Time of each SMC input pulse
        tmax - maximum time taken into consideration for calculation

        Output:
        er - Error index

        '''

        N = self.n
        T = self.time
        n_e = 0
        n_e2 = 0

        smc_spike_idx = np.where(
            np.logical_and(
                np.greater_equal(self.timespike, self.t[beg_ignore_dt]), 
                np.greater_equal(self.t[self.t.shape[0] - end_ignore_dt], self.timespike) 
                ))[0]

        smc_spikes = {}
        i = 0
        for f in smc_spike_idx:
            smc_spikes[str(i)] = self.timespike[f]
            i +=1

        for n in range(N):            
            ## Tabulate the number of voltage threshold crosses:
            th_activations = []
            for j in range(1,T):
                if (
                    self.n_t_states['vth'][n,j-1] < -40 and 
                    self.n_t_states['vth'][n,j]   > -40 and
                    self.t[j] >= smc_spikes['0']):
                    th_activations.append(self.t[j])

            ## TH neuron must have exactly one activation inside the 25ms time-window
            smc_th_activations = {}
            n_a = 0
            for s in range(len(smc_spikes)):
                after_spike = np.greater_equal(th_activations, smc_spikes[str(s)])
                before_post_spike = np.greater_equal(
                    smc_spikes[str(s)] + 25, th_activations)
                if s < len(smc_spikes) - 1:
                    before_next_spike = np.less(th_activations, smc_spikes[str(s+1)])
                else:
                    before_next_spike = np.ones(len(th_activations)).astype('bool')

                smc_th_activations[str(s)] = np.where(np.logical_and.reduce(
                    (after_spike, before_post_spike, before_next_spike)))[0]
                
                ## tabulate non-spurious spikes
                n_a += np.where(np.logical_and.reduce(
                    (after_spike, before_post_spike, before_next_spike)))[0].shape[0]

            ## faulty activation tabulation:
            for s in smc_th_activations:
                ## miss
                if smc_th_activations[s].shape[0] == 0:
                    n_e += 1
                    n_e2 += 1
                ## burst
                elif smc_th_activations[s].shape[0] > 1:
                    n_e += 1
                    n_e2 += 1

            ## spurious spikes:
            n_e += len(th_activations) - n_a
            

        n_smc = len(smc_spikes) * N
        RI = 1 - n_e / n_smc
        return {'n_e': n_e, 'n_smc': n_smc, 'EI': n_e / n_smc, 'RI': RI}


    def sampleSpikeIndex(self, i,):
        '''27/
        Draws next smc spike from inverse gamma and adds to time index (i).
        i: current time index
        A: gamma shape param
        B: gamma scale param
        '''
        ## draw spike waiting time from inverse gamma in ms.
        ipi = st.invgamma.rvs(self.smc_shape, scale=self.smc_scale)
        ## convert to time index in sim:
        i = i + np.round(ipi / self.dt)
        return int(i)

    def createSMC(self,):
        '''
        Non-regular incoming signal from SMC to thalamic neurons is modeled as 
        a series of monophasic current pulses with amplitude 3.5 microA/cm2 and 
        duration 5ms. Frequency of pulses is stochastic (gamma dist). 

        Primarily generates self.I_SMC and self.timespike attributes containing 
        the SMC stimulation time series and the spike start indices, respectively.
        '''
        self.I_smc = np.zeros(shape = len(self.t))

        ## store timespike times:
        self.timespike = []

        ## first spike:
        i = self.sampleSpikeIndex(i=0,)

        while i < len(self.t):
            self.timespike.append(self.t[i])
            end_idx = int(np.min((self.tmax / self.dt,
                i + self.deltasm / self.dt)))
            self.I_smc[i:end_idx] = self.smc_pulse[0:end_idx-i]  ## microAmps
            i = self.sampleSpikeIndex(i)


    def addDbsPulse(self, dbs_amp):
        '''
        Creates DBS train of frequency f, of length tmax, with time step dt.
        '''
        dbs_pulse = dbs_amp * np.ones(int(np.round(self.dbs_pulse_len / self.dt, 0)))
        self.I_dbs[self.time:int(np.round(self.time + self.dbs_pulse_len/self.dt, 0))] = dbs_pulse
    

    def generateNeuronPositionsAndDist(self,):
        '''
        Randomly generate neuron positions in a sphere of size self.radius (nanometers).
        '''
        sphere_space = space.RandomStructure(boundary=space.Sphere(self.radius))
        ### Interference with electrode:
        electrode_interference = True
        for r in self.regions:
            self.neuron_positions[r] = sphere_space.generate_positions(self.n)
            while electrode_interference:
                interfere_array = np.logical_and(
                    ## x-axis check.
                    np.abs(self.neuron_positions[r][0])<500,
                    np.logical_and(
                        ## y-axis check
                        self.neuron_positions[r][1]>-1500,
                        ## z-axis check
                        np.abs(self.neuron_positions[r][2])<250),
                    )
                n_interfere = np.sum(interfere_array)
                if np.sum(interfere_array) > 0:
                    ## TODO: positional / logical statement.
                    self.neuron_positions[r][:,interfere_array] = sphere_space.generate_positions(n_interfere)
                else:
                    electrode_interference = False 

        ### Generate Distances to electron position
        ## random distances to recording electrodes (unit of measure: mm) See: https://www.frontiersin.org/articles/10.3389/fnins.2020.00166/full
        for r in self.regions:
            self.neuron_dists_electrode_1[r] = np.sqrt(
                np.sum((self.neuron_positions[r] - np.expand_dims(self.recording_electrode_1_pos, axis=1))**2, axis=0)
                )
            self.neuron_dists_electrode_2[r] = np.sqrt(
                np.sum((self.neuron_positions[r] - np.expand_dims(self.recording_electrode_2_pos, axis=1))**2, axis=0)
                )

