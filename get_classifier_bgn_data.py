import os, sys
import numpy as np
import pandas as pd
import gym
from scipy.stats import norm

from env import register_envs

## Register environments and initialize PD, non-PD networks.
register_envs()

### Function to run 
def runSimulationSaveData(
    bg_network,
    stimulate = True,
    time_past_warmup = 100000,
    plot_tag = '',
    create_plots=True):

    ### Runs simulation and computes/collects state history variables for time_past_warmup // step_time steps.
    bg_network.reset()

    ## First step then plot state variables:
    if stimulate:
        action = np.clip(
            norm.rvs(scale = bg_network.ac_high[0]/2), ## scale keyword is the standard deviation.
            -bg_network.ac_high[0], 
            bg_network.ac_high[0])
    else:
        action = 0
        
    obs, rew, done, info = bg_network.step(action)

    if create_plots:
        bg_network.plot_lfp(plot_name_tag = plot_tag, dt_range = [20000, 50000])
        bg_network.plot_power_bands(plot_name_tag = plot_tag)
        bg_network.plot_rel_power_bands(plot_name_tag = plot_tag)
        bg_network.plot_coherence_bands(plot_name_tag = plot_tag)
        bg_network.plot_eeg_psd(plot_name_tag = plot_tag)
        bg_network.plot_neuron_psd(neuron_idx=0, plot_name_tag = plot_tag)
        bg_network.plot_thalamus(plot_name_tag = plot_tag)
        print("DONE PLOTTING")
        
    while bg_network.time <= int(bg_network.psd_dt_window+time_past_warmup):
        if stimulate:
            action = np.clip(
                norm.rvs(scale = bg_network.ac_high[0]/2), ## scale keyword is the standard deviation.
                -bg_network.ac_high[0], 
                bg_network.ac_high[0])
        else:
            action = 0

        obs, rew, done, info = bg_network.step(action)

    ### sample from the model
    rowsums = np.sum(bg_network.history, axis=1)
    last_val_idx = min(np.where(rowsums==0)[0])

    ### Save the state history dataframe.
    time_vec = np.arange(0, last_val_idx, step=1) * bg_network.dbs_pulse_len
    state_history = bg_network.history[0:last_val_idx,6:]
    state_history = pd.DataFrame(
        data=state_history, 
        #index=data[1:,0],    
        columns=bg_network.state_names)
    state_history['time_ms'] = time_vec
    state_history.to_csv("sim_output/state_history_hasPD" + str(bg_network.has_pd) + ".csv")
    
    plot_tag = plot_tag + str(time_past_warmup//1000) + "k_time_"

    ### Plots after the additional run time:
    if create_plots:
        bg_network.plot_power_bands(plot_name_tag = plot_tag)
        bg_network.plot_rel_power_bands(plot_name_tag = plot_tag)
        bg_network.plot_coherence_bands(plot_name_tag = plot_tag)
        bg_network.plot_eeg_psd(plot_name_tag = plot_tag)
        bg_network.plot_neuron_psd(neuron_idx=0, plot_name_tag = plot_tag)
        bg_network.plot_thalamus(plot_name_tag = plot_tag)
        print("DONE PLOTTING")
        
    if stimulate:
        print("plotting model validation with random DBS input")
        bg_network.plot_region_potentials(plot_name_tag = plot_tag)
        ## Zoom-in to see that DBS pulse is constant 
        bg_network.plot_region_potentials(
            dt_range=[bg_network.psd_dt_window, bg_network.psd_dt_window + time_past_warmup], 
            plot_name_tag = plot_tag)

if __name__ == "__main__":
    TMAX=5000 ## how long (including warm-up of 2,000 ms, which is the spectral feature computation window size, to run an episode?)
    NEURONS=10
    SEED=1
    DT=0.01
    STIMULATE=False
    CREATE_PLOTS=True
    MILLISEC_OF_DATA_COLLECT = 1000
    TIME_PAST_WARMUP = int(MILLISEC_OF_DATA_COLLECT * 1/DT)

    ## Make environment
    pd_bg_network = gym.make(
        'bg-network-v0',
        n=NEURONS, 
        tmax=TMAX, 
        dt=DT, 
        has_pd=True,
        predictor_loc="",
        seed=SEED,
        fig_loc = "simulation_document/figures/")

    no_pd_bg_network = gym.make(
        'bg-network-v0',
        n=NEURONS, 
        tmax=TMAX, 
        dt=DT, 
        has_pd=False,
        predictor_loc="",
        seed=SEED,
        fig_loc = "simulation_document/figures/")


    print("RUNNING PD MODEL (no Stimulation)")
    runSimulationSaveData(
        pd_bg_network,
        stimulate=STIMULATE,
        time_past_warmup=TIME_PAST_WARMUP,
        plot_tag="pd_",
        create_plots=CREATE_PLOTS)

    print("RUNNING NO PD MODEL (no Stimulation)")
    runSimulationSaveData(
        no_pd_bg_network,
        stimulate=STIMULATE,
        time_past_warmup=TIME_PAST_WARMUP,
        plot_tag="nopd_",
        create_plots=CREATE_PLOTS)
