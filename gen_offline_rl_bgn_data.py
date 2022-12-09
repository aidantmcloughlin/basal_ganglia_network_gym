import os, sys
import numpy as np
import pandas as pd
import gym
from scipy.stats import norm
from matplotlib import pyplot as plt

from rl_modules.envs import register_envs

## Register environments and initialize PD, non-PD networks.
register_envs()

### Function to create data sets
def genOfflinePeriodicData(
    bg_network,
    time_of_stim = 20000,
    no_stim_time = 20000,
    single_stim_period = 600, ## dt
    stim_sd_frac_of_max_stim = 0.5, ## what fraction of max stim amp should Gaussian sample sd be?
    df_file_name="sim_output/offline_data_periodic_onoff_stim.csv"):

    pulse_len = int(round(bg_network.dbs_pulse_len/bg_network.dt, 0))
    ### Create arrays of when stimulation are occuring
    stim_begs = np.arange(
        bg_network.psd_dt_window + no_stim_time,
        bg_network.psd_dt_window + no_stim_time + time_of_stim,
        single_stim_period)
    stim_ends = np.arange(
        bg_network.psd_dt_window + no_stim_time + pulse_len,
        bg_network.psd_dt_window + no_stim_time + time_of_stim,
        single_stim_period)

    stim_ranges = [[stim_begs[i], stim_ends[i]] for i in range(len(stim_begs))]
    ### Runs simulation and computes/collects state history variables for time_past_warmup // step_time steps.
    bg_network.reset()
    
    in_stim_range = False
    while bg_network.time <= int(bg_network.psd_dt_window + no_stim_time + time_of_stim):
        in_stim_range_new = np.sum([bg_network.time >= r[0] and bg_network.time <= r[1] for r in stim_ranges]) > 0
        if in_stim_range_new != in_stim_range:
            print("Stimulation is switching on or off")
        in_stim_range = in_stim_range_new

        if in_stim_range:
            action = np.clip(
                norm.rvs(scale = bg_network.ac_high[0] * stim_sd_frac_of_max_stim), ## scale keyword is the standard deviation.
                -bg_network.ac_high[0], 
                bg_network.ac_high[0])
        else:
            action = 0

        obs, rew, done, info = bg_network.step(action)

    save_res_to_df(bg_network, df_file_name=df_file_name)



#### Continuous version  
def genOfflineCtsData(
    bg_network,
    no_stim_time = 10000,
    time_past_warmup = 100000,
    stim_duration = 9000,
    no_stim_duration = 1000,
    stim_sd_frac_of_max_stim = 0.25,
    df_file_name="sim_output/offline_data_cts_onoff_stim.csv"):

    ### Create arrays of when stimulation are occuring
    stim_begs = np.arange(
        bg_network.psd_dt_window + no_stim_time, 
        bg_network.psd_dt_window + time_past_warmup, stim_duration + no_stim_duration)
    stim_ends = np.arange(
        bg_network.psd_dt_window + no_stim_time + stim_duration, 
        bg_network.psd_dt_window + time_past_warmup, stim_duration + no_stim_duration)
    if len(stim_ends) == len(stim_begs)-1:
        stim_ends = np.concatenate((stim_ends, [bg_network.psd_dt_window + time_past_warmup]))
    stim_ranges = [[stim_begs[i], stim_ends[i]] for i in range(len(stim_begs))]
    ### Runs simulation and computes/collects state history variables for time_past_warmup // step_time steps.
    bg_network.reset()
    
    in_stim_range = False
    while bg_network.time <= int(bg_network.psd_dt_window+time_past_warmup):
        in_stim_range_new = np.sum([bg_network.time >= r[0] and bg_network.time <= r[1] for r in stim_ranges]) > 0
        if in_stim_range_new != in_stim_range:
            print("Stimulation is switching on or off")
        in_stim_range = in_stim_range_new

        if in_stim_range:
            action = np.clip(
                norm.rvs(scale = bg_network.ac_high[0] * stim_sd_frac_of_max_stim), ## scale keyword is the standard deviation.
                -bg_network.ac_high[0], 
                bg_network.ac_high[0])
        else:
            action = 0

        obs, rew, done, info = bg_network.step(action)

    save_res_to_df(bg_network, df_file_name=df_file_name)



def save_res_to_df(bg_network, df_file_name="sim_output/offline_data_periodic_stim.csv"):
    ### Save the state history dataframe.
    rowsums = np.sum(bg_network.history, axis=1)
    last_val_idx = min(np.where(rowsums==0)[0])
    time_vec = np.arange(0, last_val_idx, step=1) * bg_network.dbs_pulse_len
    full_history = bg_network.history[0:last_val_idx,:]

    init_col_names = [
        'time', 'action', 'r_reliable_index',
        'r_sgpi_sse', 'r_beta_power', 'r_pred_pd',
        'r_dbs', 'r_overall']
    
    full_history = pd.DataFrame(
        data=np.concatenate((time_vec.reshape(-1, 1), full_history), axis=1),
        columns=np.concatenate((init_col_names, bg_network.state_names)))
    
    full_history.to_csv(df_file_name)
    

def special_plot_region_potentials(
        bg_network, dt_range = None, 
        dbs_mult=1e-2,
        neuron_idx = 0,
        plot_dbs=True,
        smc_color = "red", dbs_color = "purple", 
        plot_name_tag = "",
        width = 10, height = 6, save = True):
        
        ### set time range if not specified
        if dt_range is None:
            dt_range = [bg_network.psd_dt_window, bg_network.time+1]

        t_range = bg_network.t[dt_range[0]:dt_range[1]]
        dt_range = np.arange(dt_range[0], dt_range[1])

        ### Constuct subplots of each region potentials, SMC input, DBS input
        fig, axs = plt.subplots(2, 2)
        axs[0,0].plot(t_range, bg_network.n_t_states['vth'][neuron_idx, dt_range], label="VTh")
        ## SMC impulses:
        axs[0,0].plot(
            t_range, 
            bg_network.I_smc[dt_range],
            color=smc_color,
            label = 'SMC Input')
        axs[0,0].set_title('Thalamic')
        axs[0,0].legend(loc='lower right')

        axs[0,1].plot(t_range, bg_network.n_t_states['vsn'][neuron_idx, dt_range])
        axs[0,1].set_title('STN')

        axs[1,0].plot(t_range, bg_network.n_t_states['vge'][neuron_idx, dt_range])
        axs[1,0].set_title('GPe')

        axs[1,1].plot(t_range, bg_network.n_t_states['vgi'][neuron_idx, dt_range], label="VGPi")
        axs[1,1].set_title('GPi')
        ## DBS impulses:
        if plot_dbs:
            axs[1,1].plot(
                t_range, 
                bg_network.I_dbs[dt_range]*dbs_mult,
                color=dbs_color,
                label = 'DBS Amp (rescaled)')
        
            axs[1,1].legend(loc='lower right')

        for ax in axs.flat:
            ax.set(xlabel = 'Time (msec)', ylabel = 'Vm (mV)')

        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axs.flat:
            ax.label_outer()  

        ## axis labels and legend
        plt.xlabel('Time (ms)')
        plt.ylabel('Voltage (mV)')
        plt.show()

        fig.set_size_inches(w = width, h = height, forward=True)
        if save:
            fig.savefig(bg_network.fig_loc + plot_name_tag + 'region_potentials.png', dpi=100)



if __name__ == "__main__":
    TMAX=5000 ## how long (including warm-up of 2,000 ms, which is the spectral feature computation window size, to run an episode?)
    NEURONS=10
    SEED=1
    DT=0.01
    STIMULATE=False
    CREATE_PLOTS=True
    MILLISEC_OF_DATA_COLLECT = 2000
    TIME_PAST_WARMUP = int(MILLISEC_OF_DATA_COLLECT * 1/DT)

    ### TODO: full run of the script has no commented-out lines in the below section....

    ## Make environment
    # pd_bg_network = gym.make(
    #     'bg-network-v0',
    #     n=NEURONS, 
    #     tmax=TMAX, 
    #     dt=DT, 
    #     has_pd=True,
    #     predictor_loc = "predictors/bgn_rf_model.pkl",
    #     seed=SEED,
    #     fig_loc = "docs/sim_doc/figures/")

    # genOfflinePeriodicData(
    #     pd_bg_network,
    #     time_of_stim = 190000,
    #     no_stim_time = 10000,
    #     single_stim_period = 600, ## dt
    # )

    # special_plot_region_potentials(
    #     pd_bg_network, 
    #     dt_range = [200000, 240000], 
    #     dbs_mult = 5e-2,
    #     plot_name_tag = "on_off_stim_", 
    #     save=True, )

    # ## Make environment
    pd_bg_network = gym.make(
        'bg-network-v0',
        n=NEURONS, 
        tmax=TMAX, 
        dt=DT, 
        has_pd=True,
        predictor_loc = "predictors/bgn_rf_model.pkl",
        seed=SEED,
        fig_loc = "docs/sim_doc/figures/")

    genOfflineCtsData(
        pd_bg_network,
        no_stim_time = 10000,
        time_past_warmup = 100000,
        stim_duration = 9000,
        no_stim_duration = 1000,
        stim_sd_frac_of_max_stim = 0.25,)

    special_plot_region_potentials(
        pd_bg_network, 
        dt_range = [200000, 240000], 
        dbs_mult = 5e-2,
        plot_name_tag = "cts_stim_quartermaxsd_", 
        save=True, )

    # no_pd_bg_network = gym.make(
    #     'bg-network-v0',
    #     n=NEURONS, 
    #     tmax=TMAX, 
    #     dt=DT, 
    #     has_pd=False,
    #     predictor_loc = "",
    #     seed=SEED,
    #     fig_loc = "docs/sim_doc/figures/")

    # no_pd_bg_network.reset()

    # special_plot_region_potentials(
    #     no_pd_bg_network, 
    #     dt_range = [100000, 160000], 
    #     dbs_mult = 5e-2,
    #     plot_dbs=False,
    #     plot_name_tag = "no_dbs_no_pd_", 
    #     save=True, )
    

        


    
