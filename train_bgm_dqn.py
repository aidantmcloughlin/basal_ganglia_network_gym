###############################################################################
### Trains Online DQN on Basal Ganglia Model Environment
###############################################################################

import os, sys, time

import pickle as pkl
import numpy as np
import random
from scipy.stats import norm
import argparse
import gym
from gym.envs.registration import register

## register custom environment gym environment
register(
        id='bg-network-v0',
        entry_point='rl_modules.envs.bg_network:BGNetwork'
    )
env = gym.make("bg-network-v0")

from stable_baselines3 import A2C, SAC
from stable_baselines3.common.logger import configure

# logger_path = os.getcwd()
# new_logger = configure(logger_path, ["stdout", "csv"])

import torch as th


model=A2C(
    policy='MlpPolicy', 
    env=env, 
    learning_rate=0.005,
    #learning_starts=50,
    #batch_size=32,
    tensorboard_log="./a2c_tensorboard/",
    #policy_kwargs=,
    verbose=1)

#model.set_logger(new_logger)

model.learn(total_timesteps=1005, log_interval=4, tb_log_name="first_run")
model.save("sac_bgn_model")

# test_env = gym.make("bg-network-v0")
# obs = test_env.reset()
# for i in range(250):
#     action, _state = model.predict(obs, deterministic=True)
#     obs, reward, done, info = test_env.step(action)

# rowsums = np.sum(test_env.history, axis=1)
# last_val_idx = min(np.where(rowsums==0)[0])
# time_vec = np.arange(0, last_val_idx, step=1) * test_env.dbs_pulse_len

# from matplotlib import pyplot as plt

# plt.plot(time_vec, test_env.history[0:last_val_idx, 0])
# test_env.plot_region_potentials(
#     dt_range = [190000, 207000], 
#     width = 10, height = 6, save = True)

# test_env.fig_loc
# test_env.I_dbs[200500:200510]
# test_env.action_space.sample()

# ### MANUALLY CREATED RL MODULES ===============================================

# from rl_modules.infrastructure.q_and_ac_trainers import AC_Trainer, Q_Trainer
# from rl_modules.infrastructure.dqn_utils import ConstantSchedule


# ### NOTE: lots of this code / arguments are "extra" from the order select codebase.
# parser = argparse.ArgumentParser()

# ## LOGGING ARGS, N_CORES, SEED:
# parser.add_argument('--log_rel_path', type=str, default = 'output/rl_sims')
# parser.add_argument('--exp_name', type=str, default='pd_bgm_train')
# parser.add_argument('--no_gpu', '-ngpu', action='store_true', default=False)
# parser.add_argument('--which_gpu', '-gpu_id', default=0)
# parser.add_argument('--run_tf_profile', action='store_true', default=True)

# parser.add_argument('--scalar_log_freq', type=int, default=int(200)) #-1 to disable
# parser.add_argument('--layer1_report_freq', type=int, default=int(-1)) #-1 to disable

# parser.add_argument('--print_layer1_wts', action='store_true', default=False)
# parser.add_argument('--nrows_print_layer1_wts', type=int, default=int(10))
# parser.add_argument('--save_layer1_wts', action='store_true', default=False)
# parser.add_argument('--log_order_cutoff', type=int, default=5)

# parser.add_argument('--n_cores', type=int, default = 1)
# parser.add_argument('--seed', type=int, default=1)

# ## For online training, saving the last N environment steps (as a behavior policy for subsequent offline training)
# parser.add_argument('--save_buffer_last_n', type=int, default=0)
# parser.add_argument('--save_buffer_loc', type=str, default='')
# parser.add_argument('--save_critic_loc', type=str, default='')


# ### Environment Args ==========================================================

# parser.add_argument('--env_name', type=str, default='bg-network-v0')
# parser.add_argument('--offline_data_loc', type=str, default='')
# parser.add_argument('--valid_data_loc', type=str, default='')
# parser.add_argument('--valid_data_n', type=int, default=0)

# parser.add_argument('--save_wts', action='store_true', default=False)
# parser.add_argument('--save_wts_loc', type=str, default='')
# parser.add_argument('--load_wts', action='store_true', default=False)
# parser.add_argument('--load_wts_loc', type=str, default='')

# parser.add_argument('--offline_use_all_data', type=str, default='False')
# parser.add_argument('--frame_history_len', type=int, default = 1)
# parser.add_argument('--append_action_to_state', type=str, default='True')
# parser.add_argument('--true_order', type=int, default=1)


# ### Discount Factor:
# parser.add_argument('--gamma', type=float, default=0.95)

# ## Runtime Args: ==============================================================
# parser.add_argument('--num_timesteps', type=int, default = int(0))
# parser.add_argument('--no_reg_epochs_adder', type=int, default = 0)
# parser.add_argument('--reg_stop_epoch_adder', type=int, default = 0)
# parser.add_argument('--num_timesteps_adder', type=int, default = 0)
# ## Exploration strategy (RND=random network distillation)
# parser.add_argument('--use_rnd', action='store_true', default=False)
# parser.add_argument('--unsupervised_exploration', action='store_true', default=False)
# parser.add_argument('--rnd_output_size', type=int, default=5)
# parser.add_argument('--rnd_n_layers', type=int, default=2)
# parser.add_argument('--rnd_size', type=int, default=400)


# ## RL Agent Specifications ====================================================
# parser.add_argument('--agent_menu', type=str, default='q_online') #q_online, cql, ac_online,
# parser.add_argument('--regularization_type', type=str, default='none') #none, nl, hier, l1, nl_fc, nl_lin, group
# parser.add_argument('--one_gate_per_order', type=str, default="False")
# parser.add_argument('--adapt_pen_on_current', type=str, default="False")
# parser.add_argument('--reg_grad_type', type=str, default="lqa") #std, lqa, prox
# parser.add_argument('--use_adam_reg_lr', type=str, default="False")

# ## Optional L1 additions:
# parser.add_argument('--do_lasso_on_current', type=str, default="False") #in NL case, whether to apply L1 penalty to first order.
# parser.add_argument('--use_n_term_coefs', type=str, default="True") #for Hier NL, whether norms are averaged by number of terms.
# parser.add_argument('--do_hier', type=str, default="False")
# parser.add_argument('--rho', type=float, default=0.1) ## rho*lambda additive L1 penalty for NL.

# ## double-Q learning by default.
# parser.add_argument('--double_q', action='store_true', default=True)
# ## boltzmann distribution for action selection instead of e-greedy:
# parser.add_argument('--use_boltzmann', action='store_true', default=False)

# parser.add_argument('--target_update_freq', type=int, default = int(1000))
# parser.add_argument('--learning_rate', '-lr', type=float, default=0.005)
# parser.add_argument('--reg_learning_rate', type=float, default=0.005)
# parser.add_argument('--batch_size', '-b', type=int, default=64)
# parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1)
# parser.add_argument('--eval_batch_size', '-eb', type=int, default=1000) #steps collected per eval iteration

# ### Zeroing-out criteria
# parser.add_argument('--eps_thres', type=float, default=1e-4)
# parser.add_argument('--eps_thres_iters', type=int, default=50)
# parser.add_argument('--zero_before_stable', action='store_true', default=True)
# parser.add_argument('--stable_zero_thres', type=int, default=100000)

# ## Offline learning arguments:
# parser.add_argument('--offline_exploitation', action='store_true', default=True)
# parser.add_argument('--cql_alpha', type=float, default=0.5)

# parser.add_argument('--exploit_rew_shift', type=float, default=0)
# parser.add_argument('--exploit_rew_scale', type=float, default=1)



# ### Neural net architecture args: =============================================

# parser.add_argument('--size_1', type=int, default=16)
# parser.add_argument('--size_2', type=int, default=16)
# parser.add_argument('--size_3', type=int, default=8)
# parser.add_argument('--size_4', type=int, default=8)
# parser.add_argument('--activation', type=str, default='relu')
# parser.add_argument('--hidden_shapes', nargs='+', default=[16, 16, 8, 8])

# ### Actor critic args:
# parser.add_argument('--num_target_updates', '-ntu', type=int, default=10)
# parser.add_argument('--num_grad_steps_per_target_update', '-ngsptu', type=int, default=10)
# parser.add_argument('--num_critic_updates_per_agent_update', type=int, default=1)
# parser.add_argument('--num_actor_updates_per_agent_update', type=int, default=1)
# parser.add_argument('--dont_standardize_advantages', '-dsa', action='store_true', default=False)

# ### ADAM Opt defaults
# parser.add_argument("--adam_beta_1", type=float, default=0.9)
# parser.add_argument("--adam_beta_2", type=float, default=0.999)
# parser.add_argument("--adam_eps", type=float, default=1e-07)

# ### Convert to Dictionary
# args, _ = parser.parse_known_args()
# params = vars(args)


# ## convert str bools to literal bools
# for p in [
#     "use_n_term_coefs",
#     "do_lasso_on_current",
#     "one_gate_per_order",
#     "adapt_pen_on_current",
#     "do_hier",
#     "use_adam_reg_lr",
#     "offline_use_all_data",
#     "append_action_to_state",
#     ]:
#     if params[p] == "True":
#         params[p] = True
#     elif params[p] == "False":
#         params[p] = False
#     else:
#         raise ValueError('Didnt provide True/False to this command arg.')

# params["learning_freq"] = 1
# params["train_batch_size"] = params["batch_size"]
# params['exploit_weight_schedule'] = ConstantSchedule(1.0)
# if params['unsupervised_exploration']:
#     params['explore_weight_schedule'] = ConstantSchedule(1.0)
#     params['exploit_weight_schedule'] = ConstantSchedule(0.0)
#     params['learning_starts'] = params['num_exploration_steps']
# params['video_log_freq'] = -1 # This param is not used for DQN
# params['eps'] = 0.2


# ### NOTE: Parameters VERY specific to online DQN training. ====================
# params['num_exploration_steps'] = 500
# params['learning_starts'] = params['num_exploration_steps']
# params['append_action_to_state'] = True
# params['frame_history_len'] = 1

# ### Runtime:
# params['no_reg_epochs'] = 0
# params['reg_stop_epoch'] = 0
# params['num_timesteps'] = 10000

# params['env_name'] = 'bg-network-v0'# 'synth-ohiot1dm-v0'
# params['seed'] = 1
# params['agent_menu'] = 'ac_online' ## cts action space
# params['regularization_type'] = 'none'
# #params['lam'] = 0

# ## TODO: figure out how to get rid of the multiple hidden state args
# params['size_1'] = 16
# params['size_2'] = 16
# params['size_3'] = 8
# params['size_4'] = 8

# params['hidden_shapes'] = [16, 16, 8, 8]
# params['batch_size'] = 64


# ### Create Directory for Logging ==============================================
# data_path = (
#     os.path.join(
#         os.path.dirname(
#             os.path.realpath(__file__)), params['log_rel_path'])
#             )


# logdir = (
#     args.exp_name + '_' + 
#     args.env_name + '_'
#     )

# logdir = os.path.join(data_path, logdir)
# params['current_time'] = time.strftime("%d-%m-%Y_%H-%M-%S")
# params['logdir_base'] = logdir
# params['logdir'] = params['logdir_base'] + "_" + params['current_time']


# ### ADD PATHS WITH SCRIPTS
# base_path = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(base_path + "/rl_modules/") 
# sys.path.append(base_path + "/../")


# if params['agent_menu'] in ['q_online', 'cql']:
#         trainer = Q_Trainer(params)
# elif params['agent_menu'] in ['ac_online']:
#     trainer = AC_Trainer(params)
# else:
#     raise(NotImplementedError)

# trainer.run_training_loop()

