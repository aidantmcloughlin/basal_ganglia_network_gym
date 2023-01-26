# basal_ganglia_network_gym
Gym implementation of Basal Ganglia Network model from [So et Al 2012](https://pubmed.ncbi.nlm.nih.gov/21984318/)


## File structure

* *env*: Contains the BG simulation environment.  In particular, *bg_network.py* is the workhorse script for defining the transition equations, computing the EEG-level state variables, and embedding plotting functions into the class.

* *sim_output*: All outputs related to the simulation environment are stored here.

* *get_classifier_bgn_data.py*: Runs the **unstimulated** BG network (with and without PD) in order to generate state history data with which to train the random forest classifier in *train_pd_predictor.py*

* *train_pd_predictor.py*: Trains random forest model (and logistic regression model) on **unstimulated** PD and control state histories. New predictors may optionally be saved into the *bg_network* environment directory, for which the BG simulation environment can use the PD classification value (proportion) as a reward variable.

* *gen_offline_rl_bgn_data.py*: Main file demonstrating how to generate offline simulation datasets under various stimulation paradigms.


## Incoming files

* *train_bgm.py*: This file will demonstrate an example RL algorithm being applied to the simulation environment and summarize the results.



