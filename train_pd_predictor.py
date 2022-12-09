import os, sys
import pickle as pkl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

## TODO: remove if not still using.
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acovf


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

RF_SEED=11
FOLDS=5


pd_state_history = pd.read_csv("sim_output/state_history_hasPDTrue.csv", index_col=0)
no_pd_state_history = pd.read_csv("sim_output/state_history_hasPDFalse.csv", index_col=0)

pd_state_history['has_pd'] = 1
no_pd_state_history['has_pd'] = 0


### Build Classifier
state_history_df = pd.concat((pd_state_history, no_pd_state_history), axis=0)
ind_cols = np.setdiff1d(state_history_df.columns, ['has_pd', 'time_ms'])
dep_cols = ['has_pd']

### order by time
state_history_df = state_history_df.sort_values('time_ms')

### Time-based CV
df_nrow = state_history_df.shape[0]

base_fold_size = df_nrow // FOLDS
remainder = df_nrow % (base_fold_size * FOLDS)
folds_idx_vec = np.concatenate((
    (np.arange(0, FOLDS) * base_fold_size).reshape(-1, 1), 
    (np.arange(1, FOLDS+1) * base_fold_size).reshape(-1, 1), 
    ), axis=1)
folds_idx_vec[FOLDS-1, 1] += remainder

### Random Forest and Logistc classifiers have 100% success predicting PD state in held out 200 millisec segment.
for i in range(FOLDS):
    valid_idx = np.arange(folds_idx_vec[i,0], folds_idx_vec[i,1])
    train_idx = np.setdiff1d(np.arange(0, df_nrow), valid_idx)
    valid_data = state_history_df.iloc[valid_idx, :]
    train_data = state_history_df.iloc[train_idx, :]

    pd_rf_model = RandomForestClassifier(random_state=RF_SEED)    
    pd_rf_model.fit(X=train_data.loc[:, ind_cols], y=train_data.loc[:, dep_cols].to_numpy().reshape(-1))

    pd_logistic_model = LogisticRegression(random_state = RF_SEED)
    pd_logistic_model.fit(X=train_data.loc[:, ind_cols], y=train_data.loc[:, dep_cols].to_numpy().reshape(-1))

    true_vec = valid_data.loc[:, dep_cols].to_numpy().reshape(-1)
    predict_vec_rf = pd_rf_model.predict(X=valid_data.loc[:, ind_cols])
    predict_vec_logistic = pd_logistic_model.predict(X=valid_data.loc[:, ind_cols])
    

    probas_rf = pd_rf_model.predict_proba(X=valid_data.loc[:, ind_cols])
    probas_logistic = pd_logistic_model.predict_proba(X=valid_data.loc[:, ind_cols])

    accuracy_rf = np.sum(predict_vec_rf == true_vec) / len(true_vec)
    accuracy_logistic = np.sum(predict_vec_logistic == true_vec) / len(true_vec)
    print(accuracy_rf)
    print(accuracy_logistic)


### Rerun on the full data.
pd_rf_model = RandomForestClassifier(random_state=RF_SEED)
pd_rf_model.fit(X=state_history_df.loc[:, ind_cols], y=state_history_df.loc[:, dep_cols].to_numpy().reshape(-1))

pd_logistic_model = LogisticRegression(random_state=RF_SEED)
pd_logistic_model.fit(X=state_history_df.loc[:, ind_cols], y=state_history_df.loc[:, dep_cols].to_numpy().reshape(-1))

### Collect and plot feature importance.
feat_importance_rf = pd_rf_model.feature_importances_
feat_importance_logistic = np.abs(pd_logistic_model.coef_[0]) / np.sum(np.abs(pd_logistic_model.coef_[0]))

feat_importance_df = pd.DataFrame({
    'col_name': ind_cols,
    'feat_import_rf': feat_importance_rf,
    'feat_import_logistic': feat_importance_logistic,
    })

feat_importance_df.to_csv("sim_output/predictors/feat_importance.csv")


with open("sim_output/predictors/bgn_rf_model.pkl", 'wb') as f:
    pkl.dump(pd_rf_model, f)

with open("sim_output/predictors/bgn_logistic_model.pkl", 'wb') as f:
    pkl.dump(pd_logistic_model, f)


##### =========================================================================
### Manually Determined Beta Power Cutoffs
##### =========================================================================

### Under the PD EEG, the VGI, VGE and VSN regions all show increased beta band relative power 
##      (which is also reflected in the random forest model feature importance list)

print("determine cutoffs")

## Fully separating feature
fig, axs = plt.subplots(2, 2)
axs[0,0].hist(pd_state_history['vgi.beta.rel_power'], label="PD")
axs[0,0].hist(no_pd_state_history['vgi.beta.rel_power'], label="No PD")
axs[0,0].set_title("GPi", loc='left', fontweight='bold')


axs[1,0].hist(pd_state_history['vge.beta.rel_power'], label="PD")
axs[1,0].hist(no_pd_state_history['vge.beta.rel_power'], label="No PD")
axs[1,0].set_title("GPe", loc='left', fontweight='bold')

axs[0,1].hist(pd_state_history['vsn.beta.rel_power'], label="PD")
axs[0,1].hist(no_pd_state_history['vsn.beta.rel_power'], label="No PD")
#axs[0,1].legend(loc='best')
axs[0,1].set_title("STN", loc='left', fontweight='bold')

axs[-1, -1].axis('off')
handles, labels = axs[0,1].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower right')

for ax in axs.flat:
    ax.set(xlabel = 'Beta Band Relative PSD', )
  

fig.set_size_inches(w = 9, h = 7.5, forward=True)
fig.savefig('docs/sim_doc/other_figures/beta_power_histograms.png', dpi=100)
plt.close(fig)

### record mean positions of the relative powers and save as csv
healthy_beta_rel_power_means = pd.DataFrame({
    'vgi': [np.mean(no_pd_state_history['vgi.beta.rel_power'])],
    'vge': [np.mean(no_pd_state_history['vge.beta.rel_power'])],
    'vsn': [np.mean(no_pd_state_history['vsn.beta.rel_power'])],
    })


healthy_beta_rel_power_means.to_csv("sim_output/healthy_beta_rel_power_means.csv")

##### =========================================================================
### TODO: old code? ===========================================================
# single_df_nrow = pd_state_history.shape[0]

# def skipIdxVec(k, nrow=single_df_nrow):
#     return(np.arange(0,nrow, k).astype(int))

# ### Various time series plotting functions.
# plot_acf(no_pd_state_history['vsn.vge.gamma.avg_coherence'])
# plot_pacf(pd_state_history['vsn.vge.gamma.avg_coherence'])

# plot_col=18

# pd_state_history.shape
# plt.acorr(pd_state_history.iloc[:, plot_col], maxlags=None)
# plt.acorr(pd_state_history.iloc[skipIdxVec(50), plot_col])
# plt.plot(pd_state_history.iloc[:, plot_col])
