import pandas as pd
import sys
sys.path.insert(0, '../../')
sys.path.insert(0, '../')
sys.path.insert(0, '../../../')
sys.path.insert(0, '../../../../')
import numpy as np
from cvasl.mriharmonize import *
from sklearn.ensemble import ExtraTreesRegressor

import warnings
warnings.filterwarnings("ignore")

method = 'neuroharmonize' # 'neuroharmonize' or 'covbat' or 'neurocombat'
features_to_map = ['readout', 'labelling', 'sex']

Edis_path = f'../data/EDIS_output_{method}.csv'
helius_path = f'../data/HELIUS_output_{method}.csv'
sabre_path = f'../data/SABRE_output_{method}.csv'
insight_path = f'../data/Insight46_output_{method}.csv'
topmri_path = f'../data/TOP_output_{method}.csv'


edis_harm = EDISdataset(Edis_path, site_id=0, decade=True, ICV = True,features_to_drop=None)
helius_harm = HELIUSdataset(helius_path, site_id=1, decade=True, ICV = True,features_to_drop=None)
sabre_harm = SABREdataset(sabre_path, site_id=2, decade=True, ICV = True,features_to_drop=None)
topmri_harm = TOPdataset(topmri_path, site_id=3, decade=True, ICV = True,features_to_drop=None)
insight46_harm = Insight46dataset(insight_path, site_id=4, decade=True, ICV = True,features_to_drop=None)

Edis_path = '../data/EDIS_input.csv'
helius_path = '../data/HELIUS_input.csv'
sabre_path = '../data/SABRE_input.csv'
insight_path = '../data/Insight46_input.csv'
topmri_path = ['../data/TOP_input.csv','../data/StrokeMRI_input.csv']


edis = EDISdataset(Edis_path, site_id=0, decade=True, ICV = True)
helius = HELIUSdataset(helius_path, site_id=1, decade=True, ICV = True)
sabre = SABREdataset(sabre_path, site_id=2, decade=True, ICV = True)
topmri = TOPdataset(topmri_path, site_id=3, decade=True, ICV = True)
insight46 = Insight46dataset(insight_path, site_id=4, decade=True, ICV = True)

print(edis.data.head())
print(edis_harm.data.head())
[edis, helius, sabre, topmri, insight46] = encode_cat_features([edis, helius, sabre, topmri, insight46],features_to_map)
print(edis_harm.data.head())



metrics_df_val_all = []
metrics_df_all = []
metrics_df_val_all_harm = []
metrics_df_all_harm = []


for _it in range(20):
    #randomly select seed
    seed = np.random.randint(0,100000)
    pred_harm = PredictBrainAge(model_name='extratree',model_file_name='extratree',model=ExtraTreesRegressor(n_estimators=100, random_state=0),
                        datasets=[topmri_harm],datasets_validation=[helius_harm,edis_harm,sabre_harm,insight46_harm] ,features=list(edis.data.columns.difference({'participant_id','id','site','age'})),target=['age'],
                        cat_category='sex',cont_category='age',n_bins=4,splits=5,test_size_p=0.2,random_state=seed)
    pred = PredictBrainAge(model_name='extratree',model_file_name='extratree',model=ExtraTreesRegressor(n_estimators=100, random_state=0),
                        datasets=[topmri],datasets_validation=[helius,edis,sabre,insight46] ,features=list(edis.data.columns.difference({'participant_id','id','site','age'})),target=['age'],
                        cat_category='sex',cont_category='age',n_bins=4,splits=5,test_size_p=0.2,random_state=seed)

    metrics_df,metrics_df_val, predictions_df,predictions_df_val, models = pred.predict()
    metrics_df_harm,metrics_df_val_harm, predictions_df_harm,predictions_df_val_harm, models_harm = pred_harm.predict()
    
    metrics_df_all_harm.append(metrics_df_harm)
    metrics_df_val_all_harm.append(metrics_df_val_harm)
    
    metrics_df_all.append(metrics_df)
    metrics_df_val_all.append(metrics_df_val)
    
    print(f'Trial {_it+1} completed') 

#now return the mean of each column of metrics_df_val
metrics_df_val = pd.concat(metrics_df_val_all)
metrics_df = pd.concat(metrics_df_all)

metrics_df_val_harm = pd.concat(metrics_df_val_all_harm)
metrics_df_harm = pd.concat(metrics_df_all_harm)

val_mean = metrics_df_val[['explained_variance', 'max_error',
       'mean_absolute_error', 'mean_squared_error', 'mean_squared_log_error',
       'median_absolute_error', 'r2', 'mean_poisson_deviance',
       'mean_gamma_deviance', 'mean_tweedie_deviance', 'd2_tweedie_score',
       'mean_absolute_percentage_error']].mean(axis=0)
#and the stabdard error
val_se = metrics_df_val[['explained_variance', 'max_error',
       'mean_absolute_error', 'mean_squared_error', 'mean_squared_log_error',
       'median_absolute_error', 'r2', 'mean_poisson_deviance',
       'mean_gamma_deviance', 'mean_tweedie_deviance', 'd2_tweedie_score',
       'mean_absolute_percentage_error']].std(axis=0)/np.sqrt(len(metrics_df_val))

val_mean_harm = metrics_df_val_harm[['explained_variance', 'max_error',
       'mean_absolute_error', 'mean_squared_error', 'mean_squared_log_error',
       'median_absolute_error', 'r2', 'mean_poisson_deviance',
       'mean_gamma_deviance', 'mean_tweedie_deviance', 'd2_tweedie_score',
       'mean_absolute_percentage_error']].mean(axis=0)
#and the stabdard error
val_se_harm = metrics_df_val_harm[['explained_variance', 'max_error',
       'mean_absolute_error', 'mean_squared_error', 'mean_squared_log_error',
       'median_absolute_error', 'r2', 'mean_poisson_deviance',
       'mean_gamma_deviance', 'mean_tweedie_deviance', 'd2_tweedie_score',
       'mean_absolute_percentage_error']].std(axis=0)/np.sqrt(len(metrics_df_val_harm))

# concat val_mean and val_se as two columns in a new dataframe with column names 'mean' and 'se'
val_mean_se = pd.concat([val_mean,val_se,val_mean_harm,val_se_harm],axis=1)
val_mean_se.columns = ['mean_unharm','se unharm','mean_harm','se_harm']
val_mean_se['unharmonized results'] = val_mean_se['mean_unharm'].astype(str) + ' ± ' + val_mean_se['se unharm'].astype(str)
val_mean_se['harmonized results'] = val_mean_se['mean_harm'].astype(str) + ' ± ' + val_mean_se['se_harm'].astype(str)
print(val_mean_se[['unharmonized results','harmonized results']])
