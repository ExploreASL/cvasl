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

Edis_path = '../data/EDIS_input.csv'
helius_path = '../data/HELIUS_input.csv'
sabre_path = '../data/SABRE_input.csv'
insight_path = '../data/Insight46_input.csv'
topmri_path = ['../data/TOP_input.csv','../data/StrokeMRI_input.csv']
features_to_map = ['readout', 'labelling', 'sex']

edis = EDISdataset(Edis_path, site_id=0, decade=True, ICV = True)
helius = HELIUSdataset(helius_path, site_id=1, decade=True, ICV = True)
sabre = SABREdataset(sabre_path, site_id=2, decade=True, ICV = True)
topmri = TOPdataset(topmri_path, site_id=3, decade=True, ICV = True)
insight46 = Insight46dataset(insight_path, site_id=4, decade=True, ICV = True)
patient_identifier = 'participant_id'

method = 'covbat'


if method == 'neuroharmonize':

    features_to_harmonize = ['aca_b_cov', 'mca_b_cov', 'pca_b_cov', 'totalgm_b_cov', 'aca_b_cbf', 'mca_b_cbf', 'pca_b_cbf', 'totalgm_b_cbf']
    covariates = ['age', 'sex',  'icv', 'site']
    site_indicator = 'site'
    encode_cat_features([edis, helius, sabre, topmri, insight46],features_to_map)
    #ATTENTION: providing the smoothing term, e.g. ['age'] leads to longer running time and different results
    harmonizer = HarmNeuroHarmonize( features_to_harmonize = features_to_harmonize, covariates = covariates, smooth_terms = [], site_indicator=site_indicator, empirical_bayes = True)
    harmonized_data = harmonizer.harmonize([edis, helius, sabre, topmri, insight46])
    
    

elif method == 'covbat':
    edis = EDISdataset(Edis_path, site_id=0, decade=False, ICV = False)
    helius = HELIUSdataset(helius_path, site_id=1, decade=False, ICV = False)
    sabre = SABREdataset(sabre_path, site_id=2, decade=False, ICV = False)
    topmri = TOPdataset(topmri_path, site_id=3, decade=False, ICV = False)
    insight46 = Insight46dataset(insight_path, site_id=4, decade=False, ICV = False)
    encode_cat_features([edis, helius, sabre, topmri, insight46],features_to_map)

    not_harmonized= ['GM_vol', 'WM_vol', 'CSF_vol','GM_ICVRatio', 'GMWM_ICVRatio', 'WMHvol_WMvol', 'WMH_count',
                'LD', 'PLD', 'Labelling',
       'Readout', 'M0','DeepWM_B_CoV','DeepWM_B_CBF',]
    not_harmonized = [x.lower() for x in not_harmonized]
    
    features_to_harmonize = [_c.lower() for _c in edis.data.columns if _c not in not_harmonized]
    
    numerical_covariates = ['age']
    covariates = ['age', 'sex']
    site_indicator = 'site'
    harmonizer = HarmCovbat(features_to_harmonize=features_to_harmonize,covariates=covariates,site_indicator=site_indicator,patient_identifier=patient_identifier,numerical_covariates=numerical_covariates)
    harmonized_data = harmonizer.harmonize([edis, helius, sabre, topmri, insight46])

elif method == 'neurocombat':
    encode_cat_features([edis, helius, sabre, topmri, insight46],features_to_map)
    features_to_harmonize = ['ACA_B_CoV', 'MCA_B_CoV', 'PCA_B_CoV', 'TotalGM_B_CoV',
        'ACA_B_CBF', 'MCA_B_CBF', 'PCA_B_CBF', 'TotalGM_B_CBF',]
    discrete_covariates= ['sex']
    continuous_covariates=  ['age']
    site_indicator = 'site'
    harmonizer = HarmNeuroCombat(features_to_harmonize = features_to_harmonize,discrete_covariates= discrete_covariates,continuous_covariates=  continuous_covariates,site_indicator = site_indicator,patient_identifier=patient_identifier)
    harmonized_data = harmonizer.harmonize([edis, helius, sabre, topmri, insight46])

elif method == 'nestedcombat':
    encode_cat_features([edis, helius, sabre, topmri, insight46],features_to_map)
    features_to_harmonize = [
        'age', 'sex', 'ACA_B_CoV', 'MCA_B_CoV', 'PCA_B_CoV', 'TotalGM_B_CoV',
        'ACA_B_CBF', 'MCA_B_CBF', 'PCA_B_CBF', 'TotalGM_B_CBF',
    ]
    site_indicator = ['site']
    #ATTENTION: When I add readout to the discrete covariates, the code throws an error related to calculating the singular matrix, apprently messes up svd operations somehow
    discrete_covariates = ['sex','labelling']
    continuous_covariates = ['age','ld','pld']
    to_be_harmonized_or_covar  = [x.lower() for x in features_to_harmonize ]
    harmonizer = HarmNestedComBat(features_to_harmonize= features_to_harmonize,  site_indicator = site_indicator, discrete_covariates = discrete_covariates, continuous_covariates = continuous_covariates, intermediate_results_path = '.', return_extended = False, patient_identifier=patient_identifier)
    harmonized_data = harmonizer.harmonize([edis, helius, sabre, topmri, insight46])

elif method == 'comscanneuroharmonize':
    encode_cat_features([edis, helius, sabre, topmri, insight46],features_to_map)
    features_to_harmonize = ['aca_b_cov', 'mca_b_cov', 'pca_b_cov', 'totalgm_b_cov', 
                             'aca_b_cbf', 'mca_b_cbf', 'pca_b_cbf', 'totalgm_b_cbf']
    discrete_covariates = ['sex']
    continuous_covariates = ['decade']
    site_indicator = 'site'
    harmonizer = HarmComscanNeuroCombat(features_to_harmonize=features_to_harmonize,discrete_covariates=discrete_covariates,continuous_covariates=continuous_covariates,site_indicator=site_indicator) 
    harmonized_data = harmonizer.harmonize([edis, helius, sabre, topmri, insight46])

elif method == 'autocombat':
    encode_cat_features([edis, helius, sabre, topmri, insight46],features_to_map)
    features_to_harmonize = ['aca_b_cov', 'mca_b_cov', 'pca_b_cov', 'totalgm_b_cov', 
                             'aca_b_cbf', 'mca_b_cbf', 'pca_b_cbf', 'totalgm_b_cbf',]
    discrete_covariates = ['sex']
    continuous_covariates = ['decade']
    sites=['site','ld','pld','readout','labelling']
    discrete_cluster_features = ['readout','labelling']
    continuous_cluster_features = ['ld','pld'] 
    harmonizer = HarmAutoCombat(features_to_harmonize = features_to_harmonize, site_indicator=sites, discrete_covariates = discrete_covariates, continuous_covariates = continuous_covariates, continuous_cluster_features=continuous_cluster_features, discrete_cluster_features=discrete_cluster_features)
    harmonized_data = harmonizer.harmonize([edis, helius, sabre, topmri, insight46])

elif method == 'relief':
    encode_cat_features([edis, helius, sabre, topmri, insight46],features_to_map)
    edis = EDISdataset(Edis_path, site_id=0, decade=False, ICV = False, features_to_drop=["m0", "id",'site'])
    helius = HELIUSdataset(helius_path, site_id=1, decade=False, ICV = False, features_to_drop=["m0", "id",'site'])
    sabre = SABREdataset(sabre_path, site_id=2, decade=False, ICV = False, features_to_drop=["m0", "id",'site'])
    topmri = TOPdataset(topmri_path, site_id=3, decade=False, ICV = False, features_to_drop=["m0", "id",'site'])
    insight46 = Insight46dataset(insight_path, site_id=4, decade=False, ICV = False, features_to_drop=["m0", "id",'site'])
    
    features_to_harmonize = ['aca_b_cov', 'mca_b_cov', 'pca_b_cov', 'totalgm_b_cov', 
                             'aca_b_cbf', 'mca_b_cbf', 'pca_b_cbf', 'totalgm_b_cbf']
    covars = ['sex','age']
    
    harmonizer = HarmRELIEF(features_to_harmonize=features_to_harmonize,covariates=covars,patient_identifier=patient_identifier) 
    harmonized_data = harmonizer.harmonize([topmri, helius, edis,  sabre,  insight46])

elif method == 'combat++':
    encode_cat_features([edis, helius, sabre, topmri, insight46],features_to_map)
    features_to_harmonize = ['aca_b_cov', 'mca_b_cov', 'pca_b_cov', 'totalgm_b_cov', 
                             'aca_b_cbf', 'mca_b_cbf', 'pca_b_cbf', 'totalgm_b_cbf']
    discrete_covariates = ['sex']
    continuous_covariates = ['age']
    discrete_covariates_to_remove = ['labelling']
    continuous_covariates_to_remove = ['ld']
    sites='site'
    harmonizer = HarmCombatPlusPlus(features_to_harmonize = features_to_harmonize, site_indicator=sites, discrete_covariates = discrete_covariates, continuous_covariates = continuous_covariates, discrete_covariates_to_remove = discrete_covariates_to_remove, continuous_covariates_to_remove = continuous_covariates_to_remove) 
    harmonized_data = harmonizer.harmonize([edis, helius, sabre, topmri, insight46])



metrics_df_val_all = []
metrics_df_all = []
topmri, helius, edis,  sabre,  insight46 = harmonized_data
for _it in range(20):
    #randomly select seed
    seed = np.random.randint(0,100000)
    pred = PredictBrainAge(model_name='extratree',model_file_name='extratree',model=ExtraTreesRegressor(n_estimators=100, random_state=0),
                        datasets=[topmri],datasets_validation=[helius,edis,sabre,insight46] ,features=list(harmonized_data[0].data.columns.difference({'participant_id','id','site','age'})),target=['age'],
                        cat_category='sex',cont_category='age',n_bins=4,splits=5,test_size_p=0.2,random_state=seed)

    metrics_df,metrics_df_val, predictions_df,predictions_df_val, models = pred.predict()
    metrics_df_all.append(metrics_df)
    metrics_df_val_all.append(metrics_df_val)
    print(f'Trial {_it+1} completed') 

#now return the mean of each column of metrics_df_val
metrics_df_val = pd.concat(metrics_df_val_all)
metrics_df = pd.concat(metrics_df_all)
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

# concat val_mean and val_se as two columns in a new dataframe with column names 'mean' and 'se'
val_mean_se = pd.concat([val_mean,val_se],axis=1)
val_mean_se.columns = ['mean','se']
print(val_mean_se)

# [_d.reverse_encode_categorical_features() for _d in harmonized_data]
# print(harmonized_data[0].data.head())

