import pandas as pd
import sys
sys.path.insert(0, '../../')
sys.path.insert(0, '../')
sys.path.insert(0, '../../../')
sys.path.insert(0, '../../../../')
import numpy as np
from cvasl.mriharmonize import *

Edis_path = '/Users/sabaamiri/Library/CloudStorage/OneDrive-NetherlandseScienceCenter/Dev/csval-pyproject/cvasl/ml_experiments_all_harmonizedFULL/new_data/TrainingDataComplete_EDIS.csv'
helius_path = '/Users/sabaamiri/Library/CloudStorage/OneDrive-NetherlandseScienceCenter/Dev/csval-pyproject/cvasl/ml_experiments_all_harmonizedFULL/new_data/TrainingDataComplete_HELIUS.csv'
sabre_path = '/Users/sabaamiri/Library/CloudStorage/OneDrive-NetherlandseScienceCenter/Dev/csval-pyproject/cvasl/ml_experiments_all_harmonizedFULL/new_data/TrainingDataComplete_SABRE.csv'
insight_path = '/Users/sabaamiri/Library/CloudStorage/OneDrive-NetherlandseScienceCenter/Dev/csval-pyproject/cvasl/ml_experiments_all_harmonizedFULL/new_data/TrainingDataComplete_Insight46.csv'
topmri_path = ['/Users/sabaamiri/Library/CloudStorage/OneDrive-NetherlandseScienceCenter/Dev/csval-pyproject/cvasl/ml_experiments_all_harmonizedFULL/new_data/TrainingDataComplete_TOP.csv','/Users/sabaamiri/Library/CloudStorage/OneDrive-NetherlandseScienceCenter/Dev/csval-pyproject/cvasl/ml_experiments_all_harmonizedFULL/new_data/TrainingDataComplete_StrokeMRI.csv']
features_to_map = ['readout', 'labelling', 'sex']
edis = EDISdataset(Edis_path, site_id=0, decade=True, ICV = True, cat_features_to_encode=features_to_map)
helius = HELIUSdataset(helius_path, site_id=1, decade=True, ICV = True, cat_features_to_encode=features_to_map)
sabre = SABREdataset(sabre_path, site_id=2, decade=True, ICV = True, cat_features_to_encode=features_to_map)
topmri = TOPdataset(topmri_path, site_id=3, decade=True, ICV = True, cat_features_to_encode=features_to_map)
insight46 = Insight46dataset(insight_path, site_id=4, decade=True, ICV = True, cat_features_to_encode=features_to_map)

method = 'covbat'


if method == 'neuroharmonize':

    features_to_harmonize = ['aca_b_cov', 'mca_b_cov', 'pca_b_cov', 'totalgm_b_cov', 'aca_b_cbf', 'mca_b_cbf', 'pca_b_cbf', 'totalgm_b_cbf']
    covariates = ['age', 'sex', 'site', 'icv']
    harmonizer = HarmNeuroHarmonize(features_to_map, features_to_harmonize, covariates)
    harmonized_data = harmonizer.harmonize([edis, helius, sabre, topmri, insight46])

elif method == 'covbat':
    edis = EDISdataset(Edis_path, site_id=0, decade=False, ICV = False, cat_features_to_encode=features_to_map)
    helius = HELIUSdataset(helius_path, site_id=1, decade=False, ICV = False, cat_features_to_encode=features_to_map)
    sabre = SABREdataset(sabre_path, site_id=2, decade=False, ICV = False, cat_features_to_encode=features_to_map)
    topmri = TOPdataset(topmri_path, site_id=3, decade=False, ICV = False, cat_features_to_encode=features_to_map)
    insight46 = Insight46dataset(insight_path, site_id=4, decade=False, ICV = False, cat_features_to_encode=features_to_map)
    
    not_harmonized= ['GM_vol', 'WM_vol', 'CSF_vol','GM_ICVRatio', 'GMWM_ICVRatio', 'WMHvol_WMvol', 'WMH_count',
                'LD', 'PLD', 'Labelling',
       'Readout', 'M0','DeepWM_B_CoV','DeepWM_B_CBF',]
    not_harmonized = [x.lower() for x in not_harmonized]
    
    to_be_harmonized_or_covar = [_c.lower() for _c in edis.data.columns if _c not in not_harmonized]
    
    patient_identifier = 'participant_id'
    numerical_covariates = 'age'
    pheno_features = ['age', 'sex']
    harmonizer = HarmCovbat(to_be_harmonized_or_covar,pheno_features,patient_identifier=patient_identifier,numerical_covariates=numerical_covariates)
    harmonized_data = harmonizer.harmonize([edis, helius, sabre, topmri, insight46])

elif method == 'neurocombat':

    to_be_harmonized_or_covar = ['DeepWM_B_CoV', 'ACA_B_CoV', 'MCA_B_CoV', 'PCA_B_CoV', 'TotalGM_B_CoV', 'DeepWM_B_CBF', 
        'ACA_B_CBF', 'MCA_B_CBF', 'PCA_B_CBF', 'TotalGM_B_CBF', 'DeepWM_B_CoV', 'DeepWM_B_CBF',]
    harmonizer = HarmNeuroCombat(features_to_harmonize = to_be_harmonized_or_covar,cat_features = ['age'],cont_features = ['sex'],batch_col = 'site')
    harmonized_data = harmonizer.harmonize([edis, helius, sabre, topmri, insight46])

elif method == 'nestedcombat':


    to_be_harmonized_or_covar = [
        'age', 'sex','deepWM_B_CoV', 'ACA_B_CoV', 'MCA_B_CoV', 'PCA_B_CoV', 'TotalGM_B_CoV',
        'DeepWM_B_CBF', 'ACA_B_CBF', 'MCA_B_CBF', 'PCA_B_CBF', 'TotalGM_B_CBF',
    ]
    to_be_harmonized_or_covar  = [x.lower() for x in to_be_harmonized_or_covar ]
    harmonizer = HarmNestedComBat(to_be_harmonized_or_covar = to_be_harmonized_or_covar,  batch_testing_list = ['site'], categorical_testing_cols = ['sex'], continuous_testing_cols = ['age'], intermediate_results_path = '.', return_extended = False)
    harmonized_data = harmonizer.harmonize([edis, helius, sabre, topmri, insight46])

elif method == 'comscanneuroharmonize':

    features_to_harmonize = ['aca_b_cov', 'mca_b_cov', 'pca_b_cov', 'totalgm_b_cov', 
                             'aca_b_cbf', 'mca_b_cbf', 'pca_b_cbf', 'totalgm_b_cbf']
    discrete_covariates = ['sex']
    continuous_covariates = ['decade']
    harmonizer = HarmComscanNeuroCombat(features_to_harmonize,['site'],discrete_covariates,continuous_covariates) 
    harmonized_data = harmonizer.harmonize([edis, helius, sabre, topmri, insight46])

elif method == 'autocombat':

    features_to_harmonize = ['aca_b_cov', 'mca_b_cov', 'pca_b_cov', 'totalgm_b_cov', 
                             'aca_b_cbf', 'mca_b_cbf', 'pca_b_cbf', 'totalgm_b_cbf']
    discrete_covariates = ['sex']
    continuous_covariates = ['decade']
    sites=['site']
    harmonizer = HarmAutoCombat(features_to_harmonize = features_to_harmonize, sites=sites, discrete_covariates = discrete_covariates, continuous_covariates = continuous_covariates) 
    harmonized_data = harmonizer.harmonize([edis, helius, sabre, topmri, insight46])

elif method == 'relief':
    features_to_map = ['readout', 'labelling', 'sex']
    edis = EDISdataset(Edis_path, site_id=0, decade=False, ICV = False, cat_features_to_encode=features_to_map, features_to_drop=["m0", "id",'site'])
    helius = HELIUSdataset(helius_path, site_id=1, decade=False, ICV = False, cat_features_to_encode=features_to_map, features_to_drop=["m0", "id",'site'])
    sabre = SABREdataset(sabre_path, site_id=2, decade=False, ICV = False, cat_features_to_encode=features_to_map, features_to_drop=["m0", "id",'site'])
    topmri = TOPdataset(topmri_path, site_id=3, decade=False, ICV = False, cat_features_to_encode=features_to_map, features_to_drop=["m0", "id",'site'])
    insight46 = Insight46dataset(insight_path, site_id=4, decade=False, ICV = False, cat_features_to_encode=features_to_map, features_to_drop=["m0", "id",'site'])

    features_to_harmonize = ['aca_b_cov', 'mca_b_cov', 'pca_b_cov', 'totalgm_b_cov', 
                             'aca_b_cbf', 'mca_b_cbf', 'pca_b_cbf', 'totalgm_b_cbf']
    covars = ['sex','age']
    continuous_covariates = ['decade']
    sites=['site']
    
    harmonizer = HarmRELIEF(features_to_harmonize, covars) 
    harmonized_data = harmonizer.harmonize([topmri, helius, edis,  sabre,  insight46])

[_d.reverse_encode_categorical_features() for _d in harmonized_data]

print(harmonized_data[0].initial_statistics)
print(harmonized_data[0].harmonized_statistics)
