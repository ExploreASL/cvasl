import pandas as pd
import numpy as np
from cvasl.mriharmonize import *



features_to_map = ['readout', 'labelling', 'sex']
edis = EDISdataset('/Users/sabaamiri/Library/CloudStorage/OneDrive-NetherlandseScienceCenter/Dev/csval-pyproject/cvasl/ml_experiments_all_harmonizedFULL/new_data/TrainingDataComplete_EDIS.csv', site_id=0, decade=True, ICV = True, cat_features_to_encode=features_to_map)
helius = HELIUSdataset('/Users/sabaamiri/Library/CloudStorage/OneDrive-NetherlandseScienceCenter/Dev/csval-pyproject/cvasl/ml_experiments_all_harmonizedFULL/new_data/TrainingDataComplete_HELIUS.csv', site_id=1, decade=True, ICV = True, cat_features_to_encode=features_to_map)
sabre = SABREdataset('/Users/sabaamiri/Library/CloudStorage/OneDrive-NetherlandseScienceCenter/Dev/csval-pyproject/cvasl/ml_experiments_all_harmonizedFULL/new_data/TrainingDataComplete_SABRE.csv', site_id=2, decade=True, ICV = True, cat_features_to_encode=features_to_map)
topmri = TOPdataset(['/Users/sabaamiri/Library/CloudStorage/OneDrive-NetherlandseScienceCenter/Dev/csval-pyproject/cvasl/ml_experiments_all_harmonizedFULL/new_data/TrainingDataComplete_TOP.csv','/Users/sabaamiri/Library/CloudStorage/OneDrive-NetherlandseScienceCenter/Dev/csval-pyproject/cvasl/ml_experiments_all_harmonizedFULL/new_data/TrainingDataComplete_StrokeMRI.csv'], site_id=3, decade=True, ICV = True, cat_features_to_encode=features_to_map)
insight46 = Insight46dataset('/Users/sabaamiri/Library/CloudStorage/OneDrive-NetherlandseScienceCenter/Dev/csval-pyproject/cvasl/ml_experiments_all_harmonizedFULL/new_data/TrainingDataComplete_Insight46.csv', site_id=4, decade=True, ICV = True, cat_features_to_encode=features_to_map)

method = 'autocombat'


if method == 'neuroharmonize':

    features_to_harmonize = ['aca_b_cov', 'mca_b_cov', 'pca_b_cov', 'totalgm_b_cov', 'aca_b_cbf', 'mca_b_cbf', 'pca_b_cbf', 'totalgm_b_cbf']
    covariates = ['age', 'sex', 'site', 'icv']
    harmonizer = HarmNeuroHarmonize(features_to_map, features_to_harmonize, covariates)
    harmonized_data = harmonizer.harmonize([edis, helius, sabre, topmri, insight46])

elif method == 'covbat':
    
    to_be_harmonized_or_covar = [
        'Age', 'Sex', 'ACA_B_CoV', 'MCA_B_CoV', 'PCA_B_CoV','DeepWM_B_CoV',
        'ACA_B_CBF', 'MCA_B_CBF', 'PCA_B_CBF', 'LD', 'PLD','DeepWM_B_CBF',
        'Labelling', 'Readout', 'TotalGM_B_CoV',
        'TotalGM_B_CBF',
    ]
    patient_identifier = 'participant_id'
    numerical_covariates = 'age'
    pheno_features = ['Age', 'Sex']
    print(edis.data.columns)
    harmonizer = HarmCovbat(to_be_harmonized_or_covar,pheno_features,patient_identifier=patient_identifier,numerical_covariates=numerical_covariates)
    harmonized_data = harmonizer.harmonize([edis, helius, sabre, topmri, insight46])

elif method == 'neurocombat':

    to_be_harmonized_or_covar = [
        'Age', 'Sex', 'DeepWM_B_CoV', 'ACA_B_CoV', 'MCA_B_CoV', 'PCA_B_CoV', 'TotalGM_B_CoV', 'DeepWM_B_CBF', 
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

print(harmonized_data[0].data)