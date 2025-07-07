import pandas as pd
import sys
import numpy as np
from cvasl.dataset import *
from cvasl.harmonizers import *
from cvasl.prediction import *

from sklearn.ensemble import ExtraTreesRegressor

import warnings
warnings.filterwarnings("ignore")

Edis_path = '../data/EDIS_input.csv'
helius_path = '../data/HELIUS_input.csv'
sabre_path = '../data/SABRE_input.csv'
insight_path = '../data/Insight46_input.csv'
topmri_path = ['../data/TOP_input.csv','../data/StrokeMRI_input.csv']

features_to_drop = ["m0", "id"]

features_to_map = ['readout', 'labelling', 'sex']
patient_identifier = 'participant_id'

edis = MRIdataset(Edis_path, site_id=0, decade=True, ICV = True, patient_identifier=patient_identifier, features_to_drop=features_to_drop)
helius = MRIdataset(helius_path, site_id=1, decade=True, ICV = True, patient_identifier=patient_identifier, features_to_drop=features_to_drop)
sabre = MRIdataset(sabre_path, site_id=2, decade=True, ICV = True, patient_identifier=patient_identifier, features_to_drop=features_to_drop)
topmri = MRIdataset(topmri_path, site_id=3, decade=True, ICV = True, patient_identifier=patient_identifier, features_to_drop=features_to_drop)
insight46 = MRIdataset(insight_path, site_id=4, decade=True, ICV = True, patient_identifier=patient_identifier, features_to_drop=features_to_drop)

datasets = [edis, helius, sabre, topmri, insight46]
[_d.preprocess() for _d in datasets]


datasets = encode_cat_features(datasets,features_to_map)

method = 'neurocombat' 


if method == 'neuroharmonize':

    features_to_harmonize = ['aca_b_cov', 'mca_b_cov', 'pca_b_cov', 'totalgm_b_cov', 'aca_b_cbf', 'mca_b_cbf', 'pca_b_cbf', 'totalgm_b_cbf']
    covariates = ['age', 'sex',  'icv', 'site']
    site_indicator = 'site'
    #ATTENTION: providing the smoothing term, e.g. ['age'] leads to longer running time and different results
    harmonizer =  NeuroHarmonize( features_to_harmonize = features_to_harmonize, covariates = covariates, smooth_terms = [], site_indicator=site_indicator, empirical_bayes = True)
    harmonized_data = harmonizer.harmonize(datasets)
    
    

elif method == 'covbat':


    site_indicator = 'site'
    features_to_harmonize = [patient_identifier, site_indicator, 'age', 'sex', 'site', 'aca_b_cov', 'mca_b_cov', 'pca_b_cov', 'totalgm_b_cov', 'aca_b_cbf', 'mca_b_cbf', 'pca_b_cbf', 'totalgm_b_cbf']
    numerical_covariates = ['age']
    covariates = ['age', 'sex']    
    harmonizer =  Covbat(features_to_harmonize=features_to_harmonize,covariates=covariates,site_indicator=site_indicator,patient_identifier=patient_identifier,numerical_covariates=numerical_covariates)
    harmonized_data = harmonizer.harmonize(datasets)

elif method == 'neurocombat':
    
    features_to_harmonize = ['ACA_B_CoV', 'MCA_B_CoV', 'PCA_B_CoV', 'TotalGM_B_CoV','ACA_B_CBF', 'MCA_B_CBF', 'PCA_B_CBF', 'TotalGM_B_CBF']
    discrete_covariates= ['sex']
    continuous_covariates=  ['age']
    site_indicator = 'site'
    harmonizer =  NeuroCombat(features_to_harmonize = features_to_harmonize,discrete_covariates= discrete_covariates,continuous_covariates=  continuous_covariates,site_indicator = site_indicator,patient_identifier=patient_identifier)
    harmonized_data = harmonizer.harmonize(datasets)

elif method == 'nestedcombat':

    features_to_harmonize = [
        'ACA_B_CoV', 'MCA_B_CoV', 'PCA_B_CoV', 'TotalGM_B_CoV',
        'ACA_B_CBF', 'MCA_B_CBF', 'PCA_B_CBF', 'TotalGM_B_CBF'
    ]
    batch_list_harmonisations = ['readout', 'ld', 'pld']
    site_indicator = ['site']
    discrete_covariates = ['sex']
    continuous_covariates = ['age']

    harmonizer =  NestedComBat(
        features_to_harmonize=features_to_harmonize,
        batch_list_harmonisations=batch_list_harmonisations,
        site_indicator=site_indicator,
        discrete_covariates=discrete_covariates,
        continuous_covariates=continuous_covariates,
        intermediate_results_path='.',
        return_extended=False,
        use_gmm=False
    )

    harmonized_data = harmonizer.harmonize(datasets)
elif method == 'comscanneuroharmonize':
    
    features_to_harmonize = ['aca_b_cov', 'mca_b_cov', 'pca_b_cov', 'totalgm_b_cov', 
                             'aca_b_cbf', 'mca_b_cbf', 'pca_b_cbf', 'totalgm_b_cbf']
    discrete_covariates = ['sex']
    continuous_covariates = ['decade']
    site_indicator = 'site'
    harmonizer =  ComscanNeuroCombat(features_to_harmonize=features_to_harmonize,discrete_covariates=discrete_covariates,continuous_covariates=continuous_covariates,site_indicator=site_indicator) 
    harmonized_data = harmonizer.harmonize(datasets)



elif method == 'autocombat':
    
    features_to_harmonize = ['aca_b_cov', 'mca_b_cov', 'pca_b_cov', 'totalgm_b_cov', 
                             'aca_b_cbf', 'mca_b_cbf', 'pca_b_cbf', 'totalgm_b_cbf',]
    data_subset = ['aca_b_cov', 'mca_b_cov', 'pca_b_cov', 'totalgm_b_cov', 
                             'aca_b_cbf', 'mca_b_cbf', 'pca_b_cbf', 'totalgm_b_cbf','site','readout','labelling','pld','ld','sex','age']
    discrete_covariates = ['sex']
    continuous_covariates = ['age']
    sites=['site','readout','pld','ld']
    discrete_cluster_features = ['site','readout']
    continuous_cluster_features = ['pld','ld'] 
    harmonizer =  AutoCombat(data_subset = data_subset, features_to_harmonize = features_to_harmonize, site_indicator=sites, discrete_covariates = discrete_covariates, continuous_covariates = continuous_covariates, continuous_cluster_features=continuous_cluster_features, discrete_cluster_features=discrete_cluster_features)
    harmonized_data = harmonizer.harmonize(datasets)

elif method == 'relief':
    
    features_to_harmonize = ['aca_b_cov', 'mca_b_cov', 'pca_b_cov', 'totalgm_b_cov', 
                             'aca_b_cbf', 'mca_b_cbf', 'pca_b_cbf', 'totalgm_b_cbf']
    covars = ['sex','age']
    
    harmonizer =  RELIEF(features_to_harmonize=features_to_harmonize,covariates=covars,patient_identifier=patient_identifier) 
    harmonized_data = harmonizer.harmonize([topmri, helius, edis,  sabre,  insight46])

elif method == 'combat++':
    
    features_to_harmonize = ['aca_b_cov', 'mca_b_cov', 'pca_b_cov', 'totalgm_b_cov', 
                             'aca_b_cbf', 'mca_b_cbf', 'pca_b_cbf', 'totalgm_b_cbf']
    discrete_covariates = ['sex']
    continuous_covariates = ['age']
    discrete_covariates_to_remove = ['labelling']
    continuous_covariates_to_remove = ['ld']
    sites='site'
    harmonizer =  CombatPlusPlus(features_to_harmonize = features_to_harmonize, site_indicator=sites, discrete_covariates = discrete_covariates, continuous_covariates = continuous_covariates, discrete_covariates_to_remove = discrete_covariates_to_remove, continuous_covariates_to_remove = continuous_covariates_to_remove) 
    harmonized_data = harmonizer.harmonize(datasets)

print(harmonized_data[1].data.tail())

[_d.prepare_for_export() for _d in datasets]
print(harmonized_data[1].data.tail())
topmri.data.to_csv(Edis_path.replace('input',f'input_topmri'),index=False)

harmonized_data[0].data.to_csv(Edis_path.replace('input',f'output_{method}'),index=False)
harmonized_data[1].data.to_csv(helius_path.replace('input',f'output_{method}'),index=False)
harmonized_data[2].data.to_csv(sabre_path.replace('input',f'output_{method}'),index=False)
harmonized_data[3].data.to_csv(topmri_path[0].replace('input',f'output_{method}'),index=False)
harmonized_data[4].data.to_csv(insight_path.replace('input',f'output_{method}'),index=False)

