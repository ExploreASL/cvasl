import pandas as pd
import sys
import numpy as np
from cvasl.mriharmonize import *
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn import tree
from sklearn import metrics
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from prediction import PredictBrainAge

import warnings
warnings.filterwarnings("ignore")

#method = 'neurocombat'
method = 'neuroharmonize'

features_to_map = ['readout', 'labelling', 'sex']
patient_identifier = 'participant_id'
features_to_drop = ["m0", "id"]

Edis_path = f'../data/EDIS_output_{method}.csv'
helius_path = f'../data/HELIUS_output_{method}.csv'
sabre_path = f'../data/SABRE_output_{method}.csv'
insight_path = f'../data/Insight46_output_{method}.csv'
topmri_path = f'../data/TOP_output_{method}.csv'

patient_identifier = 'participant_id'

edis_harm = MRIdataset(Edis_path, site_id=0, decade=False, ICV = False, patient_identifier=patient_identifier, features_to_drop=features_to_drop)
helius_harm = MRIdataset(helius_path, site_id=1, decade=False, ICV = False, patient_identifier=patient_identifier, features_to_drop=features_to_drop)
sabre_harm = MRIdataset(sabre_path, site_id=2, decade=False, ICV = False, patient_identifier=patient_identifier, features_to_drop=features_to_drop)
topmri_harm = MRIdataset(topmri_path, site_id=3, decade=False, ICV = False, patient_identifier=patient_identifier, features_to_drop=features_to_drop)
insight46_harm = MRIdataset(insight_path, site_id=4, decade=False, ICV = False, patient_identifier=patient_identifier, features_to_drop=features_to_drop)

datasets_harm = [edis_harm, helius_harm, sabre_harm, topmri_harm, insight46_harm]

Edis_path = '../data/EDIS_input.csv'
helius_path = '../data/HELIUS_input.csv'
sabre_path = '../data/SABRE_input.csv'
insight_path = '../data/Insight46_input.csv'
topmri_path = ['../data/TOP_input.csv','../data/StrokeMRI_input.csv']


edis = MRIdataset(Edis_path, site_id=0, decade=False, ICV = False, patient_identifier=patient_identifier, features_to_drop=features_to_drop)
helius = MRIdataset(helius_path, site_id=1, decade=False, ICV = False, patient_identifier=patient_identifier, features_to_drop=features_to_drop)
sabre = MRIdataset(sabre_path, site_id=2, decade=False, ICV = False, patient_identifier=patient_identifier, features_to_drop=features_to_drop)
topmri = MRIdataset(topmri_path, site_id=3, decade=False, ICV = False, patient_identifier=patient_identifier, features_to_drop=features_to_drop)
insight46 = MRIdataset(insight_path, site_id=4, decade=False, ICV = False, patient_identifier=patient_identifier, features_to_drop=features_to_drop)

datasets = [edis, helius, sabre, topmri, insight46]


[_d.preprocess() for _d in datasets]
[_d.preprocess() for _d in datasets_harm]

datasets = encode_cat_features(datasets,features_to_map)
datasets_harm = encode_cat_features(datasets_harm,features_to_map)



metrics_df_val_all = []
metrics_df_all = []
metrics_df_val_all_harm = []
metrics_df_all_harm = []

pred_features = ['aca_b_cbf', 'aca_b_cov', 'csf_vol', 'gm_icvratio', 'gm_vol','gmwm_icvratio', 'mca_b_cbf', 'mca_b_cov','pca_b_cbf', 'pca_b_cov', 'totalgm_b_cbf','totalgm_b_cov', 'wm_vol', 'wmh_count', 'wmhvol_wmvol']
for model in [ExtraTreesRegressor(n_estimators=100,random_state=np.random.randint(0,100000), criterion='absolute_error', min_samples_split=2, min_samples_leaf=1, max_features='log2',bootstrap=False, n_jobs=-1, warm_start=True)]:#,LinearRegression(),SGDRegressor(),MLPRegressor(),SVR(),ElasticNetCV(),tree.DecisionTreeRegressor(),linear_model.Lasso(alpha=0.1),linear_model.Ridge(alpha=0.5),linear_model.BayesianRidge(),linear_model.ARDRegression(),linear_model.PassiveAggressiveRegressor(),linear_model.TheilSenRegressor(),linear_model.HuberRegressor(),linear_model.RANSACRegressor()]:   
    for _it in range(10):
        #randomly select seed
        seed = np.random.randint(0,100000)
        pred_harm = PredictBrainAge(model_name='extratree',model_file_name='extratree',model=model,
                            datasets=[topmri_harm],datasets_validation=[edis_harm,helius_harm,sabre_harm,insight46_harm] ,features=pred_features,target=['age'],
                            cat_category='sex',cont_category='age',n_bins=2,splits=10,test_size_p=0.05,random_state=seed)
        
        pred = PredictBrainAge(model_name='extratree',model_file_name='extratree',model=model,
                            datasets=[topmri],datasets_validation=[edis,helius,sabre,insight46] ,features=pred_features,target=['age'],
                            cat_category='sex',cont_category='age',n_bins=4,splits=5,test_size_p=0.1,random_state=seed)



        metrics_df_harm,metrics_df_val_harm, predictions_df_harm,predictions_df_val_harm, models_harm = pred_harm.predict()
        metrics_df,metrics_df_val, predictions_df,predictions_df_val, models = pred.predict()
        
        
        metrics_df_all_harm.append(metrics_df_harm)
        metrics_df_val_all_harm.append(metrics_df_val_harm)
        
        metrics_df_all.append(metrics_df)
        metrics_df_val_all.append(metrics_df_val)
        
        #print(f'Trial {_it+1} completed') 

    #now return the mean of each column of metrics_df_val
    metrics_df_val = pd.concat(metrics_df_val_all)
    metrics_df = pd.concat(metrics_df_all)

    metrics_df_val_harm = pd.concat(metrics_df_val_all_harm)
    metrics_df_harm = pd.concat(metrics_df_all_harm)

    explained_metrics = ['explained_variance', 'max_error',
        'mean_absolute_error', 'mean_squared_error', 'mean_squared_log_error',
        'median_absolute_error', 'r2', 'mean_poisson_deviance',
        'mean_gamma_deviance', 'mean_tweedie_deviance', 'd2_tweedie_score',
        'mean_absolute_percentage_error']

    explained_metrics = ['mean_absolute_error']
    val_mean = metrics_df_val[explained_metrics].mean(axis=0)
    #and the stabdard error
    val_se = metrics_df_val[explained_metrics].std(axis=0)/np.sqrt(len(metrics_df_val))

    val_mean_harm = metrics_df_val_harm[explained_metrics].mean(axis=0)
    #and the stabdard error
    val_se_harm = metrics_df_val_harm[explained_metrics].std(axis=0)/np.sqrt(len(metrics_df_val_harm))

    # concat val_mean and val_se as two columns in a new dataframe with column names 'mean' and 'se'
    val_mean_se = pd.concat([val_mean,val_se,val_mean_harm,val_se_harm],axis=1)
    val_mean_se.columns = ['mean_unharm','se unharm','mean_harm','se_harm']
    val_mean_se[f'unharmonized validation::{model.__class__.__name__}'] = val_mean_se['mean_unharm'].astype(str) + ' ± ' + val_mean_se['se unharm'].astype(str)
    val_mean_se[f'harmonized validation::{model.__class__.__name__}'] = val_mean_se['mean_harm'].astype(str) + ' ± ' + val_mean_se['se_harm'].astype(str)
    print(val_mean_se[[f'unharmonized validation::{model.__class__.__name__}',f'harmonized validation::{model.__class__.__name__}']])


    train_mean = metrics_df[explained_metrics].mean(axis=0)
    train_se = metrics_df[explained_metrics].std(axis=0)/np.sqrt(len(metrics_df))
    train_mean_harm = metrics_df_harm[explained_metrics].mean(axis=0)
    train_se_harm = metrics_df_harm[explained_metrics].std(axis=0)/np.sqrt(len(metrics_df_harm))


    train_mean_se = pd.concat([train_mean,train_se,train_mean_harm,train_se_harm],axis=1)
    train_mean_se.columns = ['mean_unharm','se unharm','mean_harm','se_harm']
    train_mean_se[f'unharmonized training::{model.__class__.__name__}'] = train_mean_se['mean_unharm'].astype(str) + ' ± ' + train_mean_se['se unharm'].astype(str)
    train_mean_se[f'harmonized training::{model.__class__.__name__}'] = train_mean_se['mean_harm'].astype(str) + ' ± ' + train_mean_se['se_harm'].astype(str)
    print(train_mean_se[[f'unharmonized training::{model.__class__.__name__}',f'harmonized training::{model.__class__.__name__}']])
    print('\n')


