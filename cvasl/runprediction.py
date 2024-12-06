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



# param_grid = {
#     'n_estimators': [50, 100, 150, 200, 350, 500],
#     'n_bins': [2, 3, 4, 5, 6, 7, 8, 9, 10],
#     'splits': [1, 2, 3, 4, 5, 10],
#     'test_size_p': [0.05, 0.1, 0.2, 0.3],
#     'criterion' : ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
#     'min_samples_split': [2, 4, 8],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features' : [None, 'sqrt', 'log2'],
#     'bootstrap' : [True, False],
#     'n_jobs' : [-1],
#     'warm_start' : [True, False],
    
# }

# best_score = float('inf')
# best_params = {}

# for n_estimators in param_grid['n_estimators']:
#     for n_bins in param_grid['n_bins']:
#         for splits in param_grid['splits']:
#             print(f"n_estimators: {n_estimators}, splits: {splits}, n_bins: {n_bins}")
#             for test_size_p in param_grid['test_size_p']:
#                 for criterion in param_grid['criterion']:
#                     for min_samples_split in param_grid['min_samples_split']:
#                         for min_samples_leaf in param_grid['min_samples_leaf']:
#                             for max_features in param_grid['max_features']:
#                                 for bootstrap in param_grid['bootstrap']:
#                                     for n_jobs in param_grid['n_jobs']:
#                                         for warm_start in param_grid['warm_start']:
#                                             mae_mean_training = []
#                                             mae_mean_val = []
#                                             for _ in range(10):
#                                                 seed = np.random.randint(0,100000)
#                                                 model = ExtraTreesRegressor(n_estimators=n_estimators, random_state=seed,criterion=criterion,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,max_features=max_features,bootstrap=bootstrap,n_jobs=n_jobs,warm_start=warm_start)

#                                                 pred = PredictBrainAge(model_name='extratree',model_file_name='extratree',model=model,
#                                                             datasets=[topmri_harm],datasets_validation=[edis_harm,helius_harm,sabre_harm,insight46_harm] ,features=pred_features,target=['age'],
#                                                             cat_category='sex',cont_category='age', n_bins=n_bins, splits=splits, test_size_p=test_size_p, random_state=seed)

#                                                 metrics_df, metrics_df_val, _, _, _ = pred.predict()
#                                                 mean_mae_training = metrics_df['mean_absolute_error'].mean()
#                                                 mean_mae_val =  metrics_df_val['mean_absolute_error'].mean()
#                                                 mae_mean_training.append(mean_mae_training)
#                                                 mae_mean_val.append(mean_mae_val)
                                            
#                                             mean_mae_val = np.mean(mae_mean_val)
#                                             mean_mae_training = np.mean(mae_mean_training)
#                                             if mean_mae_training < best_score:
#                                                 best_score = mean_mae_training
#                                                 best_params = {'n_estimators': n_estimators, 'n_bins': n_bins, 'splits': splits, 'test_size_p': test_size_p, 'criterion': criterion, 'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf, 'max_features': max_features, 'bootstrap': bootstrap, 'n_jobs': n_jobs, 'warm_start': warm_start}
#                                                 print(f"ETR New best MAE: {best_score}, train MAE: {mean_mae_training}, val MAE:{mean_mae_val} with \n {best_params}")


# print(f"Best parameters: {best_params}")


# param_grid = {
#     'n_estimators': [100, 150, 200, 350, 500],
#     'n_bins': [2, 3, 4, 5, 6, 7, 8, 9, 10],
#     'splits': [1, 2, 3, 4, 5, 10],
#     'test_size_p': [0.05, 0.1, 0.2, 0.3],
#     'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
#     'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
#     'subsample': [0.5, 0.75, 1.0],
#     'criterion': ['friedman_mse', 'squared_error'],
#     'min_samples_split': [2, 4, 6, 8, 10],
#     'min_samples_leaf': [1, 2, 4, 6, 8, 10],
#     'max_depth': [3, 5, 7, 9, 11, 13, 15],
# }

# best_score = float('inf')
# best_params = {}

# for n_estimators in param_grid['n_estimators']:
#     for n_bins in param_grid['n_bins']:
#         for splits in param_grid['splits']:
#             print(f"n_estimators: {n_estimators}, splits: {splits}, n_bins: {n_bins}")
#             for test_size_p in param_grid['test_size_p']:
#                 for loss in param_grid['loss']:
#                     for learning_rate in param_grid['learning_rate']:
#                         for subsample in param_grid['subsample']:
#                             for criterion in param_grid['criterion']:
#                                 for min_samples_split in param_grid['min_samples_split']:
#                                     for min_samples_leaf in param_grid['min_samples_leaf']:
#                                         for max_depth in param_grid['max_depth']:
#                                             model = GradientBoostingRegressor(n_estimators=n_estimators, random_state=42,loss=loss,learning_rate=learning_rate,subsample=subsample,criterion=criterion,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf, max_depth=max_depth) # Fix random_state for consistent results during tuning

#                                             pred = PredictBrainAge(model_name='extratree',model_file_name='extratree',model=model,
#                                                         datasets=[topmri_harm],datasets_validation=[edis_harm,helius_harm,sabre_harm,insight46_harm] ,features=pred_features,target=['age'],
#                                                         cat_category='sex',cont_category='age', n_bins=n_bins, splits=splits, test_size_p=test_size_p, random_state=42)

#                                             metrics_df, metrics_df_val, _, _, _ = pred.predict()
#                                             mean_mae = metrics_df_val['mean_absolute_error'].mean()

#                                             if mean_mae < best_score:
#                                                 best_score = mean_mae
#                                                 best_params = {'n_estimators': n_estimators, 'n_bins': n_bins, 'splits': splits, 'test_size_p': test_size_p, 'loss': loss, 'learning_rate': learning_rate, 'subsample': subsample, 'criterion': criterion, 'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf, 'max_depth': max_depth}
#                                                 print(f"GBR New best MAE: {best_score} with \n {best_params}")

#                             print(f"Best MAE: {best_score}")
#                             print(f"Best parameters: {best_params}")