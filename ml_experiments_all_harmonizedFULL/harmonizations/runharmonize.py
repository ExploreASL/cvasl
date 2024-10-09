import pandas as pd
import numpy as np
import cvasl.vendor.covbat.covbat as covbat
import cvasl.vendor.comscan.neurocombat as autocombat
import cvasl.vendor.neurocombat.neurocombat as neurocombat
import cvasl.vendor.open_nested_combat.nest as nest
from neuroHarmonize import harmonizationLearn
from cvasl.mriharmonize import *

# Usage example:
features_to_map = ['readout', 'labelling']
edis = EDISdataset('../new_data/TrainingDataComplete_EDIS.csv', site_id=0, ICV = True, cat_features_to_encode=features_to_map)
helius = HELIUSdataset('../new_data/TrainingDataComplete_HELIUS.csv', site_id=1, ICV = True, cat_features_to_encode=features_to_map)
sabre = SABREdataset('../new_data/TrainingDataComplete_SABRE.csv', site_id=2, ICV = True, cat_features_to_encode=features_to_map)
stroke_mri = StrokeMRIdataset('../new_data/TrainingDataComplete_StrokeMRI.csv', site_id=3, ICV = True, cat_features_to_encode=features_to_map)
top = TOPdataset('../new_data/TrainingDataComplete_TOP.csv', site_id=3, ICV = True, cat_features_to_encode=features_to_map)
insight46 = Insight46dataset('../new_data/TrainingDataComplete_Insight46.csv', site_id=4, ICV = True, cat_features_to_encode=features_to_map)


unharm_features = ['participant_id', 'gm_vol', 'wm_vol', 'csf_vol', 'gm_icvratio', 'gmwm_icvratio', 'wmhvol_wmvol', 'wmh_count']
features_to_harmonize = ['aca_b_cov', 'mca_b_cov', 'pca_b_cov', 'totalgm_b_cov', 
                         'aca_b_cbf', 'mca_b_cbf', 'pca_b_cbf', 'totalgm_b_cbf']
covariates = ['age', 'sex', 'site', 'icv']

harmonizer = HarmNeuroHarmonize(features_to_map, unharm_features, features_to_harmonize, covariates)
harmonized_data = harmonizer.harmonize([edis, helius, sabre, stroke_mri, top, insight46])












#covbat
features_to_map = ['readout', 'labelling']
edis = EDISdataset('../new_data/TrainingDataComplete_EDIS.csv', site_id=0, ICV = True, cat_features_to_encode=features_to_map)
helius = HELIUSdataset('../new_data/TrainingDataComplete_HELIUS.csv', site_id=1, ICV = True, cat_features_to_encode=features_to_map)
sabre = SABREdataset('../new_data/TrainingDataComplete_SABRE.csv', site_id=2, ICV = True, cat_features_to_encode=features_to_map)
stroke_mri = StrokeMRIdataset('../new_data/TrainingDataComplete_StrokeMRI.csv', site_id=3, ICV = True, cat_features_to_encode=features_to_map)
top = TOPdataset('../new_data/TrainingDataComplete_TOP.csv', site_id=3, ICV = True, cat_features_to_encode=features_to_map)
insight46 = Insight46dataset('../new_data/TrainingDataComplete_Insight46.csv', site_id=4, ICV = True, cat_features_to_encode=features_to_map)


to_be_harmonized_or_covar = [
    'Age', 'Sex', 'ACA_B_CoV', 'MCA_B_CoV', 'PCA_B_CoV','DeepWM_B_CoV',
     'ACA_B_CBF', 'MCA_B_CBF', 'PCA_B_CBF', 'LD', 'PLD','DeepWM_B_CBF',
       'Labelling', 'Readout', 'TotalGM_B_CoV',
       'TotalGM_B_CBF',
]
not_harmonized= ['GM_vol', 'WM_vol', 'CSF_vol','GM_ICVRatio', 'GMWM_ICVRatio', 'WMHvol_WMvol', 'WMH_count',
                'LD', 'PLD', 'Labelling',
       'Readout','DeepWM_B_CoV','DeepWM_B_CBF',]

#pheno_features = ['participant_id','Age', 'Sex', 'Site']
pheno_features = ['Age', 'Sex']

harmonizer = HarmCovbat(to_be_harmonized_or_covar,not_harmonized,pheno_features)
harmonized_data = harmonizer.harmonize([edis, helius, sabre, stroke_mri, top, insight46])














features_to_map = ['readout', 'labelling']
edis = EDISdataset('../new_data/TrainingDataComplete_EDIS.csv', site_id=0, ICV = True, cat_features_to_encode=features_to_map)
helius = HELIUSdataset('../new_data/TrainingDataComplete_HELIUS.csv', site_id=1, ICV = True, cat_features_to_encode=features_to_map)
sabre = SABREdataset('../new_data/TrainingDataComplete_SABRE.csv', site_id=2, ICV = True, cat_features_to_encode=features_to_map)
stroke_mri = StrokeMRIdataset('../new_data/TrainingDataComplete_StrokeMRI.csv', site_id=3, ICV = True, cat_features_to_encode=features_to_map)
top = TOPdataset('../new_data/TrainingDataComplete_TOP.csv', site_id=3, ICV = True, cat_features_to_encode=features_to_map)
insight46 = Insight46dataset('../new_data/TrainingDataComplete_Insight46.csv', site_id=4, ICV = True, cat_features_to_encode=features_to_map)

to_be_harmonized_or_covar = [
    'Age', 'Sex', 'DeepWM_B_CoV', 'ACA_B_CoV', 'MCA_B_CoV', 'PCA_B_CoV', 'TotalGM_B_CoV', 'DeepWM_B_CBF', 
    'ACA_B_CBF', 'MCA_B_CBF', 'PCA_B_CBF', 'TotalGM_B_CBF', 'DeepWM_B_CoV', 'DeepWM_B_CBF',]

not_harmonized= [
    'GM_vol', 'WM_vol', 'CSF_vol', 'GM_ICVRatio', 'GMWM_ICVRatio', 'WMHvol_WMvol', 'WMH_count', 'DeepWM_B_CoV', 'DeepWM_B_CBF',]


harmonizer = HarmNeuroCombat(['age'],['sex'],to_be_harmonized_or_covar,not_harmonized)
harmonized_data = harmonizer.harmonize([edis, helius, sabre, stroke_mri, top, insight46])















#autocombat
features_to_map = ['readout', 'labelling']
edis = EDISdataset('../new_data/TrainingDataComplete_EDIS.csv', site_id=0,decade = True, ICV = True, cat_features_to_encode=features_to_map)
helius = HELIUSdataset('../new_data/TrainingDataComplete_HELIUS.csv', site_id=1,decade = True, ICV = True, cat_features_to_encode=features_to_map)
sabre = SABREdataset('../new_data/TrainingDataComplete_SABRE.csv', site_id=2,decade = True, ICV = True, cat_features_to_encode=features_to_map)
stroke_mri = StrokeMRIdataset('../new_data/TrainingDataComplete_StrokeMRI.csv', site_id=3,decade = True, ICV = True, cat_features_to_encode=features_to_map)
top = TOPdataset('../new_data/TrainingDataComplete_TOP.csv', site_id=3,decade = True, ICV = True, cat_features_to_encode=features_to_map)
insight46 = Insight46dataset('../new_data/TrainingDataComplete_Insight46.csv', site_id=4,decade = True, ICV = True, cat_features_to_encode=features_to_map)

features_to_harmonize = ['aca_b_cov', 'mca_b_cov', 'pca_b_cov', 'totalgm_b_cov', 
                         'aca_b_cbf', 'mca_b_cbf', 'pca_b_cbf', 'totalgm_b_cbf']
unharm_features = ['participant_id', 'gm_vol', 'wm_vol', 'csf_vol', 
                   'gm_icvratio', 'gmwm_icvratio', 'wmhvol_wmvol', 'wmh_count']
discrete_covariates = ['sex']
continuous_covariates = ['decade']
harmonizer = HarmAutoCombat(features_to_harmonize,['site'],discrete_covariates,continuous_covariates,unharm_features)
harmonized_data = harmonizer.harmonize([edis, helius, sabre, stroke_mri, top, insight46])

# %%
