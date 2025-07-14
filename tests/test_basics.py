from cvasl.harmonizers import NeuroHarmonize, Covbat, NeuroCombat, NestedComBat, ComscanNeuroCombat

def test_neurocombat():
    site_indicator = 'site'
    features_to_harmonize = ['aca_b_cov', 'mca_b_cov', 'pca_b_cov', 'totalgm_b_cov', 
                              'aca_b_cbf', 'mca_b_cbf', 'pca_b_cbf', 'totalgm_b_cbf']
    discrete_covariates = ['sex']
    continuous_covariates = ['age']
    harmonizer = NeuroCombat(features_to_harmonize=features_to_harmonize,
                             discrete_covariates=discrete_covariates,
                             continuous_covariates=continuous_covariates,
                             site_indicator=site_indicator)

