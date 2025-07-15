import os
from cvasl.harmonizers import NeuroHarmonize, Covbat, NeuroCombat, NestedComBat, ComscanNeuroCombat
from cvasl.dataset import MRIdataset, encode_cat_features


def load_datasets(shared_datadir):
    input_paths = [os.path.realpath(shared_datadir / "TestingData_Site1_fake.csv"),
                   os.path.realpath(shared_datadir / "TestingData_Site2_fake.csv"),
                   os.path.realpath(shared_datadir / "TrainingData_Site1_fake.csv")]
    input_sites = [1, 2, 1]

    mri_datasets = [MRIdataset(input_path, input_site, "participant_id", features_to_drop=[])
                    for input_site, input_path in zip(input_sites, input_paths) ]
    for mri_dataset in mri_datasets:
        mri_dataset.preprocess()
    features_to_map = ['sex']
    encode_cat_features(mri_datasets, features_to_map)
    return mri_datasets


def test_neurocombat(shared_datadir):
    """Test whether the NeuroCombat harmonizer runs."""

    datasets = load_datasets(shared_datadir)
    features_to_harmonize = ['ACA_B_CoV', 'MCA_B_CoV', 'PCA_B_CoV', 'TotalGM_B_CoV','ACA_B_CBF', 'MCA_B_CBF', 'PCA_B_CBF', 'TotalGM_B_CBF']
    discrete_covariates= ['sex']
    continuous_covariates=  ['age']
    patient_identifier = 'participant_id'
    site_indicator = 'site'

    harmonizer = NeuroCombat(features_to_harmonize, discrete_covariates, continuous_covariates, patient_identifier, site_indicator)
    harmonized_data = harmonizer.harmonize(datasets)
