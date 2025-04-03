import argparse
import pandas as pd
import sys
import os
import numpy as np
from cvasl.dataset import MRIdataset, encode_cat_features

# Import scikit-learn models directly
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import xgboost as XGBRegressor  # Import if available
from cvasl.prediction import PredictBrainAge  # Assuming this class exists in your codebase

def main():
    parser = argparse.ArgumentParser(description="Run predictions on MRI datasets.")
    
    # Data parameters
    parser.add_argument("--dataset_paths", required=True, 
                      help="Comma-separated paths to dataset CSV files. For datasets with multiple paths, use semicolon to separate paths, e.g. 'path1;path2',path3,path4")
    parser.add_argument("--site_ids", required=True, 
                      help="Comma-separated site IDs for each dataset.")
    parser.add_argument("--patient_identifier", default="participant_id", 
                      help="Column name for patient identifier.")
    parser.add_argument("--features_to_drop", default="m0,id", 
                      help="Comma-separated list of features to drop.")
    parser.add_argument("--cat_features_to_encode", default="readout,labelling,sex", 
                      help="Comma-separated list of categorical features to encode.")
    parser.add_argument("--decade", type=str, default="True", 
                      help="Whether to add decade-related features (True/False).")
    parser.add_argument("--icv", type=str, default="True", 
                      help="Whether to include intracranial volume as a feature (True/False).")
    parser.add_argument("--features", default="aca_b_cbf,aca_b_cov,csf_vol,gm_icvratio,gm_vol,gmwm_icvratio,mca_b_cbf,mca_b_cov,pca_b_cbf,pca_b_cov,totalgm_b_cbf,totalgm_b_cov,wm_vol,wmh_count,wmhvol_wmvol", 
                      help="Comma-separated list of features to use for prediction.")
    
    # Prediction parameters
    parser.add_argument("--method", required=True, choices=["random_forest", "extra_trees", "svm", "xgboost", "neural_network", "gradient_boosting"], 
                      help="Prediction method to use.")
    parser.add_argument("--target_variable", required=True, 
                      help="Target variable to predict.")
    parser.add_argument("--test_size", type=float, default=0.2, 
                      help="Proportion of data to use for testing (0.0-1.0).")
    parser.add_argument("--random_state", type=int, default=42, 
                      help="Random seed for reproducibility.")
    parser.add_argument("--splits", type=int, default=5, 
                      help="Number of cross-validation folds.")
    parser.add_argument("--n_bins", type=int, default=2, 
                      help="Number of bins for categorical data.")
    parser.add_argument("--cat_category", default="sex", 
                      help="Categorical category for stratification.")
    parser.add_argument("--cont_category", default="age", 
                      help="Continuous category for stratification.")
    parser.add_argument("--output_path", default="./prediction_results.csv", 
                      help="Path to save prediction results.")
    
    # Method-specific arguments
    # Random Forest specific arguments
    parser.add_argument("--n_estimators", type=int, default=100, 
                      help="Number of trees in the forest.")
    parser.add_argument("--max_depth", type=int, default=None, 
                      help="Maximum depth of the trees.")
    parser.add_argument("--criterion", default="absolute_error", choices=["absolute_error", "squared_error", "friedman_mse", "poisson"], 
                      help="The function to measure the quality of a split.")
    parser.add_argument("--min_samples_split", type=int, default=2, 
                      help="The minimum number of samples required to split an internal node.")
    parser.add_argument("--min_samples_leaf", type=int, default=1, 
                      help="The minimum number of samples required to be at a leaf node.")
    parser.add_argument("--max_features", default="log2", 
                      help="The number of features to consider when looking for the best split.")
    parser.add_argument("--bootstrap", type=str, default="False", 
                      help="Whether bootstrap samples are used when building trees.")
    parser.add_argument("--n_jobs", type=int, default=-1, 
                      help="The number of jobs to run in parallel.")
    parser.add_argument("--warm_start", type=str, default="True", 
                      help="Whether to reuse the solution of the previous call.")
    
    # SVM specific arguments
    parser.add_argument("--svm_kernel", default="rbf", choices=["linear", "poly", "rbf", "sigmoid"], 
                      help="Kernel type for SVM.")
    parser.add_argument("--svm_C", type=float, default=1.0, 
                      help="Regularization parameter for SVM.")
    
    # Neural Network specific arguments
    parser.add_argument("--nn_hidden_layers", default="100,50", 
                      help="Comma-separated list of hidden layer sizes.")
    parser.add_argument("--nn_activation", default="relu", 
                      help="Activation function for neural network.")
    parser.add_argument("--nn_learning_rate", type=float, default=0.001, 
                      help="Learning rate for neural network.")
    parser.add_argument("--nn_epochs", type=int, default=100, 
                      help="Number of training epochs for neural network.")
    
    args = parser.parse_args()
    
    try:
        # Process arguments
        dataset_paths = [x.split(';') for x in args.dataset_paths.split(',')]
        site_ids = [int(x) for x in args.site_ids.split(',')]
        features_to_drop = args.features_to_drop.split(',') if args.features_to_drop else []
        cat_features_to_encode = args.cat_features_to_encode.split(',') if args.cat_features_to_encode else []
        decade = args.decade.lower() == 'true'
        icv = args.icv.lower() == 'true'
        prediction_features = args.features.split(',') if args.features else []
        target = [args.target_variable]
        bootstrap = args.bootstrap.lower() == 'true'
        warm_start = args.warm_start.lower() == 'true'
        
        # Check that site_ids match dataset_paths in length
        if len(site_ids) != len(dataset_paths):
            print(f"Error: Number of site IDs ({len(site_ids)}) does not match number of datasets ({len(dataset_paths)})")
            sys.exit(1)
        
        # Create MRIdataset objects
        datasets = []
        for i, paths in enumerate(dataset_paths):
            ds = MRIdataset(
                path=paths,
                site_id=site_ids[i],
                patient_identifier=args.patient_identifier,
                features_to_drop=features_to_drop,
                cat_features_to_encode=cat_features_to_encode if i == 0 else None,
                decade=decade,
                ICV=icv
            )
            datasets.append(ds)

        # Preprocess datasets
        [_d.preprocess() for _d in datasets]
        datasets = encode_cat_features(datasets, cat_features_to_encode)
        
        # Create model based on method choice
        model = None
        if args.method == "random_forest":
            model = RandomForestRegressor(
                n_estimators=args.n_estimators, 
                max_depth=args.max_depth,
                random_state=args.random_state,
                criterion=args.criterion,
                min_samples_split=args.min_samples_split,
                min_samples_leaf=args.min_samples_leaf,
                max_features=args.max_features,
                bootstrap=bootstrap,
                n_jobs=args.n_jobs,
                warm_start=warm_start
            )
        elif args.method == "extra_trees":
            model = ExtraTreesRegressor(
                n_estimators=args.n_estimators, 
                max_depth=args.max_depth,
                random_state=args.random_state,
                criterion=args.criterion,
                min_samples_split=args.min_samples_split,
                min_samples_leaf=args.min_samples_leaf,
                max_features=args.max_features,
                bootstrap=bootstrap,
                n_jobs=args.n_jobs,
                warm_start=warm_start
            )
        elif args.method == "svm":
            model = SVR(
                kernel=args.svm_kernel,
                C=args.svm_C,
                gamma='auto'
            )
        elif args.method == "xgboost":
            model = XGBRegressor.XGBRegressor(
                n_estimators=args.n_estimators,
                max_depth=args.max_depth if args.max_depth else 6,
                learning_rate=args.nn_learning_rate,
                random_state=args.random_state
            )
        elif args.method == "neural_network":
            hidden_layers = [int(x) for x in args.nn_hidden_layers.split(',')]
            model = MLPRegressor(
                hidden_layer_sizes=tuple(hidden_layers),
                activation=args.nn_activation,
                learning_rate_init=args.nn_learning_rate,
                max_iter=args.nn_epochs,
                random_state=args.random_state
            )
        elif args.method == "gradient_boosting":
            model = GradientBoostingRegressor(
                n_estimators=args.n_estimators,
                max_depth=args.max_depth if args.max_depth else 3,
                learning_rate=args.nn_learning_rate,
                random_state=args.random_state
            )
            
        # Create predictor instance and predict
        if not model:
            print(f"Error: Invalid prediction method: {args.method}")
            sys.exit(1)
            
        # Use 80% of datasets as training and 20% as validation
        train_size = int(len(datasets) * 0.8)
        if train_size == 0:
            train_size = 1  # Ensure at least one dataset for training
            
        training_datasets = datasets[:train_size]
        validation_datasets = datasets[train_size:] if train_size < len(datasets) else []
        
        predictor = PredictBrainAge(
            model_name=args.method,
            model_file_name=args.method,
            model=model,
            datasets=training_datasets,
            datasets_validation=validation_datasets,
            features=prediction_features,
            target=target,
            cat_category=args.cat_category,
            cont_category=args.cont_category,
            n_bins=args.n_bins,
            splits=args.splits,
            test_size_p=args.test_size,
            random_state=args.random_state
        )
        
        # Run prediction
        metrics_df, metrics_df_val, predictions_df, predictions_df_val, models = predictor.predict()
        
        # Save results
        result_data = pd.concat([predictions_df, predictions_df_val]) if predictions_df_val is not None else predictions_df
        result_data.to_csv(args.output_path, index=False)
        
        # Save metrics if available
        if metrics_df is not None:
            metrics_path = args.output_path.replace('.csv', '_metrics.csv')
            metrics_df.to_csv(metrics_path, index=False)
            
        if metrics_df_val is not None:
            val_metrics_path = args.output_path.replace('.csv', '_val_metrics.csv')
            metrics_df_val.to_csv(val_metrics_path, index=False)
        
        print(f"Prediction completed successfully. Results saved to {args.output_path}")
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()