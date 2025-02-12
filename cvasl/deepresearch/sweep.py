import optuna
import subprocess
import re
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

LOG_FILE = "optuna_sweep.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a'),
        logging.StreamHandler()
    ]
)
logging.info(f"Starting Optuna Sweep. Logs will be saved to {LOG_FILE}")

TRIAL_RESULTS = []

def objective(trial):
    
    model_type = trial.suggest_categorical("model_type", ["large", "resnet", "densenet", "efficientnet3d", "improved_cnn", "resnext3d"])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 12, 20])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    split_strategy = "stratified_group_sex"

    
    model_params = {}
    if model_type == "large":
        model_params['large_cnn_layers'] = large_cnn_layers = trial.suggest_int("large_cnn_layers", 1, 6)
        model_params['large_cnn_filters'] = large_cnn_filters = trial.suggest_int("large_cnn_filters", 8, 64)
        model_params['large_cnn_use_se'] = large_cnn_use_se = trial.suggest_categorical("large_cnn_use_se", [True, False])
        model_params['large_cnn_dropout_rate'] = large_cnn_dropout_rate = trial.suggest_float("large_cnn_dropout_rate", 0.05, 0.4)
        model_params['large_cnn_use_bn'] = large_cnn_use_bn = trial.suggest_categorical("large_cnn_use_bn", [True, False])
        model_params['large_cnn_use_dropout'] = large_cnn_use_dropout = trial.suggest_categorical("large_cnn_use_dropout", [True, False])

    elif model_type == "resnet":
        model_params['resnet_layers_l1'] = resnet_layers_l1 = trial.suggest_int("resnet_layers_l1", 1, 5)
        model_params['resnet_layers_l2'] = resnet_layers_l2 = trial.suggest_int("resnet_layers_l2", 1, 5)
        model_params['resnet_layers_l3'] = resnet_layers_l3 = trial.suggest_int("resnet_layers_l3", 1, 5)
        model_params['resnet_initial_filters'] = resnet_initial_filters = trial.suggest_int("resnet_initial_filters", 8, 64)
        model_params['resnet_use_se'] = resnet_use_se = trial.suggest_categorical("resnet_use_se", [True, False])
        model_params['resnet_dropout'] = resnet_dropout = trial.suggest_categorical("resnet_dropout", [True, False])

    elif model_type == "densenet":
        model_params['densenet_layers_l1'] = densenet_layers_l1 = trial.suggest_int("densenet_layers_l1", 1, 5)
        model_params['densenet_layers_l2'] = densenet_layers_l2 = trial.suggest_int("densenet_layers_l2", 1, 5)
        model_params['densenet_growth_rate'] = densenet_growth_rate = trial.suggest_int("densenet_growth_rate", 4, 48)
        model_params['densenet_use_se'] = densenet_use_se = trial.suggest_categorical("densenet_use_se", [True, False])
        model_params['densenet_transition_filters_multiplier'] = densenet_transition_filters_multiplier = trial.suggest_float("densenet_transition_filters_multiplier", 1.1, 3.5) # Added range for multiplier
        model_params['densenet_dropout'] = densenet_dropout = trial.suggest_categorical("densenet_dropout", [True, False])

    elif model_type == "efficientnet3d":
        model_params['efficientnet_width_coefficient'] = efficientnet_width_coefficient = trial.suggest_float("efficientnet_width_coefficient", 0.5, 2.1) # Slightly narrower width range
        model_params['efficientnet_depth_coefficient'] = efficientnet_depth_coefficient = trial.suggest_float("efficientnet_depth_coefficient", 0.5, 2.1) # Slightly narrower depth range
        model_params['efficientnet_dropout'] = efficientnet_dropout = trial.suggest_float("efficientnet_dropout", 0.05, 0.5)
        model_params['efficientnet_initial_filters'] = efficientnet_initial_filters = trial.suggest_int("efficientnet_initial_filters", 8, 64)

    elif model_type == "improved_cnn":
        model_params['improved_cnn_num_conv_layers'] = improved_cnn_num_conv_layers = trial.suggest_int("improved_cnn_num_conv_layers", 1, 6)
        model_params['improved_cnn_initial_filters'] = improved_cnn_initial_filters = trial.suggest_int("improved_cnn_initial_filters", 8, 64)
        model_params['improved_cnn_dropout_rate'] = improved_cnn_dropout_rate = trial.suggest_float("improved_cnn_dropout_rate", 0.05, 0.5)
        model_params['improved_cnn_use_se'] = improved_cnn_use_se = trial.suggest_categorical("improved_cnn_use_se", [True, False])
        model_params['improved_cnn_filters_multiplier'] = improved_cnn_filters_multiplier = trial.suggest_float("improved_cnn_filters_multiplier", 1.1, 3.2)

    elif model_type == "resnext3d":
        model_params['resnext_cardinality'] = resnext_cardinality = trial.suggest_categorical("resnext_cardinality", [16, 24, 32, 48, 64])
        model_params['resnext_bottleneck_width'] = resnext_bottleneck_width = trial.suggest_categorical("resnext_bottleneck_width", [2, 4, 6, 8, 10, 12])
        model_params['resnext_layers_l1'] = resnext_layers_l1 = trial.suggest_int("resnext_layers_l1", 1, 5)
        model_params['resnext_layers_l2'] = resnext_layers_l2 = trial.suggest_int("resnext_layers_l2", 1, 5)
        model_params['resnext_layers_l3'] = resnext_layers_l3 = trial.suggest_int("resnext_layers_l3", 1, 5)
        model_params['resnext_use_se'] = resnext_use_se = trial.suggest_categorical("resnext_use_se", [True, False])
        model_params['resnext_dropout'] = resnext_dropout = trial.suggest_categorical("resnext_dropout", [True, False])
        model_params['resnext_initial_filters'] = resnext_initial_filters = trial.suggest_int("resnext_initial_filters", 8, 78)


    output_file = f"trial_{trial.number}_train.log"
    command = [
        "python", "train.py",
        "--model_type", model_type,
        "--csv_file", "./trainingdata/mock_dataset/mock_data.csv",
        "--image_dir", "./trainingdata/mock_dataset/images",
        "--num_epochs", "10",
        "--batch_size", str(batch_size),
        "--learning_rate", str(learning_rate),
        "--weight_decay", str(weight_decay),
        "--split_strategy", split_strategy,
        "--bins", "2",
        "--output_dir", "./optuna_saved_models",
        "--use_cuda",
        "--use_wandb",
        "--wandb_prefix", "SWEEP_",
    ]

    
    if model_type == "large":
        command.extend([
            "--large_cnn_layers", str(large_cnn_layers),
            "--large_cnn_filters", str(large_cnn_filters),
        ])
        if large_cnn_use_bn:
            command.append("--large_cnn_use_bn")
        if large_cnn_use_se:
            command.append("--large_cnn_use_se")
        if large_cnn_use_dropout:
            command.append("--large_cnn_use_dropout")
        command.extend([
            "--large_cnn_dropout_rate", str(large_cnn_dropout_rate)
        ])
    elif model_type == "resnet":
        command.extend([
            "--resnet_layers", str(resnet_layers_l1), str(resnet_layers_l2), str(resnet_layers_l3),
            "--resnet_initial_filters", str(resnet_initial_filters),
        ])
        if resnet_use_se:
            command.append("--resnet_use_se")
        if resnet_dropout:
            command.append("--resnet_dropout")
    elif model_type == "densenet":
        command.extend([
            "--densenet_layers", str(densenet_layers_l1), str(densenet_layers_l2),
            "--densenet_growth_rate", str(densenet_growth_rate),
            "--densenet_transition_filters_multiplier", str(densenet_transition_filters_multiplier)
        ])
        if densenet_use_se:
            command.append("--densenet_use_se")
        if densenet_dropout:
            command.append("--densenet_dropout")
    elif model_type == "efficientnet3d":
        command.extend([
            "--efficientnet_width_coefficient", str(efficientnet_width_coefficient),
            "--efficientnet_depth_coefficient", str(efficientnet_depth_coefficient),
            "--efficientnet_dropout", str(efficientnet_dropout),
            "--efficientnet_initial_filters", str(efficientnet_initial_filters),
        ])
    elif model_type == "improved_cnn":
        command.extend([
            "--improved_cnn_num_conv_layers", str(improved_cnn_num_conv_layers),
            "--improved_cnn_initial_filters", str(improved_cnn_initial_filters),
            "--improved_cnn_dropout_rate", str(improved_cnn_dropout_rate),
        ])
        if improved_cnn_use_se:
            command.append("--improved_cnn_use_se")
        command.extend([
            "--improved_cnn_filters_multiplier", str(improved_cnn_filters_multiplier),
        ])
    elif model_type == "resnext3d":
        command.extend([
            "--resnext_cardinality", str(resnext_cardinality),
            "--resnext_bottleneck_width", str(resnext_bottleneck_width),
            "--resnext_layers", str(resnext_layers_l1), str(resnext_layers_l2), str(resnext_layers_l3),
            "--resnext_initial_filters", str(resnext_initial_filters),
        ])
        if resnext_use_se:
            command.append("--resnext_use_se")
        if resnext_dropout:
            command.append("--resnext_dropout")


    logging.info(f"Starting trial {trial.number} with model {model_type} and command: {' '.join(command)}")

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    output_str = stdout.decode('utf-8')
    error_str = stderr.decode('utf-8')

    trial_log_path = os.path.join("./optuna_saved_models", output_file)
    with open(trial_log_path, "w") as f_trial_log:
        f_trial_log.write("STDOUT:\n")
        f_trial_log.write(output_str)
        f_trial_log.write("\nSTDERR:\n")
        f_trial_log.write(error_str)

    if process.returncode != 0:
        logging.error(f"train.py FAILED for trial {trial.number}. See trial log: {trial_log_path}")
        logging.error(error_str)
        raise optuna.TrialPruned()  #prune trial if training fails

    
    mae_match = re.search(r"Epoch \d+ Test MAE: ([0-9.]+)", output_str)
    if mae_match:
        mae = float(mae_match.group(1))
        logging.info(f"Trial {trial.number} finished with MAE: {mae}")

        
        trial_result = {
            'trial_number': trial.number,
            'model_type': model_type,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'weight_decay': weight_decay,
            'mae': mae,
            **model_params # model-specific parameters
        }
        TRIAL_RESULTS.append(trial_result)

        
        trial_results_df = pd.DataFrame(TRIAL_RESULTS)
        create_parameter_charts(trial_results_df)
        logging.info("Parameter interplay charts updated after trial.")

        return mae
    else:
        logging.error("MAE not found in train.py output. See trial log: {trial_log_path}")
        logging.error(output_str)
        raise Exception("MAE not found in train.py output")


def create_parameter_charts(study_results_df, output_dir="./optuna_charts"):
    os.makedirs(output_dir, exist_ok=True)
    model_types = study_results_df['model_type'].unique()

    for model in model_types:
        model_df = study_results_df[study_results_df['model_type'] == model]
        if model_df.empty:
            logging.warning(f"No data to create charts for model type: {model}")
            continue

        output_prefix = os.path.join(output_dir, f"{model}_param_interplay")

        
        numerical_params = [col for col in model_df.columns if model_df[col].dtype in ['float64', 'int64'] and col not in ['trial_number', 'mae']] 
        for param in numerical_params:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=param, y='mae', data=model_df)
            plt.title(f'{model} - MAE vs. {param}')
            plt.xlabel(param)
            plt.ylabel('Test MAE')
            chart_path = f"{output_prefix}_MAE_vs_{param}.png"
            plt.savefig(chart_path)
            plt.close()
            logging.info(f"Saved/Updated MAE vs. {param} chart for {model} to {chart_path}")

        try:
            numerical_cols_for_pairplot = numerical_params + ['mae']
            pair_plot_df = model_df[numerical_cols_for_pairplot]
            if not pair_plot_df.empty:
                pair_plot = sns.pairplot(pair_plot_df)
                pair_plot.fig.suptitle(f"{model} - Pair Plot of Numerical Parameters vs. MAE", y=1.03)
                chart_path = f"{output_prefix}_pairplot.png"
                pair_plot.savefig(chart_path)
                plt.close()
                logging.info(f"Saved/Updated Pair Plot for {model} to {chart_path}")
        except Exception as e:
            logging.warning(f"Pair plot generation failed for {model}: {e}")


if __name__ == "__main__":
    sampler = optuna.samplers.TPESampler() # TPESampler for Bayesian optimization
    study = optuna.create_study(direction="minimize", sampler=sampler)  # Minimize MAE, using Bayesian TPE sampler. Could use MAPE if we use it as loss in the actual training
    try:
        study.optimize(objective, n_trials=10000)
    except Exception as e:
        logging.error(f"Optuna study interrupted due to an error: {e}")

    logging.info("Optimization finished.")
    logging.info("Best trial:")
    trial = study.best_trial
    logging.info(f"  Value (Best Test MAE): {trial.value}")
    logging.info("  Params: ")
    for key, value in trial.params.items():
        logging.info(f"    {key}: {value}")

    
    results_df = pd.DataFrame(TRIAL_RESULTS)
    results_csv_path = os.path.join("./optuna_saved_models", "optuna_trial_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    logging.info(f"Trial results saved to {results_csv_path}")


    
    logging.info("Final Parameter interplay charts creation after all trials...")
    create_parameter_charts(results_df)
    logging.info("Parameter interplay charts created and saved/updated in ./optuna_charts/")


    
    with open("best_params.json", "w") as f:
        json.dump(trial.params, f, indent=4)

    print("Optuna sweep completed. Check optuna_sweep.log for full logs, and optuna_charts directory for parameter charts.")