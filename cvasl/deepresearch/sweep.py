import wandb
import subprocess
import yaml
import os
import argparse
import torch
torch.set_float32_matmul_precision("high")

def run_training_process(model_type, config):
    """
    Constructs the command-line arguments for train_extended_g.py and runs it.

    Args:
        model_type (str): The type of model to train.
        config (dict): A dictionary of hyperparameters from wandb.config.
    """

    # Base command - these are the arguments that don't change with the sweep
    base_command = [
        "python", "train_extended_g.py",
        "--model_type", model_type,
        "--csv_file", "./trainingdata/mock_dataset/mock_data.csv",
        "--image_dir", "./trainingdata/mock_dataset/images",
        "--num_epochs", "40",  # Keep low for demonstration, increase for real training
        "--batch_size", "2",
        "--learning_rate", "0.00015",  # or get from config if you sweep it
        "--use_wandb",
        "--use_cuda",
        "--bins", "20",
        "--split_strategy", "stratified_group_sex",
        "--output_dir", "./saved_models_test"
    ]

    # Add model-specific arguments based on config.  Handle defaults explicitly.
    model_args = []
    if model_type == "large":
        model_args = [
            "--large_cnn_use_bn",
            "--large_cnn_use_se",
            "--large_cnn_use_dropout",
            "--large_cnn_dropout_rate", str(config.get("large_cnn_dropout_rate", 0.3)),
            "--large_cnn_layers", str(config.get("large_cnn_layers", 3)),
            "--large_cnn_filters", str(config.get("large_cnn_filters", 16)),
            "--large_cnn_filters_multiplier", str(config.get("large_cnn_filters_multiplier", 2.0))
        ]
    elif model_type == "improved_cnn":
        model_args = [
            "--improved_cnn_use_se",
            "--improved_cnn_dropout_rate", str(config.get("improved_cnn_dropout_rate", 0.4)),
            "--improved_cnn_num_conv_layers", str(config.get("improved_cnn_num_conv_layers", 2)),
            "--improved_cnn_initial_filters", str(config.get("improved_cnn_initial_filters", 32)),
            "--improved_cnn_filters_multiplier", str(config.get("improved_cnn_filters_multiplier", 2.0)),
        ]
    elif model_type == "resnet":
        model_args = [
            "--resnet_layers", *[str(x) for x in config.get("resnet_layers", [1, 2, 1])], # Handle list
            "--resnet_initial_filters", str(config.get("resnet_initial_filters", 16)),
            "--resnet_use_se",
            "--resnet_dropout",
            "--resnet_filters_multiplier", str(config.get("resnet_filters_multiplier", 2.0))
        ]
    elif model_type == "densenet":
        model_args = [
            "--densenet_use_se",
            "--densenet_dropout",
            "--densenet_layers", *[str(x) for x in config.get("densenet_layers", [2, 3])], # Handle list
            "--densenet_growth_rate", str(config.get("densenet_growth_rate", 8)),
            "--densenet_initial_filters", str(config.get("densenet_initial_filters", 16)),
            "--densenet_transition_filters_multiplier", str(config.get("densenet_transition_filters_multiplier", 2.0)),
        ]
    elif model_type == "resnext3d":
        model_args = [
            "--resnext_use_se",
            "--resnext_dropout",
            "--resnext_layers", *[str(x) for x in config.get("resnext_layers", [1, 2, 1])], # Handle list
            "--resnext_initial_filters", str(config.get("resnext_initial_filters", 16)),
            "--resnext_cardinality", str(config.get("resnext_cardinality", 8)),
            "--resnext_bottleneck_width", str(config.get("resnext_bottleneck_width", 2)),
            "--resnext_filters_multiplier", str(config.get("resnext_filters_multiplier", 2.0)),
        ]
    elif model_type == "efficientnet3d":
        model_args = [
            "--efficientnet_dropout", str(config.get("efficientnet_dropout", 0.1)),
            "--efficientnet_width_coefficient", str(config.get("efficientnet_width_coefficient", 0.8)),
            "--efficientnet_depth_coefficient", str(config.get("efficientnet_depth_coefficient", 0.8)),
            "--efficientnet_initial_filters", str(config.get("efficientnet_initial_filters", 8)),
            "--efficientnet_filters_multiplier", str(config.get("efficientnet_filters_multiplier", 1.2))
        ]
    elif model_type == "hybrid_cnn_transformer":
         model_args = [
             "--hybrid_cnn_backbone_type", "resnet",
             "--hybrid_cnn_resnet_layers", *[str(x) for x in config.get("hybrid_cnn_resnet_layers", [1,1,1])],
             "--hybrid_transformer_layers", str(config.get("hybrid_transformer_layers", 1)),
             "--hybrid_transformer_heads", str(config.get("hybrid_transformer_heads", 2))
         ]

    # Combine base command and model-specific arguments
    full_command = base_command + model_args

    # Execute the training script
    subprocess.run(full_command, check=True)


def define_sweep_configuration(model_type):
    """
    Defines the hyperparameter sweep configuration for a given model type.

    Args:
        model_type (str):  The type of model ('large', 'resnet', etc.)

    Returns:
        dict:  The wandb sweep configuration.
    """

    sweep_config = {
        'method': 'bayes',  # Bayesian optimization
        'metric': {
            'name': 'final_test_mae',
            'goal': 'minimize'
        },
        'parameters': {}
    }

    # Model-specific hyperparameter ranges (informed by model analysis)
    if model_type == "large":
      sweep_config['parameters'].update({
          'large_cnn_dropout_rate': {'distribution': 'uniform', 'min': 0.1, 'max': 0.5},
          'large_cnn_layers': {'values': [2, 3, 4]},
          'large_cnn_filters': {'values': [8, 16, 32, 64]},
          'large_cnn_filters_multiplier': {'distribution': 'uniform', 'min': 1.0, 'max': 3.0}
      })

    elif model_type == "improved_cnn":
        sweep_config['parameters'].update({
            'improved_cnn_dropout_rate': {'distribution': 'uniform', 'min': 0.1, 'max': 0.6},
            'improved_cnn_num_conv_layers': {'values': [2, 3, 4]},
            'improved_cnn_initial_filters': {'values': [16, 32, 64]},
            'improved_cnn_filters_multiplier': {'distribution': 'uniform', 'min': 1.0, 'max': 3.0}

        })

    elif model_type == "resnet":
        sweep_config['parameters'].update({
            'resnet_layers': {
                'values': [[1, 1, 1], [2, 2, 2], [3, 2, 1], [1, 2, 3]]  # Different layer combinations
            },
            'resnet_initial_filters': {'values': [16, 32, 64]},
            'resnet_filters_multiplier': {'distribution':'uniform', 'min': 1.0, 'max': 3.0}

        })

    elif model_type == "densenet":
        sweep_config['parameters'].update({
            'densenet_layers': {
                'values': [[2, 2], [3, 3], [4, 4], [2, 3], [3, 4]]  # Different layer configs
            },
            'densenet_growth_rate': {'values': [8, 12, 16, 24]},
            'densenet_initial_filters': {'values': [16, 32, 64]},
            'densenet_transition_filters_multiplier': {'distribution': 'uniform', 'min': 1.0, 'max': 3.0}
        })

    elif model_type == "resnext3d":
      sweep_config['parameters'].update({
          'resnext_layers': {
              'values': [[1, 1, 1], [2, 2, 2], [3, 2, 1], [1, 2, 3]]
          },
          'resnext_initial_filters': {'values': [16, 32, 64]},
          'resnext_cardinality': {'values': [8, 16, 32]},
          'resnext_bottleneck_width': {'values': [2, 4, 8]},
          'resnext_filters_multiplier': {'distribution': 'uniform', 'min': 1.0, 'max': 3.0}
      })
    elif model_type == "efficientnet3d":
        sweep_config['parameters'].update({
            'efficientnet_dropout': {'distribution': 'uniform', 'min': 0.0, 'max': 0.5},  # Wider range
            'efficientnet_width_coefficient': {'distribution': 'uniform', 'min': 0.8, 'max': 1.5},
            'efficientnet_depth_coefficient': {'distribution': 'uniform', 'min': 0.8, 'max': 1.5},
            'efficientnet_initial_filters': {'values': [8, 16, 32]},
            'efficientnet_filters_multiplier': {'distribution': 'uniform', 'min': 1.0, 'max': 2.0}
        })

    elif model_type == "hybrid_cnn_transformer":
        sweep_config['parameters'].update({
            'hybrid_cnn_resnet_layers':{
                'values':[[1,1,1], [2,2,2], [3,2,1]]
            },
            'hybrid_transformer_layers': {'values': [1, 2, 3]},
            'hybrid_transformer_heads': {'values': [2, 4, 8]}
        })
    return sweep_config




def main():
    parser = argparse.ArgumentParser(description="Run WandB Sweeps for Brain Age Prediction")
    parser.add_argument("--model_type", type=str, required=True,
                        choices=["large", "improved_cnn", "resnet", "densenet", "resnext3d", "efficientnet3d", "hybrid_cnn_transformer"],
                        help="Type of model to sweep")
    parser.add_argument("--count", type=int, default=20,  # Reduced count for demonstration
                        help="Number of runs for the sweep agent")
    parser.add_argument("--project", type=str, default="asl-brainage-sweeps",
                        help="WandB project name")

    args = parser.parse_args()

    sweep_config = define_sweep_configuration(args.model_type)

    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project=args.project)

    # Define the training function to be used by the sweep agent.  This
    # function *must* take no arguments, and get all configuration
    # information from `wandb.config`.
    def train_func():
        with wandb.init() as run:
            run_training_process(args.model_type, run.config)

    # Run the sweep agent
    wandb.agent(sweep_id, function=train_func, count=args.count)

if __name__ == "__main__":
    main()