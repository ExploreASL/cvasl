import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchviz import make_dot
import torchview
import os
import sys
from PIL import Image

# Assuming the model class is in the same directory
sys.path.append('.')
from models.cnn import Large3DCNN

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchviz import make_dot
import torchview
import os
import sys
import re
from PIL import Image

# Assuming the model class is in the same directory
sys.path.append('.')


def parse_model_name(model_name):
    """
    Parse parameters from the model name string.
    """
    # Extract core parameters using regex
    layers_match = re.search(r'_layers(\d+)', model_name)
    filters_match = re.search(r'_filters(\d+)', model_name)
    filtermult_match = re.search(r'_filtermult([\d\.]+)', model_name)
    dropout_match = re.search(r'_dropout([\d\.]+)', model_name)
    
    if not all([layers_match, filters_match, filtermult_match, dropout_match]):
        raise ValueError(f"Could not parse all required parameters from model name: {model_name}")
    
    params = {
        "num_conv_layers": int(layers_match.group(1)),
        "initial_filters": int(filters_match.group(1)),
        "filters_multiplier": float(filtermult_match.group(1)),
        "use_bn": "_BN" in model_name,
        "use_se": "_SE" in model_name,
        "use_dropout": "_DO" in model_name,
        "dropout_rate": float(dropout_match.group(1)),
        "use_demographics": "_with_demographics" in model_name
    }
    
    return params

def visualize_model_torchviz(model, input_shape, demographics_shape, filename='model_visualization'):
    """
    Visualize the model architecture using torchviz and save as PNG.
    """
    # Create dummy input tensors
    x = torch.randn(input_shape)
    demographics = torch.randn(demographics_shape)
    
    # Forward pass
    y = model(x, demographics)
    
    # Generate visualization
    dot = make_dot(y, params=dict(model.named_parameters()))
    dot.format = 'png'
    dot.render(filename, cleanup=True)
    print(f"Model visualization saved to {filename}.png")

def visualize_model_torchview(model, input_shape, demographics_shape, filename='model_structure'):
    """
    Visualize the model architecture using torchview and save as PNG.
    """
    # Create model graph
    graph = torchview.draw_graph(
        model, 
        input_data=[torch.zeros(input_shape), torch.zeros(demographics_shape)],
        save_graph=True,
        directory=".",
        filename=filename,
        expand_nested=True
    )
    print(f"Model structure visualization saved to {filename}.png")

def visualize_model_tensorboard(model, input_shape, demographics_shape, log_dir='./logs'):
    """
    Visualize the model architecture using TensorBoard.
    """
    # Create writer and directory
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    # Create dummy input tensors
    x = torch.randn(input_shape)
    demographics = torch.randn(demographics_shape)
    
    # Add model graph to TensorBoard
    writer.add_graph(model, [x, demographics])
    writer.close()
    print(f"Model visualization saved to TensorBoard log directory: {log_dir}")

def create_model(params):
    """
    Create a model instance based on the parsed parameters.
    """
    return Large3DCNN(
        num_demographics=5,  # Adjust based on your actual demographics data dimension
        num_conv_layers=params["num_conv_layers"],
        initial_filters=params["initial_filters"],
        filters_multiplier=params["filters_multiplier"],
        use_bn=params["use_bn"],
        use_dropout=params["use_dropout"],
        use_se=params["use_se"],
        dropout_rate=params["dropout_rate"],
        use_demographics=params["use_demographics"]
    )

def format_model_name_for_filename(model_name):
    """
    Create a clean filename from the model name.
    """
    # Extract the core model identifier (without date/time stamp)
    match = re.search(r'(Large3DCNN_layers\d+_filters\d+_filtermult[\d\.]+.*?)_\d{4}-\d{2}-\d{2}', model_name)
    if match:
        return match.group(1)
    return model_name.replace('_', '-')  # Fallback if pattern doesn't match

def visualize_all_models(model_names):
    """
    Visualize all models in the provided list.
    """
    # Input shapes - same for all models
    batch_size = 1
    input_shape = (batch_size, 1, 120, 144, 120)  # Adjust based on your actual input dimensions
    num_demographics = 5  # Adjust based on your actual demographics data dimension
    demographics_shape = (batch_size, num_demographics)
    
    # Process each model
    for model_name in model_names:
        # Clean the model name if it starts with underscore
        if model_name.startswith('_'):
            model_name = model_name[1:]
            
        print(f"\n{'='*80}\nProcessing model: {model_name}\n{'='*80}")
        
        try:
            # Parse the model parameters
            params = parse_model_name(model_name)
            
            # Print extracted parameters for confirmation
            print(f"Extracted model parameters:")
            for key, value in params.items():
                print(f"{key}: {value}")
            
            # Create the model
            model = create_model(params)
            
            # Create filename base
            filename_base = format_model_name_for_filename(model_name)
            
            # Create output directory for this model
            model_dir = f"visualizations/{filename_base}"
            os.makedirs(model_dir, exist_ok=True)
            
            # Apply each visualization method
            try:
                visualize_model_torchviz(model, input_shape, demographics_shape, f"{model_dir}/torchviz")
            except Exception as e:
                print(f"Torchviz visualization failed: {e}")
            
            try:
                visualize_model_torchview(model, input_shape, demographics_shape, f"{model_dir}/torchview")
            except Exception as e:
                print(f"Torchview visualization failed: {e}")
            
            try:
                visualize_model_tensorboard(model, input_shape, demographics_shape, f"{model_dir}/tensorboard")
            except Exception as e:
                print(f"TensorBoard visualization failed: {e}")
                
            print(f"Completed visualization for {model_name}")
            
        except Exception as e:
            print(f"Failed to process model {model_name}: {e}")

if __name__ == "__main__":
    # List of models to visualize
    model_names = [
        "_Large3DCNN_layers4_filters20_filtermult1.5_BN_SE_DO_dropout0.05_without_demographics_lr1.5e_04_epochs120_bs5_split-stratified_group_sex_test0.2_bins-20_2025-02-19_15-33-49",
        "_Large3DCNN_layers4_filters20_filtermult2.5_BN_SE_DO_dropout0.05_without_demographics_lr1.5e_04_epochs120_bs5_split-stratified_group_sex_test0.2_bins-20_2025-02-19_15-48-33",
        "_Large3DCNN_layers4_filters64_filtermult1.5_BN_SE_DO_dropout0.05_without_demographics_lr1.5e_04_epochs120_bs5_split-stratified_group_sex_test0.2_bins-20_2025-02-19_16-56-13",
        "_Large3DCNN_layers5_filters64_filtermult2.5_BN_SE_DO_dropout0.05_without_demographics_lr1.5e_04_epochs120_bs5_split-stratified_group_sex_test0.2_bins-20_2025-02-19_20-12-00",
        "_Large3DCNN_layers6_filters8_filtermult2.5_BN_SE_DO_dropout0.25_without_demographics_lr1.5e_04_epochs120_bs5_split-stratified_group_sex_test0.2_bins-20_2025-02-20_08-18-23"
    ]
    
    # Create main output directory
    os.makedirs("visualizations", exist_ok=True)
    
    # Process all models
    visualize_all_models(model_names)
    
    print("\nAll model visualizations complete!")
    print("To view TensorBoard visualizations, run: tensorboard --logdir=./visualizations")