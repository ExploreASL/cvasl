import logging
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb


def wrap_title(title, width=80):
    return textwrap.fill(title, width)

def get_normalization(norm_type, num_features, num_groups=8):
    """
    Returns a normalization layer based on the given type.
    Options:
      - 'batch': BatchNorm3d
      - 'group': GroupNorm (with num_groups groups)
      - 'instance': InstanceNorm3d
    """
    if norm_type == "batch":
        return nn.BatchNorm3d(num_features)
    elif norm_type == "group":
        # for GroupNorm, the number of groups is adjustable.
        return nn.GroupNorm(num_groups, num_features)
    elif norm_type == "instance":
        return nn.InstanceNorm3d(num_features)
    else:
        raise ValueError("Normalization type not recognized. Use 'batch', 'group', or 'instance'.")


def create_demographics_table(dataset, wandb_run):
    """Creates and logs a demographic table to wandb."""
    df = dataset.data_df.copy()
    # group by 'Site', 'Sex' and calculate the mean and std for 'Age'
    summary_df = (
        df.groupby(["Site", "Sex"])
        .agg(
            mean_age=pd.NamedAgg(column="Age", aggfunc="mean"),
            std_age=pd.NamedAgg(column="Age", aggfunc="std"),
            count=pd.NamedAgg(column="Age", aggfunc="count"),
        )
        .reset_index()
    )
    # create a wandb table, could be adjusted later with more columns
    demographics_table = wandb.Table(dataframe=summary_df)
    wandb_run.log({"demographics_table": demographics_table})
    logging.info("Demographics table created and logged to wandb.")

def create_prediction_chart(model, test_loader, device, wandb_run):
    """Creates and logs a prediction vs actual chart to wandb."""
    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].unsqueeze(1).to(device)
            ages = batch["age"].unsqueeze(1).to(device)
            demographics = batch["demographics"].to(device)
            outputs = model(images, demographics)
            all_predictions.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(ages.cpu().numpy().flatten())
    plt.figure(figsize=(8, 6))
    plt.scatter(all_targets, all_predictions, alpha=0.5)
    plt.xlabel("Actual Age")
    plt.ylabel("Predicted Age")
    plt.title("Actual vs Predicted Age")
    plt.plot(
        [min(all_targets), max(all_targets)],
        [min(all_targets), max(all_targets)],
        color="red",
    )  # diagonal line fixed
    plt.grid(True)
    plt.tight_layout()
    wandb_run.log({"prediction_chart": plt})
    logging.info("Prediction chart created and logged to wandb.")