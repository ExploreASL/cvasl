import argparse
import datetime
import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import wandb
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from .data import BrainAgeDataset
from .models.cnn import Large3DCNN
from .models.densenet3d import DenseNet3D
from .models.efficientnet3d import EfficientNet3D
from .models.improvedcnn3d import Improved3DCNN
from .models.loss import BrainAgeLoss
from .models.resnet3d import ResNet3D
from .models.resnext3d import ResNeXt3D
from .models.vit3d import VisionTransformer3D
from .utils import create_demographics_table, create_prediction_chart

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

torch.manual_seed(42)
np.random.seed(42)

def verify_brain_masking(dataset, output_dir=None, num_samples=5):
    """
    Verifies brain masking by plotting the middle slice in three orthogonal views 
    for the first few images in the dataset immediately after they're loaded.
    
    Args:
        dataset: BrainAgeDataset or list of samples
        output_dir: Directory to save the verification plots (defaults to './mask_verification')
        num_samples: Number of samples to check
    """
    import os

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap
    
    if output_dir is None:
        output_dir = './mask_verification'
    os.makedirs(output_dir, exist_ok=True)
    
    view_names = ['sagittal', 'coronal', 'axial']
    view_axes = {'sagittal': 0, 'coronal': 1, 'axial': 2}
    
    # Create a custom colormap with true black for zeros
    custom_gray = plt.cm.gray(np.arange(256))
    custom_gray[0] = [0, 0, 0, 1]  # Pure black for zero values
    custom_cmap = ListedColormap(custom_gray)
    
    # Determine if we're dealing with a dataset or a list
    if hasattr(dataset, '__getitem__') and not isinstance(dataset, list):
        sample_getter = lambda i: dataset[i]
        n_samples = min(num_samples, len(dataset))
    else:
        sample_getter = lambda i: dataset[i]
        n_samples = min(num_samples, len(dataset))
    
    for i in range(n_samples):
        sample = sample_getter(i)
        if sample is None:
            logging.info(f"Sample {i} is None, skipping")
            continue
            
        image = sample['image'].numpy() if torch.is_tensor(sample['image']) else sample['image']
        age = sample['age'].item() if torch.is_tensor(sample['age']) else sample['age']
        patient_id = sample.get('patient_id', f"patient_{i}")
        
        if isinstance(patient_id, torch.Tensor):
            patient_id = patient_id.item() if patient_id.numel() == 1 else str(patient_id)
        
        fig, axes = plt.subplots(4, 3, figsize=(15, 20))
        fig.suptitle(f"Patient {patient_id} - Age {age:.1f} - Mask Verification", fontsize=16)
        
        # Calculate masking statistics
        total_voxels = image.size
        zero_voxels = np.sum(image == 0)
        nonzero_voxels = total_voxels - zero_voxels
        percent_zero = (zero_voxels / total_voxels) * 100
        percent_nonzero = 100 - percent_zero
        
        stats_text = (
            f"Total voxels: {total_voxels:,}\n"
            f"Zero voxels: {zero_voxels:,} ({percent_zero:.2f}%)\n"
            f"Non-zero voxels: {nonzero_voxels:,} ({percent_nonzero:.2f}%)"
        )
        fig.text(0.5, 0.02, stats_text, ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
        
        for j, view_name in enumerate(view_names):
            view_axis = view_axes[view_name]
            slice_index = image.shape[view_axis] // 2
            
            # Get middle slice
            original_slice = np.take(image, indices=slice_index, axis=view_axis)
            
            # Create binary mask (1 where brain is present, 0 elsewhere)
            binary_mask = original_slice != 0
            
            # Row 1: Original image with default colormap (shows how matplotlib renders by default)
            axes[0, j].imshow(original_slice, cmap='gray')
            axes[0, j].set_title(f"{view_name.capitalize()} - Default gray colormap")
            axes[0, j].axis('off')
            
            # Row 2: Original image with custom colormap (pure black for zeros)
            axes[1, j].imshow(original_slice, cmap=custom_cmap, vmin=0)
            axes[1, j].set_title(f"{view_name.capitalize()} - Custom colormap (black zeros)")
            axes[1, j].axis('off')
            
            # Row 3: Binary mask (white = brain, black = background)
            axes[2, j].imshow(binary_mask, cmap='binary')
            axes[2, j].set_title(f"{view_name.capitalize()} - Binary mask")
            axes[2, j].axis('off')
            
            # Row 4: Histogram of pixel values
            if np.any(binary_mask):
                # Log-scale histogram of non-zero values
                nonzero_values = original_slice[binary_mask]
                axes[3, j].hist(nonzero_values.flatten(), bins=50, log=True)
                axes[3, j].axvline(x=0, color='r', linestyle='--', alpha=0.7)
                axes[3, j].set_title(f"{view_name.capitalize()} - Histogram (log scale)")
            else:
                axes[3, j].text(0.5, 0.5, "No non-zero values", ha='center')
                axes[3, j].set_title(f"{view_name.capitalize()} - No brain voxels")
                axes[3, j].axis('off')
        
        plt.tight_layout(rect=[0, 0.04, 1, 0.96])
        plt.savefig(os.path.join(output_dir, f"mask_verification_patient_{patient_id}.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Create a mask boundary visualization
        fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
        fig2.suptitle(f"Patient {patient_id} - Mask Boundary Visualization", fontsize=16)
        
        for j, view_name in enumerate(view_names):
            view_axis = view_axes[view_name]
            slice_index = image.shape[view_axis] // 2
            
            # Get original slice
            original_slice = np.take(image, indices=slice_index, axis=view_axis)
            brain_mask = original_slice != 0
            
            # Use masked array for proper visualization
            masked_data = np.ma.masked_where(~brain_mask, original_slice)
            
            # Plot with proper masking
            axes2[j].imshow(original_slice, cmap='gray', alpha=0.7)
            
            # Highlight the boundary
            from scipy import ndimage
            boundary = ndimage.binary_dilation(brain_mask, iterations=1) ^ brain_mask
            y, x = np.where(boundary)
            if len(y) > 0:
                axes2[j].scatter(x, y, s=1, c='red', alpha=0.8)
                
            axes2[j].set_title(f"{view_name.capitalize()} - Mask Boundary")
            axes2[j].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"mask_boundary_{patient_id}.png"), dpi=300, bbox_inches='tight')
        plt.close(fig2)
    
    logging.info(f"Saved mask verification images for {n_samples} patients to: {output_dir}")
    return output_dir


def train_model(
    csv_file,
    image_dir,
    model,
    model_type="large", 
    batch_size=4,
    learning_rate=0.001,
    num_epochs=100,
    use_wandb=True,
    pretrained_model_path=None,
    use_cuda=False,
    split_strategy="stratified_group",
    test_size=0.2,
    bins=10,
    output_dir="./saved_models",
    wandb_prefix="",
    brainage_alpha=0.0,
    brainage_beta=0.0,
    brainage_gamma=0.0,
    brainage_eps=1e-8,
    brainage_smoothing=0.0,
    brainage_use_huber=False,
    brainage_delta=0.0,
    weight_decay=0.05,
    store_model=True,
    mem_opt=False,
):
    logging.info("Starting training process...")
    os.makedirs(output_dir, exist_ok=True)

    
    try:
        model_name = model.get_name()
    except AttributeError:
        model_name = model_type 
    scaler = GradScaler()
    lr_str = f"{learning_rate:.1e}".replace("+", "").replace("-", "_")
    param_str = (
        f"{wandb_prefix}_{model_name}_{lr_str}_{num_epochs}_" 
        f"{batch_size}_{brainage_alpha}_{brainage_beta}_{brainage_gamma}_{brainage_eps}_{brainage_smoothing}_{brainage_use_huber}_{brainage_delta}"
    )

    if use_wandb:
        wandb_config = {
            "model_type": model_type,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "use_cuda": use_cuda,
            "split_strategy": split_strategy,
            "test_size": test_size,
            "bins": bins,
            "weight_decay": weight_decay,
            "brainage_alpha": brainage_alpha,
            "brainage_beta": brainage_beta,
            "brainage_gamma": brainage_gamma,
            "brainage_eps": brainage_eps,
            "brainage_smoothing": brainage_smoothing,
            "brainage_use_huber": brainage_use_huber,
            "brainage_delta": brainage_delta,
            "params": param_str,
            
        }
        try:
            model_params = model.get_params()
            wandb_config.update(model_params)
        except AttributeError:
            logging.info("Model does not have get_params method, skipping parameter logging to wandb.")

        wandb.init(
            project="asl-brainage",
            name=param_str,
            config=wandb_config,
        )
        run = wandb.run
    else:
        run = None
    dataset = BrainAgeDataset(csv_file, image_dir,mask_path=csv_file)
    if run:
        create_demographics_table(dataset, run)
    dataset = [
        sample for sample in dataset if sample is not None
    ]
    logging.info(
        "Number of samples after filtering for missing data: %d", len(dataset))
    device = torch.device(
        "cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    logging.info("Using device: %s", device)
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = (
            torch.cuda.get_device_properties(0).total_memory / 1024**3
        )
        gpu_info = (
            f"**\033[1mDetected GPU:\033[0m** {gpu_name}, Memory: {gpu_memory:.2f} GB"
        )
        logging.info(gpu_info)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    logging.info(f"Using split strategy: {split_strategy}")

    if split_strategy == "random":
        train_size = int((1 - test_size) * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size]
        )
        
    elif split_strategy == "stratified":
        ages = [sample["age"].item() for sample in dataset]
        age_bins = pd.qcut(ages, q=bins, labels=False)
        train_idx, test_idx = train_test_split(
            range(len(dataset)), test_size=test_size, stratify=age_bins, random_state=42
        )
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        test_dataset = torch.utils.data.Subset(dataset, test_idx)
    elif split_strategy == "group":
        sites = [
            sample["demographics"][1].item() for sample in dataset
        ]  # Site is index 1
        gss = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=42)
        train_idx, test_idx = next(
            gss.split(range(len(dataset)), groups=sites))
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        test_dataset = torch.utils.data.Subset(dataset, test_idx)
    elif split_strategy == "stratified_group_sex":
        # Combine age bins and site for stratification
        ages = [sample["age"].item() for sample in dataset]
        sites = [sample["demographics"][1].item() for sample in dataset]
        sexes = [
            sample["demographics"][0].item() for sample in dataset
        ]  
        age_bins = pd.qcut(ages, q=bins, labels=False)
        stratify_groups = [
            f"{site}_{sex}_{age_bin}"
            for site, sex, age_bin in zip(sites, sexes, age_bins)
        ]  
        train_idx, test_idx = train_test_split(
            range(len(dataset)),
            test_size=test_size,
            stratify=stratify_groups,
            random_state=42,
        )
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        test_dataset = torch.utils.data.Subset(dataset, test_idx)
    elif split_strategy == "stratified_group":
        
        ages = [sample["age"].item() for sample in dataset]
        sites = [sample["demographics"][1].item() for sample in dataset]
        age_bins = pd.qcut(ages, q=bins, labels=False)
        stratify_groups = [
            f"{site}_{age_bin}" for site, age_bin in zip(sites, age_bins)
        ]
        train_idx, test_idx = train_test_split(
            range(len(dataset)),
            test_size=test_size,
            stratify=stratify_groups,
            random_state=42,
        )
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        test_dataset = torch.utils.data.Subset(dataset, test_idx)
    else:
        raise ValueError(f"Invalid split strategy: {split_strategy}")
    #save indices of test dataset
    
    csv_file_path = os.path.dirname(csv_file)
    if os.path.exists(os.path.join(csv_file_path, f"train_indices.npy")) and os.path.exists(os.path.join(csv_file_path, f"test_indices.npy")):
        train_idx = np.load(os.path.join(csv_file_path, f"train_indices.npy"))
        test_idx = np.load(os.path.join(csv_file_path, f"test_indices.npy"))
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        test_dataset = torch.utils.data.Subset(dataset, test_idx)
        logging.info("Train and test indices loaded")
    else:        
        np.save(os.path.join(csv_file_path, f"train_indices.npy"), train_idx)
        np.save(os.path.join(csv_file_path, f"test_indices.npy"), test_idx)
        logging.info("Train and test indices saved")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers = 2, pin_memory=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers = 2, pin_memory=True)
    
    logging.info("Data loaders created")
    
    model.to(device)
    cmodel = torch.compile(model)
    logging.info(f"Model created: {cmodel.get_name() if hasattr(cmodel, 'get_name') else model_type}")

    #criterion = nn.L1Loss()
    criterion = BrainAgeLoss(
        alpha=brainage_alpha,
        beta=brainage_beta,
        gamma=brainage_gamma,
        eps=brainage_eps,
        smoothing=brainage_smoothing,
        use_huber=brainage_use_huber,
        delta=brainage_delta
    )
    logging.info(f"Loss function: {criterion.get_name()} with parameters: alpha={brainage_alpha}, beta={brainage_beta}, gamma={brainage_gamma}, eps={brainage_eps}, smoothing={brainage_smoothing}, use_huber={brainage_use_huber}, delta={brainage_delta}")

    #optimizer = optim.Adam(cmodel.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR.ReduceLROnPlateau(
    #     optimizer,
    #     mode='min',
    #     factor=0.5,
    #     patience=20,
    #     verbose=True,
    # )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    warmup_epochs = 20
    eta_min = 1e-6
    warmup_scheduler = optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=1e-6/learning_rate,  # start from a very small LR
    end_factor=1.0,              # reach initial_lr at the end of warmup
    total_iters=warmup_epochs    # number of warmup epochs
    )
    main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_epochs - warmup_epochs,  # adjust T_max for warmup, this seems to be working fine for our use-case
    eta_min=eta_min
    )
    scheduler = optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, main_scheduler],
    milestones=[warmup_epochs]  # fallback to main_scheduler after warmup
    )
    
    logging.info(f"Loss function and optimizer set up.")
    current_time = datetime.datetime.now().strftime("%y-%m-%d_%H-%M")
    best_model_path = os.path.join(output_dir, f"{param_str}_{current_time}.pth")
    if not pretrained_model_path:
        best_test_mae = float("inf")
        epochs_no_improve = 0
        for epoch in range(num_epochs):
            logging.info(f"Starting epoch: {epoch}")
            cmodel.train()
            train_loss = 0
            for i, batch in enumerate(train_loader):
                logging.debug(f"Processing batch: {i}")
                images = (
                    batch["image"].unsqueeze(1).to(device)
                )
                ages = batch["age"].unsqueeze(1).to(device)
                demographics = batch["demographics"].to(device)
                optimizer.zero_grad()
                if mem_opt:
                    with autocast():
                        outputs = cmodel(images, demographics)
                        loss, metrics = criterion(outputs, ages, demographics)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = cmodel(images, demographics)
                    #loss = criterion(outputs, ages)
                    loss, metrics = criterion(outputs, ages, demographics)
                    loss.backward()
                    optimizer.step()
                train_loss += loss.item()
                logging.debug(f"Batch {i} processed. Loss: {loss.item()}")
            train_loss = train_loss / len(train_loader)
            logging.info(f"Epoch: {epoch}, Training Loss: {train_loss}")
            cmodel.eval()
            with torch.no_grad():
                test_mae, test_rmse, test_r2, test_pearson, test_mape = evaluate_model(
                    cmodel, test_loader, device
                )
            logging.info(
                f"Epoch {epoch} Test MAE: {test_mae}, Test MAPE: {test_mape}, RMSE: {test_rmse}, R2: {test_r2}, Pearson Correlation: {test_pearson}"
            )
            if run:
                wandb.log(
                    {
                        "train_loss": train_loss,
                        "test_mae": test_mae,
                        "test_mape": test_mape,
                        "test_rmse": test_rmse,
                        "test_r2": test_r2,
                        "test_pearson": test_pearson,
                        "metrics_mae": metrics["mae"],
                        "metrics_corr_loss": metrics["corr_loss"],
                        "metrics_std_ratio_loss": metrics["std_ratio_loss"],
                        "metrics_correlation": metrics["correlation"],
                        "metrics_pred_std": metrics["pred_std"],
                        "metrics_target_std": metrics["target_std"],
                        
                    }
                )
            #scheduler.step(test_mae)
            scheduler.step()
            #log if learning rate changes
            logging.info(f"-> Learning rate: {optimizer.param_groups[0]['lr']}") if optimizer.param_groups[0]['lr'] != learning_rate else None
            if round(test_mae, 2) < round(best_test_mae, 2):
                epochs_no_improve = 0
                best_test_mae = test_mae
                if store_model:
                    _sd = cmodel.state_dict()
                    _sd = {key.replace("_orig_mod.", ""): value for key, value in _sd.items()}
                    model.load_state_dict(_sd)
                    torch.save(model, best_model_path)

                    logging.info(
                        f"Model saved at epoch {epoch} as test MAE {test_mae} is better than previous best {best_test_mae}"
                    )
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= 40:
                    logging.info(
                        f"Early stopping at epoch {epoch} as test MAE did not improve over 20 epochs"
                    )
                    break

        logging.info("Training completed.")
    cmodel.eval()
    with torch.no_grad():
        test_mae, test_rmse, test_r2, test_pearson, test_mape = evaluate_model(
            cmodel, test_loader, device
        )
    logging.info(
        f"Final Test MAE: {test_mae}, Test MAPE: {test_mape}, RMSE: {test_rmse}, R2: {test_r2}, Pearson Correlation: {test_pearson}"
    )
    if run:
        create_prediction_chart(cmodel, test_loader, device, run)
        wandb.log(
            {
                "final_test_mae": test_mae,
                "final_test_mape": test_mape,
                "final_test_rmse": test_rmse,
                "final_test_r2": test_r2,
                "final_test_pearson": test_pearson,
            }
        )
    if run:
        wandb.finish()
    return model


def evaluate_model(model, test_loader, device):
    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].unsqueeze(1).to(device)
            ages = batch["age"].unsqueeze(1).to(device)
            demographics = batch["demographics"].to(device)
            outputs = model(images, demographics)  # forward pass
            all_predictions.extend(
                outputs.cpu().numpy().flatten()
            )
            all_targets.extend(
                ages.cpu().numpy().flatten()
            )
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    mae = mean_absolute_error(all_targets, all_predictions)
    rmse = np.sqrt(mean_squared_error(
        all_targets, all_predictions))

    r2 = r2_score(all_targets, all_predictions)
    pearson, _ = pearsonr(all_targets, all_predictions)
    mape = np.mean(np.abs((all_targets - all_predictions) / (all_targets + 1e-8))) * 100
    return mae, rmse, r2, pearson, mape


def main():
    parser = argparse.ArgumentParser(
        description="Brain Age Prediction Training Script")

    parser.add_argument(
        "--csv_file",
        type=str,
        default="/home/radv/samiri/my-scratch/trainingdata/topmri.csv",
        help="Path to the CSV file",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="/home/radv/samiri/my-scratch/trainingdata/topmri/",
        help="Path to the image directory",
    )
    parser.add_argument("--wandb_prefix", type=str,
                        default="", help="wandb job prefix")
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.000015,
        help="Learning rate for training",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=150, help="Number of epochs for training"
    )
    parser.add_argument("--use_wandb", action="store_true",
                        help="Enable wandb logging")
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=None,
        help="Path to a pretrained model file",
    )
    parser.add_argument(
        "--use_cuda",
        action="store_true",
        default=True,
        help="Enable CUDA (GPU) if available",
    )
    parser.add_argument(
      "--store_model",
      action="store_true",
      help="Store the best model",
    )

    parser.add_argument(
        "--split_strategy",
        type=str,
        default="stratified_group_sex",
        choices=[
            "random",
            "stratified",
            "group",
            "stratified_group",
            "stratified_group_sex",
        ],
        help="Data split strategy (random/stratified/group/stratified_group)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay for Adam optimizer",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion of data to use for testing",
    )
    parser.add_argument(
        "--bins", type=int, default=10, help="Number of bins for stratification"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./saved_models",
        help="Directory to save trained models",
    )

    parser.add_argument(
        "--model_type",
        type=str,
        default="large",
        choices=[
            "large",
            "resnet",
            "densenet",
            "resnext3d",
            "efficientnet3d",
            "improved_cnn",
            "vit3d",
        ],
        help="Type of model to use",
    )


    # Large3DCNN Parameters
    large_cnn_group = parser.add_argument_group("Large3DCNN arguments")
    large_cnn_group.add_argument("--large_cnn_layers", type=int, default=3, help="Number of conv layers for Large3DCNN")
    large_cnn_group.add_argument("--large_cnn_filters", type=int, default=16, help="Initial filters for Large3DCNN")
    large_cnn_group.add_argument("--large_cnn_filters_multiplier", type=float, default=2.0, help="Filters multiplier for Large3DCNN") # Added filter multiplier
    large_cnn_group.add_argument("--large_cnn_use_bn", action="store_true", help="Use BN in Large3DCNN")
    large_cnn_group.add_argument("--large_cnn_use_se", action="store_true", help="Use SE blocks in Large3DCNN")
    large_cnn_group.add_argument("--large_cnn_use_dropout", action="store_true", help="Use dropout in Large3DCNN")
    large_cnn_group.add_argument("--large_cnn_dropout_rate", type=float, default=0.2, help="Dropout rate for Large3DCNN")

    # ResNet3D Parameters
    resnet_group = parser.add_argument_group("ResNet3D arguments")
    resnet_group.add_argument("--resnet_layers", type=int, nargs='+', default=[2, 2, 2], help="Number of blocks per ResNet layer (list of 3 ints)")
    resnet_group.add_argument("--resnet_initial_filters", type=int, default=32, help="Initial filters for ResNet3D")
    resnet_group.add_argument("--resnet_filters_multiplier", type=float, default=2.0, help="Filters multiplier for ResNet3D") # Added filter multiplier
    resnet_group.add_argument("--resnet_use_se", action="store_true", help="Use SE blocks in ResNet3D")
    resnet_group.add_argument("--resnet_dropout", action="store_true", help="Use dropout in ResNet3D")

    # DenseNet3D Parameters
    densenet_group = parser.add_argument_group("DenseNet3D arguments")
    densenet_group.add_argument("--densenet_growth_rate", type=int, default=16, help="Growth rate for DenseNet3D")
    densenet_group.add_argument("--densenet_layers", type=int, nargs='+', default=[4, 4], help="Number of layers per DenseBlock (list of 2 ints)")
    densenet_group.add_argument("--densenet_initial_filters", type=int, default=32, help="Initial filters for DenseNet3D")
    densenet_group.add_argument("--densenet_transition_filters_multiplier", type=float, default=2.0, help="Filters multiplier for DenseNet3D transition layers") # Added filter multiplier
    densenet_group.add_argument("--densenet_use_se", action="store_true", help="Use SE blocks in DenseNet3D")
    densenet_group.add_argument("--densenet_dropout", action="store_true", help="Use dropout in DenseNet3D")

    # ResNeXt3D Parameters
    resnext_group = parser.add_argument_group("ResNeXt3D arguments")
    resnext_group.add_argument("--resnext_layers", type=int, nargs='+', default=[2, 2, 2], help="Number of blocks per ResNeXt layer (list of 3 ints)")
    resnext_group.add_argument("--resnext_cardinality", type=int, default=32, help="Cardinality for ResNeXt3D")
    resnext_group.add_argument("--resnext_bottleneck_width", type=int, default=4, help="Bottleneck width for ResNeXt3D")
    resnext_group.add_argument("--resnext_initial_filters", type=int, default=64, help="Initial filters for ResNeXt3D")
    resnext_group.add_argument("--resnext_filters_multiplier", type=float, default=2.0, help="Filters multiplier for ResNeXt3D") # Added filter multiplier
    resnext_group.add_argument("--resnext_use_se", action="store_true", help="Use SE blocks in ResNeXt3D")
    resnext_group.add_argument("--resnext_dropout", action="store_true", help="Use dropout in ResNeXt3D")

    # EfficientNet3D Parameters
    efficientnet_group = parser.add_argument_group("EfficientNet3D arguments")
    efficientnet_group.add_argument("--efficientnet_width_coefficient", type=float, default=1.0, help="Width coefficient for EfficientNet3D")
    efficientnet_group.add_argument("--efficientnet_depth_coefficient", type=float, default=1.0, help="Depth coefficient for EfficientNet3D")
    efficientnet_group.add_argument("--efficientnet_initial_filters", type=int, default=32, help="Initial filters for EfficientNet3D")
    efficientnet_group.add_argument("--efficientnet_filters_multiplier", type=float, default=1.2, help="Filters multiplier for EfficientNet3D") # Added filter multiplier
    efficientnet_group.add_argument("--efficientnet_dropout", type=float, default=0.2, help="Dropout rate for EfficientNet3D (set 0 to disable)")

    # Improved3DCNN Parameters
    improved_cnn_group = parser.add_argument_group("Improved3DCNN arguments")
    improved_cnn_group.add_argument("--improved_cnn_initial_filters", type=int, default=32, help="Initial filters for Improved3DCNN")
    improved_cnn_group.add_argument("--improved_cnn_filters_multiplier", type=float, default=2.0, help="Filters multiplier for Improved3DCNN") # Added filter multiplier
    improved_cnn_group.add_argument("--improved_cnn_num_conv_layers", type=int, default=3, help="Number of conv layers for Improved3DCNN") # Added num_conv_layers
    improved_cnn_group.add_argument("--improved_cnn_use_se", action="store_true", help="Use SE blocks in Improved3DCNN")
    improved_cnn_group.add_argument("--improved_cnn_dropout_rate", type=float, default=0.3, help="Dropout rate for Improved3DCNN")
    
    # VisionTransformer3D Parameters
    vit3d_group = parser.add_argument_group("VisionTransformer3D arguments")
    vit3d_group.add_argument("--vit3d_patch_size", type=int, nargs='+', default=[16, 16, 16], help="Patch size for ViT3D (list of 3 ints)")
    vit3d_group.add_argument("--vit3d_embed_dim", type=int, default=768, help="Embedding dimension for ViT3D")
    vit3d_group.add_argument("--vit3d_depth", type=int, default=12, help="Depth (number of layers) for ViT3D")
    vit3d_group.add_argument("--vit3d_num_heads", type=int, default=12, help="Number of attention heads for ViT3D")
    vit3d_group.add_argument("--vit3d_mlp_ratio", type=float, default=4.0, help="MLP ratio for ViT3D")
    vit3d_group.add_argument("--vit3d_qkv_bias", action="store_true", help="Use QKV bias in ViT3D attention")
    vit3d_group.add_argument("--vit3d_drop_rate", type=float, default=0.0, help="Dropout rate for ViT3D")
    vit3d_group.add_argument("--vit3d_attn_drop_rate", type=float, default=0.0, help="Attention dropout rate for ViT3D")
    vit3d_group.add_argument("--vit3d_drop_path_rate", type=float, default=0.0, help="Drop path rate for ViT3D")
    vit3d_group.add_argument("--vit3d_global_pool", action="store_true", help="Use global average pooling in ViT3D")    
    vit3d_group.add_argument("--vit3d_use_cls_token", action="store_true", help="Use CLS token in ViT3D")
    vit3d_group.add_argument("--vit3d_use_hybrid_embed", action="store_true", help="Use hybrid embedding in ViT3D")
    vit3d_group.add_argument("--vit3d_hybrid_kernel_size", type=int, default=3, help="Hybrid embedding kernel size for ViT3D")
    
    # BrainAgeLoss Parameters
    brainage_loss_group = parser.add_argument_group("BrainAgeLoss arguments")
    brainage_loss_group.add_argument("--brainage_alpha", type=float, default=1.0, help="Weight for correlation loss in BrainAgeLoss")
    brainage_loss_group.add_argument("--brainage_beta", type=float, default=0.3, help="Weight for bias regularization in BrainAgeLoss")
    brainage_loss_group.add_argument("--brainage_gamma", type=float, default=0.2, help="Weight for age-specific weighting in BrainAgeLoss")
    brainage_loss_group.add_argument("--brainage_eps", type=float, default=1e-8, help="Epsilon for numerical stability in BrainAgeLoss")
    brainage_loss_group.add_argument("--brainage_smoothing", type=float, default=0.1, help="Label smoothing factor in BrainAgeLoss")
    brainage_loss_group.add_argument("--brainage_use_huber", action="store_true", default=True, help="Use Huber loss in BrainAgeLoss")
    brainage_loss_group.add_argument("--brainage_delta", type=float, default=1.0, help="Delta parameter for Huber loss in BrainAgeLoss")
    

    args = parser.parse_args()

    num_demographics = 6  # Number of demographic features

    if args.model_type == "improved_cnn":
        model = Improved3DCNN(
            num_demographics=num_demographics,
            initial_filters=args.improved_cnn_initial_filters,
            filters_multiplier=args.improved_cnn_filters_multiplier,
            num_conv_layers=args.improved_cnn_num_conv_layers,
            use_se=args.improved_cnn_use_se,
            dropout_rate=args.improved_cnn_dropout_rate,
        )

    elif args.model_type == "large":
        model = Large3DCNN(
            num_demographics=num_demographics,
            num_conv_layers=args.large_cnn_layers,
            initial_filters=args.large_cnn_filters,
            filters_multiplier=args.large_cnn_filters_multiplier,
            use_bn=args.large_cnn_use_bn,
            use_se=args.large_cnn_use_se,
            use_dropout=args.large_cnn_use_dropout,
            dropout_rate=args.large_cnn_dropout_rate
        )

    elif args.model_type == "resnet":
        model = ResNet3D(
            num_demographics=num_demographics,
            num_blocks_per_layer=args.resnet_layers,
            initial_filters=args.resnet_initial_filters,
            filters_multiplier=args.resnet_filters_multiplier,
            use_se=args.resnet_use_se,
            use_dropout=args.resnet_dropout
        )

    elif args.model_type == "densenet":
        model = DenseNet3D(
            num_demographics=num_demographics,
            growth_rate=args.densenet_growth_rate,
            num_dense_layers=args.densenet_layers,
            initial_filters=args.densenet_initial_filters,
            transition_filters_multiplier=args.densenet_transition_filters_multiplier,
            use_se_blocks=args.densenet_use_se,
            use_dropout=args.densenet_dropout,
        )
    elif args.model_type == "resnext3d":
        model = ResNeXt3D(
            num_demographics=num_demographics,
            num_blocks_per_layer=args.resnext_layers,
            cardinality=args.resnext_cardinality,
            bottleneck_width=args.resnext_bottleneck_width,
            initial_filters=args.resnext_initial_filters,
            filters_multiplier=args.resnext_filters_multiplier,
            use_se=args.resnext_use_se,
            use_dropout=args.resnext_dropout,
        )
    elif args.model_type == "efficientnet3d":
        model = EfficientNet3D(
            num_demographics=num_demographics,
            width_coefficient=args.efficientnet_width_coefficient,
            depth_coefficient=args.efficientnet_depth_coefficient,
            initial_filters=args.efficientnet_initial_filters,
            filters_multiplier=args.efficientnet_filters_multiplier,
            use_dropout=args.efficientnet_dropout,
        )

    elif args.model_type == "vit3d":
        model = VisionTransformer3D(
            num_demographics=num_demographics,
            img_size=[121, 145, 121], 
            patch_size=args.vit3d_patch_size,
            embed_dim=args.vit3d_embed_dim,
            depth=args.vit3d_depth,
            num_heads=args.vit3d_num_heads,
            mlp_ratio=args.vit3d_mlp_ratio,
            qkv_bias=args.vit3d_qkv_bias,
            drop_rate=args.vit3d_drop_rate,
            attn_drop_rate=args.vit3d_attn_drop_rate,
            drop_path_rate=args.vit3d_drop_path_rate,
            global_pool=args.vit3d_global_pool,
            use_cls_token = args.vit3d_use_cls_token,
            use_hybrid_embed = args.vit3d_use_hybrid_embed,
            hybrid_kernel_size = args.vit3d_hybrid_kernel_size,
            use_demographics=True
        )
            
    else:
        raise ValueError(f"Invalid model_type: {args.model_type}")

    train_model(
        csv_file=args.csv_file,
        image_dir=args.image_dir,
        model=model,
        model_type=args.model_type,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        use_wandb=args.use_wandb,
        pretrained_model_path=args.pretrained_model_path,
        use_cuda=args.use_cuda,
        split_strategy=args.split_strategy,
        test_size=args.test_size,
        bins=args.bins,
        output_dir=args.output_dir,
        wandb_prefix=args.wandb_prefix,
        weight_decay=args.weight_decay,
        store_model=args.store_model,
        brainage_alpha=args.brainage_alpha,
        brainage_beta=args.brainage_beta,
        brainage_gamma=args.brainage_gamma,
        brainage_eps=args.brainage_eps,
        brainage_smoothing=args.brainage_smoothing,
        brainage_use_huber=args.brainage_use_huber,
        brainage_delta=args.brainage_delta,
        
    )

if __name__ == "__main__":
    main()