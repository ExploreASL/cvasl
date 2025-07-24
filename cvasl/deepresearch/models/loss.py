import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BrainAgeLoss(nn.Module):
    """
    Combined loss function for brain age prediction tasks.
    
    This loss combines:
    1. Mean Absolute Error (MAE) for accurate age prediction
    2. Correlation penalty to ensure predictions correlate with true ages
    3. Age-specific weighting to address age bias
    4. Regularization to prevent overfitting to the mean
    
    Parameters:
    -----------
    alpha : float, default=1.0
        Weight for the correlation component.
    beta : float, default=0.3
        Weight for the bias regularization component.
    gamma : float, default=0.2
        Weight for the age-specific weighting component.
    eps : float, default=1e-8
        Small constant for numerical stability.
    smoothing : float, default=0.1
        Amount of label smoothing to apply.
    use_huber : bool, default=False
        Whether to use Huber loss instead of MAE for robustness.
    delta : float, default=1.0
        Parameter for Huber loss if used.
    """
    def __init__(self, alpha=1.0, beta=0.3, gamma=0.2, eps=1e-8, 
                 smoothing=0.1, use_huber=True, delta=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps
        self.smoothing = smoothing
        self.use_huber = use_huber
        self.delta = delta
        
    def forward(self, pred, target):
        """
        Calculate the combined loss.
        
        Parameters:
        -----------
        pred : torch.Tensor
            Model predictions, shape (batch_size,)
        target : torch.Tensor
            Ground truth ages, shape (batch_size,)
        demographic_factors : torch.Tensor, optional
            Demographic factors like sex, ethnicity that might influence the loss weighting,
            shape (batch_size, num_factors)
            
        Returns:
        --------
        loss : torch.Tensor
            The combined loss value
        metrics : dict
            Dictionary containing individual loss components for monitoring
        """
        # Apply label smoothing if enabled
        if self.smoothing > 0:
            # Calculate mean age in the batch
            mean_age = torch.mean(target)
            # Smooth targets toward the mean age
            target_smooth = target * (1 - self.smoothing) + mean_age * self.smoothing
        else:
            target_smooth = target
            
        # 1. Basic error term (MAE or Huber)
        if self.use_huber:
            error_term = F.huber_loss(pred, target_smooth, delta=self.delta, reduction='none')
        else:
            error_term = torch.abs(pred - target_smooth)
            
        # 2. Age-specific weighting (optional)
        if self.gamma > 0:
            # compute weights that give more importance to under-represented age groups
            age_weights = 1.0 + self.gamma * (target / torch.max(target))
            weighted_error = error_term * age_weights
            basic_loss = torch.mean(weighted_error)
        else:
            basic_loss = torch.mean(error_term)
            
        # 3. Correlation term
        # Center the data
        pred_centered = pred - torch.mean(pred)
        target_centered = target - torch.mean(target)
        
        # Calculate Pearson correlation
        pred_var = torch.sum(pred_centered ** 2)
        target_var = torch.sum(target_centered ** 2)
        correlation = torch.sum(pred_centered * target_centered) / (torch.sqrt(pred_var * target_var) + self.eps)
        
        # convert to a loss (1 - correlation)
        correlation_loss = 1.0 - correlation
        # 4. bias regularization to prevent prediction toward mean
        pred_std = torch.std(pred)
        target_std = torch.std(target)
        std_ratio = torch.abs(pred_std / (target_std + self.eps) - 1.0)
        
        # 5. Combine all terms
        combined_loss = basic_loss + self.alpha * correlation_loss + self.beta * std_ratio
        
        # Return individual components for monitoring
        metrics = {
            'mae': torch.mean(torch.abs(pred - target)),  # Pure MAE for monitoring
            'corr_loss': correlation_loss.item(),
            'std_ratio_loss': std_ratio.item(),
            'correlation': correlation.item(),
            'pred_std': pred_std.item(),
            'target_std': target_std.item()
        }
        
        return combined_loss, metrics
    
    def get_name(self):
        """Returns a descriptive name of the loss function and its parameters."""
        name = f"BrainAgeLoss_a{self.alpha}_b{self.beta}_g{self.gamma}"
        if self.use_huber:
            name += f"_huber{self.delta}"
        if self.smoothing > 0:
            name += f"_smooth{self.smoothing}"
        return name