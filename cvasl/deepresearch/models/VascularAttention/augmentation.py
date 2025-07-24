import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class PhysiologicalAugmentation:
    """
    Applies physiologically plausible augmentations to ASL images.
    """
    def __init__(self, intensity_shift_range=0.1, smoothing_kernel_size=3, smoothing_sigma_range=1.0):
        self.intensity_shift_range = intensity_shift_range
        self.smoothing_kernel_size = smoothing_kernel_size
        self.smoothing_sigma_range = smoothing_sigma_range

    def __call__(self, image):
        """
        Applies a combination of intensity shift and localized smoothing.

        Args:
            image (np.ndarray): Input ASL image (3D numpy array).

        Returns:
            np.ndarray: Augmented ASL image.
        """
        augmented_image = self.intensity_shift(image)
        augmented_image = self.localized_smoothing(augmented_image)
        return augmented_image

    def intensity_shift(self, image):
        """
        Applies a random intensity shift to the image.

        Args:
            image (np.ndarray): Input ASL image.

        Returns:
            np.ndarray: Intensity-shifted image.
        """
        shift_factor = np.random.uniform(-self.intensity_shift_range, self.intensity_shift_range)
        return image + shift_factor * np.max(np.abs(image)) # Scale shift by image intensity range

    def localized_smoothing(self, image):
        """
        Applies localized Gaussian smoothing to the image.

        Args:
            image (np.ndarray): Input ASL image.

        Returns:
            np.ndarray: Smoothed image.
        """
        sigma = np.random.uniform(0, self.smoothing_sigma_range)
        if sigma > 0:
            # Convert numpy array to torch tensor, add channel and batch dimension
            image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
            smoothed_tensor = self._gaussian_blur3d(image_tensor, kernel_size=self.smoothing_kernel_size, sigma=sigma)
            return smoothed_tensor.squeeze(0).squeeze(0).numpy() # Convert back to numpy array and remove dimensions
        return image # No smoothing if sigma is 0


    def _gaussian_blur3d(self, img, kernel_size, sigma):
        """
        Apply Gaussian blur in 3D using PyTorch functional.

        Args:
            img (torch.Tensor): Input image tensor (B, C, D, H, W).
            kernel_size (int): Size of the Gaussian kernel. Should be odd.
            sigma (float): Standard deviation of the Gaussian kernel.

        Returns:
            torch.Tensor: Blurred image tensor.
        """
        padding = kernel_size // 2
        channels = img.shape[1]

        # Create 3D Gaussian kernel
        kernel_z = self._gaussian_kernel_1d(kernel_size, sigma).reshape(-1, 1, 1)
        kernel_y = self._gaussian_kernel_1d(kernel_size, sigma).reshape(1, -1, 1)
        kernel_x = self._gaussian_kernel_1d(kernel_size, sigma).reshape(1, 1, -1)

        kernel_3d = kernel_z * kernel_y * kernel_x
        kernel_3d = kernel_3d / kernel_3d.sum() # Normalize kernel
        kernel_3d = kernel_3d.float().unsqueeze(0).unsqueeze(0) # Add channel and output channel dims
        kernel_3d = kernel_3d.repeat(channels, 1, 1, 1, 1) # Repeat for each input channel


        # Move kernel to the same device as the image
        kernel_3d = kernel_3d.to(img.device)

        # Apply convolution in 3D, using groups to apply kernel channel-wise
        blurred_img = F.conv3d(img, weight=kernel_3d, padding=padding, groups=channels)
        return blurred_img


    def _gaussian_kernel_1d(self, kernel_size, sigma):
        """
        Creates a 1D Gaussian kernel.

        Args:
            kernel_size (int): Size of the kernel (should be odd).
            sigma (float): Standard deviation of the Gaussian.

        Returns:
            torch.Tensor: 1D Gaussian kernel.
        """
        x = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32)
        gauss = torch.exp(-(x**2) / (2 * sigma**2))
        return gauss / gauss.sum() # Normalize kernel

