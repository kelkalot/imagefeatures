"""
Gabor texture feature extractor.

Computes texture features using Gabor wavelet filters at multiple scales and orientations.
"""

import numpy as np
from imagefeatures.base import GlobalFeature, register_feature
from imagefeatures.utils.metrics import dist_l2


@register_feature("gabor")
class Gabor(GlobalFeature):
    """
    Gabor texture feature descriptor.
    
    Applies Gabor filters at multiple scales and orientations to extract
    texture features. Returns a 60-dimensional feature vector (4 scales x 6 orientations x mean,std).
    
    Based on the LIRE implementation by Marko Keuschnig & Christian Penz.
    
    Example:
        >>> from imagefeatures.features import Gabor
        >>> from imagefeatures.utils import load_image
        >>> 
        >>> gabor = Gabor()
        >>> gabor.extract(load_image('image.jpg'))
        >>> features = gabor.get_feature_vector()  # 60-dim vector
    """
    
    # Filter parameters
    U_H = 0.4  # Upper frequency
    U_L = 0.05  # Lower frequency
    S = 4  # Number of scales
    T = 6  # Number of orientations (renamed from T to avoid confusion)
    
    def __init__(self):
        self._feature_vector = None
        self._precomputed = False
        self._gabor_wavelets = None
    
    def extract(self, image: np.ndarray) -> None:
        """Extract Gabor features from image."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = (0.299 * image[:, :, 0] + 
                    0.587 * image[:, :, 1] + 
                    0.114 * image[:, :, 2]).astype(np.float64)
        else:
            gray = image.astype(np.float64)
        
        h, w = gray.shape
        
        # Downsample large images for speed (Gabor works well on smaller images)
        max_size = 128
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            # Simple downsampling by slicing
            step_h = max(1, h // new_h)
            step_w = max(1, w // new_w)
            gray = gray[::step_h, ::step_w]
        
        # Compute Gabor responses
        features = []
        
        for m in range(self.S):  # scales
            for n in range(self.T):  # orientations
                # Create Gabor kernel for this scale/orientation
                kernel = self._create_gabor_kernel(m, n, size=11)  # Smaller kernel
                
                # Apply filter using convolution
                response = self._convolve(gray, kernel)
                
                # Compute magnitude
                magnitude = np.abs(response)
                
                # Extract mean and std as features
                features.append(np.mean(magnitude))
                features.append(np.std(magnitude))
        
        self._feature_vector = np.array(features, dtype=np.float64)
        
        # Normalize
        self._normalize()
    
    def _create_gabor_kernel(self, m: int, n: int, size: int = 15) -> np.ndarray:
        """
        Create a Gabor wavelet kernel.
        
        Args:
            m: Scale index
            n: Orientation index
            size: Kernel size
        """
        # Compute frequency and orientation
        a = (self.U_H / self.U_L) ** (1.0 / (self.S - 1)) if self.S > 1 else 1
        u_m = self.U_H / (a ** m)
        theta_n = n * np.pi / self.T
        
        # Compute sigma
        sigma_x = 1.0 / (2 * np.pi * u_m)
        sigma_y = sigma_x
        
        # Create kernel
        half = size // 2
        kernel = np.zeros((size, size), dtype=np.complex128)
        
        for y in range(-half, half + 1):
            for x in range(-half, half + 1):
                # Rotate coordinates
                x_theta = x * np.cos(theta_n) + y * np.sin(theta_n)
                y_theta = -x * np.sin(theta_n) + y * np.cos(theta_n)
                
                # Gaussian envelope
                gaussian = np.exp(-0.5 * (x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2))
                
                # Complex sinusoid
                sinusoid = np.exp(2j * np.pi * u_m * x_theta)
                
                kernel[y + half, x + half] = gaussian * sinusoid
        
        return kernel
    
    def _convolve(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Simple 2D convolution.
        
        Uses separable approach for efficiency where possible.
        """
        kh, kw = kernel.shape
        ih, iw = image.shape
        half_h, half_w = kh // 2, kw // 2
        
        # Pad image
        padded = np.pad(image, ((half_h, half_h), (half_w, half_w)), mode='reflect')
        
        # Subsample for speed (use every 4th pixel)
        step = 4
        result = np.zeros((ih // step + 1, iw // step + 1), dtype=np.complex128)
        
        for i, y in enumerate(range(0, ih, step)):
            for j, x in enumerate(range(0, iw, step)):
                patch = padded[y:y+kh, x:x+kw]
                result[i, j] = np.sum(patch * kernel)
        
        return result
    
    def _normalize(self) -> None:
        """Normalize feature vector."""
        if self._feature_vector is None:
            return
        
        # Separate means and stds
        means = self._feature_vector[0::2]
        stds = self._feature_vector[1::2]
        
        # Normalize each part separately
        mean_norm = np.std(means)
        std_norm = np.std(stds)
        
        if mean_norm > 0:
            self._feature_vector[0::2] = (means - np.mean(means)) / mean_norm
        if std_norm > 0:
            self._feature_vector[1::2] = (stds - np.mean(stds)) / std_norm
    
    def get_feature_vector(self) -> np.ndarray:
        """Return the Gabor feature vector."""
        if self._feature_vector is None:
            return np.zeros(self.S * self.T * 2, dtype=np.float64)
        return self._feature_vector
    
    @property
    def name(self) -> str:
        return "gabor"
    
    def get_distance(self, other: 'Gabor') -> float:
        """Use L2 distance for Gabor comparison."""
        return dist_l2(self._feature_vector, other._feature_vector)
