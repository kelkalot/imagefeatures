"""
Haralick Texture Features from Gray-Level Co-occurrence Matrix (GLCM).

Classic texture descriptor based on spatial relationships of pixel intensities.
"""

import numpy as np
from imagefeatures.base import GlobalFeature, register_feature
from imagefeatures.utils.metrics import dist_l2


@register_feature("haralick")
class Haralick(GlobalFeature):
    """
    Haralick Texture Features.
    
    Computes texture features from the Gray-Level Co-occurrence Matrix (GLCM).
    Features computed:
    - Contrast
    - Correlation
    - Energy (Angular Second Moment)
    - Homogeneity (Inverse Difference Moment)
    - Entropy
    - Dissimilarity
    
    Computed for 4 directions (0°, 45°, 90°, 135°) and averaged.
    Returns 6 dimensions (one per feature).
    
    Example:
        >>> from imagefeatures.features import Haralick
        >>> from imagefeatures.utils import load_image
        >>> 
        >>> har = Haralick()
        >>> har.extract(load_image('image.jpg'))
        >>> features = har.get_feature_vector()  # 6-dim vector
    """
    
    def __init__(self, levels: int = 16, distance: int = 1):
        """
        Args:
            levels: Number of gray levels for quantization (default: 16)
            distance: Pixel distance for co-occurrence (default: 1)
        """
        self.levels = levels
        self.distance = distance
        self._features = None
    
    def extract(self, image: np.ndarray) -> None:
        """Extract Haralick features from image."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = (0.299 * image[:, :, 0] + 
                    0.587 * image[:, :, 1] + 
                    0.114 * image[:, :, 2]).astype(np.float64)
        else:
            gray = image.astype(np.float64)
        
        # Quantize to fewer levels
        gray = (gray / 256 * self.levels).astype(np.int32)
        gray = np.clip(gray, 0, self.levels - 1)
        
        h, w = gray.shape
        
        # Downsample large images
        if max(h, w) > 256:
            step = max(h, w) // 256
            gray = gray[::step, ::step]
            h, w = gray.shape
        
        # Compute GLCM for 4 directions
        directions = [(0, 1), (-1, 1), (-1, 0), (-1, -1)]  # 0°, 45°, 90°, 135°
        
        all_features = []
        for dy, dx in directions:
            glcm = self._compute_glcm(gray, dy * self.distance, dx * self.distance)
            features = self._compute_features(glcm)
            all_features.append(features)
        
        # Average over directions
        self._features = np.mean(all_features, axis=0)
    
    def _compute_glcm(self, gray: np.ndarray, dy: int, dx: int) -> np.ndarray:
        """Compute Gray-Level Co-occurrence Matrix."""
        h, w = gray.shape
        glcm = np.zeros((self.levels, self.levels), dtype=np.float64)
        
        y_start = max(0, dy)
        y_end = h + min(0, dy)
        x_start = max(0, dx)
        x_end = w + min(0, dx)
        
        for y in range(y_start, y_end):
            for x in range(x_start, x_end):
                i = gray[y, x]
                j = gray[y - dy, x - dx]
                glcm[i, j] += 1
        
        # Normalize
        total = np.sum(glcm)
        if total > 0:
            glcm /= total
        
        return glcm
    
    def _compute_features(self, glcm: np.ndarray) -> np.ndarray:
        """Compute Haralick features from GLCM."""
        features = np.zeros(6, dtype=np.float64)
        
        # Create index arrays
        i, j = np.ogrid[0:self.levels, 0:self.levels]
        
        # Mean and variance for correlation
        mu_i = np.sum(i * glcm)
        mu_j = np.sum(j * glcm)
        var_i = np.sum((i - mu_i)**2 * glcm)
        var_j = np.sum((j - mu_j)**2 * glcm)
        
        # Contrast
        features[0] = np.sum((i - j)**2 * glcm)
        
        # Correlation
        if var_i > 0 and var_j > 0:
            features[1] = np.sum((i - mu_i) * (j - mu_j) * glcm) / np.sqrt(var_i * var_j)
        
        # Energy (Angular Second Moment)
        features[2] = np.sum(glcm**2)
        
        # Homogeneity (Inverse Difference Moment)
        features[3] = np.sum(glcm / (1 + np.abs(i - j)))
        
        # Entropy
        mask = glcm > 0
        features[4] = -np.sum(glcm[mask] * np.log2(glcm[mask] + 1e-10))
        
        # Dissimilarity
        features[5] = np.sum(np.abs(i - j) * glcm)
        
        return features
    
    def get_feature_vector(self) -> np.ndarray:
        """Return the Haralick features."""
        if self._features is None:
            return np.zeros(6, dtype=np.float64)
        return self._features
    
    @property
    def name(self) -> str:
        return "haralick"
    
    def get_distance(self, other: 'Haralick') -> float:
        """Use L2 distance for comparison."""
        return dist_l2(self._features, other._features)
