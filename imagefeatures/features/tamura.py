"""
Tamura texture feature extractor.

Computes three Tamura texture features: coarseness, contrast, and directionality.
"""

import numpy as np
from imagefeatures.base import GlobalFeature, register_feature
from imagefeatures.utils.metrics import dist_l2


@register_feature("tamura")
class Tamura(GlobalFeature):
    """
    Tamura texture features: coarseness, contrast, and directionality.
    
    These three features capture perceptual texture properties:
    - Coarseness: granularity of the texture
    - Contrast: dynamic range and sharpness
    - Directionality: presence of oriented patterns
    
    Returns a 18-dimensional histogram (3 histograms of 6 bins each, one per feature).
    
    Based on the LIRE implementation by Marko Keuschnig & Christian Penz.
    
    Example:
        >>> from imagefeatures.features import Tamura
        >>> from imagefeatures.utils import load_image
        >>> 
        >>> tamura = Tamura()
        >>> tamura.extract(load_image('image.jpg'))
        >>> features = tamura.get_feature_vector()
    """
    
    def __init__(self):
        self._histogram = None
        self._coarseness = 0.0
        self._contrast = 0.0
        self._directionality = 0.0
    
    def extract(self, image: np.ndarray) -> None:
        """Extract Tamura features from image."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = (0.299 * image[:, :, 0] + 
                    0.587 * image[:, :, 1] + 
                    0.114 * image[:, :, 2]).astype(np.float64)
        else:
            gray = image.astype(np.float64)
        
        h, w = gray.shape
        
        # Compute features
        self._coarseness = self._compute_coarseness(gray)
        self._contrast = self._compute_contrast(gray)
        self._directionality = self._compute_directionality(gray)
        
        # Create histogram representation (3 features x 6 bins each = 18 dims)
        self._histogram = np.zeros(18, dtype=np.float64)
        
        # Quantize coarseness (0-1 range typical)
        coarse_bin = min(5, int(self._coarseness * 6))
        self._histogram[coarse_bin] = 1.0
        
        # Quantize contrast (can be large, normalize to ~0-100)
        contrast_norm = min(1.0, self._contrast / 100.0)
        contrast_bin = min(5, int(contrast_norm * 6))
        self._histogram[6 + contrast_bin] = 1.0
        
        # Quantize directionality (0-1 range)
        dir_bin = min(5, int(self._directionality * 6))
        self._histogram[12 + dir_bin] = 1.0
    
    def _compute_coarseness(self, gray: np.ndarray) -> float:
        """Compute coarseness feature (optimized)."""
        h, w = gray.shape
        
        # Downsample large images for speed
        max_size = 128
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            # Simple downsampling by slicing
            step_h = max(1, h // new_h)
            step_w = max(1, w // new_w)
            gray = gray[::step_h, ::step_w]
            h, w = gray.shape
        
        # Use simplified coarseness based on variance at different scales
        coarseness_values = []
        
        for k in range(1, 4):  # Reduced scales for speed
            size = 2 ** k
            if size >= min(h, w) // 2:
                break
            
            # Compute local variance using block averaging
            # Pad and reshape for block averaging
            pad_h = (size - h % size) % size
            pad_w = (size - w % size) % size
            
            if pad_h > 0 or pad_w > 0:
                gray_padded = np.pad(gray, ((0, pad_h), (0, pad_w)), mode='edge')
            else:
                gray_padded = gray
            
            # Block average
            new_h = gray_padded.shape[0] // size
            new_w = gray_padded.shape[1] // size
            blocks = gray_padded[:new_h*size, :new_w*size].reshape(new_h, size, new_w, size)
            block_means = blocks.mean(axis=(1, 3))
            
            # Compute differences between adjacent blocks
            diff_h = np.abs(np.diff(block_means, axis=0))
            diff_w = np.abs(np.diff(block_means, axis=1))
            
            max_diff = max(diff_h.mean() if diff_h.size > 0 else 0,
                          diff_w.mean() if diff_w.size > 0 else 0)
            coarseness_values.append((k, max_diff))
        
        if not coarseness_values:
            return 0.0
        
        # Best scale is where difference is maximum
        best_k = max(coarseness_values, key=lambda x: x[1])[0]
        return best_k / 5.0  # Normalize to 0-1
    
    def _compute_contrast(self, gray: np.ndarray) -> float:
        """Compute contrast feature."""
        # Compute 4th moment (kurtosis) for contrast
        mean = np.mean(gray)
        std = np.std(gray)
        
        if std == 0:
            return 0.0
        
        # Kurtosis
        n = gray.size
        m4 = np.mean((gray - mean) ** 4)
        m2 = np.mean((gray - mean) ** 2)
        
        if m2 == 0:
            return 0.0
        
        kurtosis = m4 / (m2 ** 2)
        
        # Contrast formula: sigma / (kurtosis ** 0.25)
        if kurtosis > 0:
            contrast = std / (kurtosis ** 0.25)
        else:
            contrast = std
        
        return contrast
    
    def _compute_directionality(self, gray: np.ndarray) -> float:
        """Compute directionality feature."""
        h, w = gray.shape
        
        # Compute gradients using simple Prewitt-like operators
        # Horizontal gradient
        delta_h = np.zeros((h, w), dtype=np.float64)
        delta_h[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
        
        # Vertical gradient
        delta_v = np.zeros((h, w), dtype=np.float64)
        delta_v[1:-1, :] = gray[2:, :] - gray[:-2, :]
        
        # Gradient magnitude and direction
        magnitude = np.sqrt(delta_h ** 2 + delta_v ** 2)
        
        # Threshold for edge detection
        threshold = 0.1 * np.max(magnitude)
        edge_mask = magnitude > threshold
        
        if not np.any(edge_mask):
            return 0.0
        
        # Compute direction histogram (16 bins from 0 to pi)
        direction = np.arctan2(delta_v, delta_h)
        direction = np.mod(direction, np.pi)  # Map to [0, pi]
        
        # Create histogram of edge directions
        dir_hist = np.zeros(16, dtype=np.float64)
        dir_indices = (direction[edge_mask] * 16 / np.pi).astype(np.int32)
        dir_indices = np.clip(dir_indices, 0, 15)
        
        for idx in dir_indices.flatten():
            dir_hist[idx] += 1
        
        # Normalize
        if np.sum(dir_hist) > 0:
            dir_hist /= np.sum(dir_hist)
        
        # Directionality = 1 - entropy (high value = strong direction)
        # Use peak sharpness instead
        max_peak = np.max(dir_hist)
        return max_peak
    
    def get_feature_vector(self) -> np.ndarray:
        """Return the Tamura histogram."""
        if self._histogram is None:
            return np.zeros(18, dtype=np.float64)
        return self._histogram
    
    @property
    def name(self) -> str:
        return "tamura"
    
    def get_distance(self, other: 'Tamura') -> float:
        """Use L2 distance for Tamura comparison."""
        return dist_l2(self._histogram, other._histogram)
    
    @property
    def coarseness(self) -> float:
        """Return coarseness value."""
        return self._coarseness
    
    @property
    def contrast(self) -> float:
        """Return contrast value."""
        return self._contrast
    
    @property
    def directionality(self) -> float:
        """Return directionality value."""
        return self._directionality
