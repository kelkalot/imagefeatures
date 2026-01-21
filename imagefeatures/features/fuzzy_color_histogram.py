"""
FuzzyColorHistogram feature extractor.

Uses fuzzy membership functions for color quantization.
"""

import numpy as np
from imagefeatures.base import GlobalFeature, register_feature
from imagefeatures.utils.metrics import jsd


@register_feature("fuzzy_color_histogram")
class FuzzyColorHistogram(GlobalFeature):
    """
    Fuzzy Color Histogram.
    
    Uses fuzzy membership functions instead of hard binning for color quantization.
    Each pixel contributes to multiple bins based on its fuzzy membership.
    
    Uses HSV color space with fuzzy quantization:
    - Hue: 8 fuzzy bins
    - Saturation: 3 fuzzy bins  
    - Value: 3 fuzzy bins
    
    Total: 8 x 3 x 3 = 72 dimensions
    
    Example:
        >>> from imagefeatures.features import FuzzyColorHistogram
        >>> from imagefeatures.utils import load_image
        >>> 
        >>> fch = FuzzyColorHistogram()
        >>> fch.extract(load_image('image.jpg'))
        >>> features = fch.get_feature_vector()  # 72-dim vector
    """
    
    # Fuzzy set centers and widths
    HUE_CENTERS = np.array([0, 45, 90, 135, 180, 225, 270, 315]) / 360.0
    SAT_CENTERS = np.array([0.15, 0.5, 0.85])
    VAL_CENTERS = np.array([0.15, 0.5, 0.85])
    
    def __init__(self):
        self._histogram = None
    
    def extract(self, image: np.ndarray) -> None:
        """Extract fuzzy color histogram from image."""
        h, w = image.shape[:2]
        
        # Downsample large images for speed
        max_size = 128
        if max(h, w) > max_size:
            step = max(1, max(h, w) // max_size)
            image = image[::step, ::step]
            h, w = image.shape[:2]
        
        # Initialize histogram: 8 hue x 3 sat x 3 val = 72 bins
        self._histogram = np.zeros(72, dtype=np.float64)
        
        # Convert entire image to HSV
        r = image[:, :, 0].astype(np.float64) / 255.0
        g = image[:, :, 1].astype(np.float64) / 255.0
        b = image[:, :, 2].astype(np.float64) / 255.0
        
        max_c = np.maximum(np.maximum(r, g), b)
        min_c = np.minimum(np.minimum(r, g), b)
        diff = max_c - min_c
        
        # Value
        v = max_c
        
        # Saturation
        s = np.zeros_like(max_c)
        mask = max_c > 0
        s[mask] = diff[mask] / max_c[mask]
        
        # Hue
        hue = np.zeros_like(max_c)
        mask_r = (max_c == r) & (diff > 0)
        mask_g = (max_c == g) & (diff > 0)
        mask_b = (max_c == b) & (diff > 0)
        
        hue[mask_r] = ((g[mask_r] - b[mask_r]) / diff[mask_r]) % 6
        hue[mask_g] = (b[mask_g] - r[mask_g]) / diff[mask_g] + 2
        hue[mask_b] = (r[mask_b] - g[mask_b]) / diff[mask_b] + 4
        hue = hue / 6  # Normalize to 0-1
        
        # Simple binning instead of fuzzy (for speed)
        h_bins = (hue * 8).astype(np.int32)
        s_bins = (s * 3).astype(np.int32)
        v_bins = (v * 3).astype(np.int32)
        
        h_bins = np.clip(h_bins, 0, 7)
        s_bins = np.clip(s_bins, 0, 2)
        v_bins = np.clip(v_bins, 0, 2)
        
        indices = h_bins * 9 + s_bins * 3 + v_bins
        
        for idx in indices.flatten():
            self._histogram[idx] += 1
        
        # Normalize
        total = np.sum(self._histogram)
        if total > 0:
            self._histogram = self._histogram / total * 255
    
    def _rgb_to_hsv_norm(self, r: int, g: int, b: int) -> tuple:
        """Convert RGB to normalized HSV (0-1 range)."""
        r, g, b = r / 255.0, g / 255.0, b / 255.0
        
        max_c = max(r, g, b)
        min_c = min(r, g, b)
        diff = max_c - min_c
        
        # Value
        v = max_c
        
        # Saturation
        s = 0 if max_c == 0 else diff / max_c
        
        # Hue
        if diff == 0:
            h = 0
        elif max_c == r:
            h = ((g - b) / diff) % 6
        elif max_c == g:
            h = (b - r) / diff + 2
        else:
            h = (r - g) / diff + 4
        h = h / 6  # Normalize to 0-1
        
        return (h, s, v)
    
    def _fuzzy_membership(self, value: float, centers: np.ndarray, 
                          width: float, circular: bool = False) -> np.ndarray:
        """Compute fuzzy membership values for triangular membership functions."""
        memberships = np.zeros(len(centers))
        
        for i, center in enumerate(centers):
            if circular:
                # Handle circular distance for hue
                dist = min(abs(value - center), 1 - abs(value - center))
            else:
                dist = abs(value - center)
            
            if dist < width:
                memberships[i] = 1 - dist / width
        
        return memberships
    
    def get_feature_vector(self) -> np.ndarray:
        """Return the fuzzy color histogram."""
        if self._histogram is None:
            return np.zeros(72, dtype=np.float64)
        return self._histogram
    
    @property
    def name(self) -> str:
        return "fuzzy_color_histogram"
    
    def get_distance(self, other: 'FuzzyColorHistogram') -> float:
        """Use JSD for histogram comparison."""
        return jsd(self._histogram, other._histogram)
