"""
Color Histogram feature extractor.

Computes a color histogram in various color spaces (RGB, HSV, Luminance).
"""

import numpy as np
from imagefeatures.base import GlobalFeature, register_feature
from imagefeatures.utils.color import rgb_to_hsv_array
from imagefeatures.utils.metrics import jsd


@register_feature("color_histogram")
class ColorHistogram(GlobalFeature):
    """
    Simple color histogram feature.
    
    Computes a histogram of colors in the specified color space.
    Default is RGB with 64 bins (4 bins per channel = 4x4x4).
    
    Args:
        color_space: One of 'rgb', 'hsv', 'luminance'
        bins: Number of bins in the histogram
    
    Example:
        >>> from imagefeatures.features import ColorHistogram
        >>> from imagefeatures.utils import load_image
        >>> 
        >>> hist = ColorHistogram(color_space='rgb', bins=64)
        >>> hist.extract(load_image('image.jpg'))
        >>> features = hist.get_feature_vector()
    """
    
    def __init__(self, color_space: str = 'rgb', bins: int = 64):
        self.color_space = color_space.lower()
        self.bins = bins
        self._histogram = None
        
        # Compute bins per channel for RGB/HSV
        self._bins_per_channel = int(round(bins ** (1/3)))
        if self._bins_per_channel ** 3 != bins and self.color_space != 'luminance':
            self._bins_per_channel = 4  # Default to 4x4x4 = 64
            self.bins = 64
    
    def extract(self, image: np.ndarray) -> None:
        """Extract color histogram from image."""
        if self.color_space == 'hsv':
            self._extract_hsv(image)
        elif self.color_space == 'luminance':
            self._extract_luminance(image)
        else:  # rgb
            self._extract_rgb(image)
        
        # Normalize histogram
        max_val = np.max(self._histogram)
        if max_val > 0:
            self._histogram = (self._histogram / max_val * 255).astype(np.float64)
    
    def _extract_rgb(self, image: np.ndarray) -> None:
        """Extract RGB histogram."""
        bins_per_ch = self._bins_per_channel
        self._histogram = np.zeros(bins_per_ch ** 3, dtype=np.float64)
        
        # Quantize each channel
        r_q = (image[:, :, 0] * bins_per_ch / 256).astype(np.int32)
        g_q = (image[:, :, 1] * bins_per_ch / 256).astype(np.int32)
        b_q = (image[:, :, 2] * bins_per_ch / 256).astype(np.int32)
        
        # Clip to valid range
        r_q = np.clip(r_q, 0, bins_per_ch - 1)
        g_q = np.clip(g_q, 0, bins_per_ch - 1)
        b_q = np.clip(b_q, 0, bins_per_ch - 1)
        
        # Compute bin indices
        indices = r_q * (bins_per_ch ** 2) + g_q * bins_per_ch + b_q
        
        # Count occurrences
        for idx in indices.flatten():
            self._histogram[idx] += 1
    
    def _extract_hsv(self, image: np.ndarray) -> None:
        """Extract HSV histogram."""
        hsv = rgb_to_hsv_array(image)
        
        # Quantization: H has 32 bins, S has 4 bins, V has 4 bins = 512 bins
        # Or for 64 bins: H=4, S=4, V=4
        bins_per_ch = self._bins_per_channel
        self._histogram = np.zeros(bins_per_ch ** 3, dtype=np.float64)
        
        # Quantize
        h_q = (hsv[:, :, 0] * bins_per_ch / 360).astype(np.int32)
        s_q = (hsv[:, :, 1] * bins_per_ch / 100).astype(np.int32)
        v_q = (hsv[:, :, 2] * bins_per_ch / 100).astype(np.int32)
        
        h_q = np.clip(h_q, 0, bins_per_ch - 1)
        s_q = np.clip(s_q, 0, bins_per_ch - 1)
        v_q = np.clip(v_q, 0, bins_per_ch - 1)
        
        indices = h_q * (bins_per_ch ** 2) + s_q * bins_per_ch + v_q
        
        for idx in indices.flatten():
            self._histogram[idx] += 1
    
    def _extract_luminance(self, image: np.ndarray) -> None:
        """Extract luminance histogram."""
        # Convert to grayscale
        gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        gray = gray.astype(np.uint8)
        
        # Simple histogram
        self._histogram = np.zeros(self.bins, dtype=np.float64)
        indices = (gray * self.bins / 256).astype(np.int32)
        indices = np.clip(indices, 0, self.bins - 1)
        
        for idx in indices.flatten():
            self._histogram[idx] += 1
    
    def get_feature_vector(self) -> np.ndarray:
        """Return the color histogram."""
        if self._histogram is None:
            return np.zeros(self.bins, dtype=np.float64)
        return self._histogram
    
    @property
    def name(self) -> str:
        return f"color_histogram_{self.color_space}"
    
    def get_distance(self, other: 'ColorHistogram') -> float:
        """Use Jensen-Shannon divergence for histogram comparison."""
        return jsd(self._histogram, other._histogram)
