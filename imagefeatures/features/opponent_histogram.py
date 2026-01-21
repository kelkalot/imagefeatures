"""
OpponentHistogram feature extractor.

Color histogram in the opponent color space (O1, O2, O3).
"""

import numpy as np
from imagefeatures.base import GlobalFeature, register_feature
from imagefeatures.utils.metrics import jsd


@register_feature("opponent_histogram")
class OpponentHistogram(GlobalFeature):
    """
    Opponent Color Histogram.
    
    Computes a histogram in the opponent color space:
    - O1 = (R - G) / sqrt(2)
    - O2 = (R + G - 2B) / sqrt(6)
    - O3 = (R + G + B) / sqrt(3)
    
    Uses 32 bins per channel = 32768 dimensions (reduced to 512 via binning).
    
    Example:
        >>> from imagefeatures.features import OpponentHistogram
        >>> from imagefeatures.utils import load_image
        >>> 
        >>> oh = OpponentHistogram()
        >>> oh.extract(load_image('image.jpg'))
        >>> features = oh.get_feature_vector()  # 512-dim vector
    """
    
    BINS_PER_CHANNEL = 8  # 8x8x8 = 512 bins
    
    def __init__(self):
        self._histogram = None
    
    def extract(self, image: np.ndarray) -> None:
        """Extract opponent color histogram from image."""
        h, w = image.shape[:2]
        
        # Convert to float
        r = image[:, :, 0].astype(np.float64)
        g = image[:, :, 1].astype(np.float64)
        b = image[:, :, 2].astype(np.float64)
        
        # Convert to opponent color space
        sqrt2 = np.sqrt(2)
        sqrt6 = np.sqrt(6)
        sqrt3 = np.sqrt(3)
        
        o1 = (r - g) / sqrt2  # Range: ~[-180, 180]
        o2 = (r + g - 2*b) / sqrt6  # Range: ~[-208, 208]
        o3 = (r + g + b) / sqrt3  # Range: [0, ~441]
        
        # Normalize to [0, 1] range
        o1 = (o1 + 181) / 362  # Shift and scale
        o2 = (o2 + 209) / 418
        o3 = o3 / 442
        
        # Clip to valid range
        o1 = np.clip(o1, 0, 0.9999)
        o2 = np.clip(o2, 0, 0.9999)
        o3 = np.clip(o3, 0, 0.9999)
        
        # Quantize to bins
        bins = self.BINS_PER_CHANNEL
        o1_q = (o1 * bins).astype(np.int32)
        o2_q = (o2 * bins).astype(np.int32)
        o3_q = (o3 * bins).astype(np.int32)
        
        # Create histogram
        self._histogram = np.zeros(bins ** 3, dtype=np.float64)
        indices = o1_q * (bins ** 2) + o2_q * bins + o3_q
        
        for idx in indices.flatten():
            self._histogram[idx] += 1
        
        # Normalize
        total = np.sum(self._histogram)
        if total > 0:
            self._histogram = self._histogram / total * 255
    
    def get_feature_vector(self) -> np.ndarray:
        """Return the opponent histogram."""
        if self._histogram is None:
            return np.zeros(self.BINS_PER_CHANNEL ** 3, dtype=np.float64)
        return self._histogram
    
    @property
    def name(self) -> str:
        return "opponent_histogram"
    
    def get_distance(self, other: 'OpponentHistogram') -> float:
        """Use JSD for histogram comparison."""
        return jsd(self._histogram, other._histogram)
