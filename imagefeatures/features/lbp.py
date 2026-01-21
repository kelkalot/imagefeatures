"""
Local Binary Patterns (LBP) feature extractor.

Computes a histogram of local binary patterns for texture description.
"""

import numpy as np
from imagefeatures.base import GlobalFeature, register_feature
from imagefeatures.utils.metrics import dist_l1


@register_feature("lbp")
class LocalBinaryPatterns(GlobalFeature):
    """
    Local Binary Patterns texture descriptor.
    
    Computes a 256-bin histogram of LBP codes. Each pixel is compared to its
    8 neighbors in a 3x3 window, producing an 8-bit binary code.
    
    Example:
        >>> from imagefeatures.features import LocalBinaryPatterns
        >>> from imagefeatures.utils import load_image
        >>> 
        >>> lbp = LocalBinaryPatterns()
        >>> lbp.extract(load_image('image.jpg'))
        >>> features = lbp.get_feature_vector()  # 256-dim vector
    """
    
    def __init__(self):
        self._histogram = None
    
    def extract(self, image: np.ndarray) -> None:
        """Extract LBP histogram from image."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = (0.299 * image[:, :, 0] + 
                    0.587 * image[:, :, 1] + 
                    0.114 * image[:, :, 2]).astype(np.uint8)
        else:
            gray = image.astype(np.uint8)
        
        h, w = gray.shape
        self._histogram = np.zeros(256, dtype=np.float64)
        
        # LBP computation with 3x3 neighborhood
        # Pattern: 0 1 2
        #          3 c 4
        #          5 6 7
        # Binary weights: 1, 2, 4, 8, 16, 32, 64, 128
        
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                center = gray[y, x]
                
                # Compare with 8 neighbors
                code = 0
                if gray[y-1, x-1] >= center: code |= 1
                if gray[y-1, x  ] >= center: code |= 2
                if gray[y-1, x+1] >= center: code |= 4
                if gray[y,   x+1] >= center: code |= 8
                if gray[y+1, x+1] >= center: code |= 16
                if gray[y+1, x  ] >= center: code |= 32
                if gray[y+1, x-1] >= center: code |= 64
                if gray[y,   x-1] >= center: code |= 128
                
                self._histogram[code] += 1
        
        # Normalize histogram
        max_val = np.max(self._histogram)
        if max_val > 0:
            self._histogram = (self._histogram / max_val * 127).astype(np.float64)
    
    def get_feature_vector(self) -> np.ndarray:
        """Return the LBP histogram."""
        if self._histogram is None:
            return np.zeros(256, dtype=np.float64)
        return self._histogram
    
    @property
    def name(self) -> str:
        return "lbp"
    
    def get_distance(self, other: 'LocalBinaryPatterns') -> float:
        """Use L1 distance for LBP comparison."""
        return dist_l1(self._histogram, other._histogram)
