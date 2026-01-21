"""
Rotation Invariant Local Binary Patterns feature extractor.

A rotation-invariant variant of LBP that maps patterns to their minimum rotation.
"""

import numpy as np
from imagefeatures.base import GlobalFeature, register_feature
from imagefeatures.utils.metrics import dist_l1


@register_feature("rotation_invariant_lbp")
class RotationInvariantLBP(GlobalFeature):
    """
    Rotation Invariant Local Binary Patterns.
    
    Similar to standard LBP but maps each pattern to its minimum rotation,
    reducing the feature from 256 bins to 36 unique rotation-invariant patterns.
    
    Example:
        >>> from imagefeatures.features import RotationInvariantLBP
        >>> from imagefeatures.utils import load_image
        >>> 
        >>> rilbp = RotationInvariantLBP()
        >>> rilbp.extract(load_image('image.jpg'))
        >>> features = rilbp.get_feature_vector()  # 36-dim vector
    """
    
    # Precomputed mapping from 256 LBP codes to rotation-invariant equivalents
    # Each code maps to its minimum rotation
    _RI_MAP = None
    
    def __init__(self):
        self._histogram = None
        if RotationInvariantLBP._RI_MAP is None:
            RotationInvariantLBP._RI_MAP = self._compute_ri_map()
    
    def _compute_ri_map(self) -> np.ndarray:
        """Compute rotation-invariant mapping for all 256 LBP codes."""
        ri_map = np.zeros(256, dtype=np.int32)
        unique_patterns = {}
        next_id = 0
        
        for code in range(256):
            # Find minimum rotation
            min_rot = code
            current = code
            for _ in range(7):
                # Rotate bits left
                current = ((current << 1) | (current >> 7)) & 0xFF
                min_rot = min(min_rot, current)
            
            if min_rot not in unique_patterns:
                unique_patterns[min_rot] = next_id
                next_id += 1
            
            ri_map[code] = unique_patterns[min_rot]
        
        return ri_map
    
    def extract(self, image: np.ndarray) -> None:
        """Extract rotation-invariant LBP histogram from image."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = (0.299 * image[:, :, 0] + 
                    0.587 * image[:, :, 1] + 
                    0.114 * image[:, :, 2]).astype(np.uint8)
        else:
            gray = image.astype(np.uint8)
        
        h, w = gray.shape
        
        # Get number of unique RI patterns
        num_patterns = len(set(self._RI_MAP))
        self._histogram = np.zeros(num_patterns, dtype=np.float64)
        
        # Compute LBP for each pixel
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                center = gray[y, x]
                
                # Compute 8-bit LBP code
                code = 0
                if gray[y-1, x-1] >= center: code |= 1
                if gray[y-1, x  ] >= center: code |= 2
                if gray[y-1, x+1] >= center: code |= 4
                if gray[y,   x+1] >= center: code |= 8
                if gray[y+1, x+1] >= center: code |= 16
                if gray[y+1, x  ] >= center: code |= 32
                if gray[y+1, x-1] >= center: code |= 64
                if gray[y,   x-1] >= center: code |= 128
                
                # Map to rotation-invariant pattern
                ri_code = self._RI_MAP[code]
                self._histogram[ri_code] += 1
        
        # Normalize
        max_val = np.max(self._histogram)
        if max_val > 0:
            self._histogram = self._histogram / max_val * 127
    
    def get_feature_vector(self) -> np.ndarray:
        """Return the RI-LBP histogram."""
        if self._histogram is None:
            return np.zeros(36, dtype=np.float64)
        return self._histogram
    
    @property
    def name(self) -> str:
        return "rotation_invariant_lbp"
    
    def get_distance(self, other: 'RotationInvariantLBP') -> float:
        """Use L1 distance for comparison."""
        return dist_l1(self._histogram, other._histogram)
