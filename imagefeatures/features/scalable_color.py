"""
Scalable Color Descriptor (MPEG-7).

Hierarchical color histogram with Haar transform.
"""

import numpy as np
from imagefeatures.base import GlobalFeature, register_feature
from imagefeatures.utils.metrics import dist_l1


@register_feature("scalable_color")
class ScalableColor(GlobalFeature):
    """
    MPEG-7 Scalable Color Descriptor.
    
    Uses a hierarchical representation of a color histogram in HSV space.
    Applies Haar transform to create a scalable representation.
    
    Returns 64 dimensions by default (can be configured).
    
    Example:
        >>> from imagefeatures.features import ScalableColor
        >>> from imagefeatures.utils import load_image
        >>> 
        >>> sc = ScalableColor()
        >>> sc.extract(load_image('image.jpg'))
        >>> features = sc.get_feature_vector()  # 64-dim vector
    """
    
    def __init__(self, num_coeffs: int = 64):
        """
        Args:
            num_coeffs: Number of coefficients to keep (16, 32, 64, 128, 256)
        """
        self.num_coeffs = num_coeffs
        self._coeffs = None
    
    def extract(self, image: np.ndarray) -> None:
        """Extract scalable color descriptor from image."""
        h, w = image.shape[:2]
        
        # Convert to HSV
        hsv = self._rgb_to_hsv(image)
        
        # Create 256-bin histogram (16 H x 4 S x 4 V)
        histogram = np.zeros(256, dtype=np.float64)
        
        for y in range(h):
            for x in range(w):
                hue, sat, val = hsv[y, x]
                
                # Quantize
                h_bin = int(hue * 16 / 360) % 16
                s_bin = min(3, int(sat * 4 / 256))
                v_bin = min(3, int(val * 4 / 256))
                
                idx = h_bin * 16 + s_bin * 4 + v_bin
                histogram[idx] += 1
        
        # Normalize
        total = np.sum(histogram)
        if total > 0:
            histogram = histogram / total * 255
        
        # Apply Haar transform
        coeffs = self._haar_transform(histogram)
        
        # Keep only requested number of coefficients
        self._coeffs = coeffs[:self.num_coeffs]
    
    def _rgb_to_hsv(self, image: np.ndarray) -> np.ndarray:
        """Convert RGB to HSV."""
        h, w = image.shape[:2]
        hsv = np.zeros((h, w, 3), dtype=np.float64)
        
        for y in range(h):
            for x in range(w):
                r, g, b = image[y, x, 0], image[y, x, 1], image[y, x, 2]
                
                max_c = max(r, g, b)
                min_c = min(r, g, b)
                diff = max_c - min_c
                
                # Value
                hsv[y, x, 2] = max_c
                
                # Saturation
                hsv[y, x, 1] = 0 if max_c == 0 else (diff / max_c) * 255
                
                # Hue
                if diff == 0:
                    hsv[y, x, 0] = 0
                elif max_c == r:
                    hsv[y, x, 0] = 60 * ((g - b) / diff % 6)
                elif max_c == g:
                    hsv[y, x, 0] = 60 * ((b - r) / diff + 2)
                else:
                    hsv[y, x, 0] = 60 * ((r - g) / diff + 4)
        
        return hsv
    
    def _haar_transform(self, data: np.ndarray) -> np.ndarray:
        """Apply 1D Haar transform."""
        n = len(data)
        result = data.copy()
        
        while n > 1:
            half = n // 2
            temp = np.zeros_like(result)
            
            for i in range(half):
                temp[i] = (result[2*i] + result[2*i + 1]) / 2
                temp[half + i] = (result[2*i] - result[2*i + 1]) / 2
            
            result[:n] = temp[:n]
            n = half
        
        return result
    
    def get_feature_vector(self) -> np.ndarray:
        """Return the scalable color coefficients."""
        if self._coeffs is None:
            return np.zeros(self.num_coeffs, dtype=np.float64)
        return self._coeffs
    
    @property
    def name(self) -> str:
        return "scalable_color"
    
    def get_distance(self, other: 'ScalableColor') -> float:
        """Use L1 distance for comparison."""
        return dist_l1(self._coeffs, other._coeffs)
