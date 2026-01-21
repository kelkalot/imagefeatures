"""
Color Moments feature extractor.

First 3 moments (mean, std, skewness) of each color channel.
"""

import numpy as np
from imagefeatures.base import GlobalFeature, register_feature
from imagefeatures.utils.metrics import dist_l2


@register_feature("color_moments")
class ColorMoments(GlobalFeature):
    """
    Color Moments Descriptor.
    
    Computes the first 3 statistical moments for each color channel:
    - Mean (1st moment)
    - Standard deviation (2nd moment)
    - Skewness (3rd moment)
    
    Can use RGB or HSV color space. Returns 9 dimensions (3 channels x 3 moments).
    
    Example:
        >>> from imagefeatures.features import ColorMoments
        >>> from imagefeatures.utils import load_image
        >>> 
        >>> cm = ColorMoments()
        >>> cm.extract(load_image('image.jpg'))
        >>> features = cm.get_feature_vector()  # 9-dim vector
    """
    
    def __init__(self, color_space: str = 'rgb'):
        """
        Args:
            color_space: 'rgb' or 'hsv'
        """
        self.color_space = color_space.lower()
        self._moments = None
    
    def extract(self, image: np.ndarray) -> None:
        """Extract color moments from image."""
        if self.color_space == 'hsv':
            image = self._rgb_to_hsv(image)
        
        self._moments = np.zeros(9, dtype=np.float64)
        
        for c in range(3):
            channel = image[:, :, c].astype(np.float64).flatten()
            
            # Mean (1st moment)
            mean = np.mean(channel)
            self._moments[c * 3] = mean
            
            # Standard deviation (2nd moment)
            std = np.std(channel)
            self._moments[c * 3 + 1] = std
            
            # Skewness (3rd moment)
            if std > 0:
                skewness = np.mean(((channel - mean) / std) ** 3)
            else:
                skewness = 0
            self._moments[c * 3 + 2] = skewness
    
    def _rgb_to_hsv(self, image: np.ndarray) -> np.ndarray:
        """Convert RGB to HSV."""
        r, g, b = image[:,:,0]/255., image[:,:,1]/255., image[:,:,2]/255.
        max_c = np.maximum(np.maximum(r, g), b)
        min_c = np.minimum(np.minimum(r, g), b)
        diff = max_c - min_c
        
        h = np.zeros_like(max_c)
        s = np.zeros_like(max_c)
        v = max_c * 255
        
        mask = max_c > 0
        s[mask] = (diff[mask] / max_c[mask]) * 255
        
        mask = diff > 0
        h[mask & (max_c == r)] = 60 * ((g[mask & (max_c == r)] - b[mask & (max_c == r)]) / diff[mask & (max_c == r)] % 6)
        h[mask & (max_c == g)] = 60 * ((b[mask & (max_c == g)] - r[mask & (max_c == g)]) / diff[mask & (max_c == g)] + 2)
        h[mask & (max_c == b)] = 60 * ((r[mask & (max_c == b)] - g[mask & (max_c == b)]) / diff[mask & (max_c == b)] + 4)
        
        return np.stack([h, s, v], axis=-1).astype(np.float64)
    
    def get_feature_vector(self) -> np.ndarray:
        """Return the color moments."""
        if self._moments is None:
            return np.zeros(9, dtype=np.float64)
        return self._moments
    
    @property
    def name(self) -> str:
        return f"color_moments_{self.color_space}"
    
    def get_distance(self, other: 'ColorMoments') -> float:
        """Use L2 distance for comparison."""
        return dist_l2(self._moments, other._moments)
