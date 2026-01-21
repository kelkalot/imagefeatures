"""
Centrist (Census Transform Histogram) feature extractor.

Based on the Census Transform for texture description.

Reference:
Wu, J., Rehg, J.M.: CENTRIST: A Visual Descriptor for Scene Categorization.
IEEE PAMI 2011
"""

import numpy as np
from imagefeatures.base import GlobalFeature, register_feature
from imagefeatures.utils.metrics import dist_l1


@register_feature("centrist")
class Centrist(GlobalFeature):
    """
    CENTRIST - CENsus TRansform hISTogram.
    
    Computes Census Transform for each pixel (comparison with neighbors)
    and creates a histogram of the resulting codes.
    
    Similar to LBP but uses Census Transform (bit comparison with center).
    
    Example:
        >>> from imagefeatures.features import Centrist
        >>> from imagefeatures.utils import load_image
        >>> 
        >>> centrist = Centrist()
        >>> centrist.extract(load_image('image.jpg'))
        >>> features = centrist.get_feature_vector()  # 256-dim vector
    """
    
    def __init__(self, spatial_pyramid: bool = False):
        """
        Args:
            spatial_pyramid: If True, use 3-level spatial pyramid (results in 5*256=1280 dims)
        """
        self.spatial_pyramid = spatial_pyramid
        self._histogram = None
    
    def extract(self, image: np.ndarray) -> None:
        """Extract Centrist histogram from image."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = (0.299 * image[:, :, 0] + 
                    0.587 * image[:, :, 1] + 
                    0.114 * image[:, :, 2]).astype(np.uint8)
        else:
            gray = image.astype(np.uint8)
        
        h, w = gray.shape
        
        if self.spatial_pyramid:
            # 1 + 4 = 5 regions
            self._histogram = np.zeros(5 * 256, dtype=np.float64)
            
            # Level 0: whole image
            hist = self._compute_census_histogram(gray, 0, 0, w, h)
            self._histogram[0:256] = hist
            
            # Level 1: 2x2 grid
            idx = 256
            for i in range(2):
                for j in range(2):
                    hist = self._compute_census_histogram(
                        gray, j * w // 2, i * h // 2, w // 2, h // 2
                    )
                    self._histogram[idx:idx+256] = hist
                    idx += 256
        else:
            self._histogram = self._compute_census_histogram(gray, 0, 0, w, h)
    
    def _compute_census_histogram(self, gray: np.ndarray, 
                                   start_x: int, start_y: int,
                                   width: int, height: int) -> np.ndarray:
        """Compute Census Transform histogram for a region."""
        histogram = np.zeros(256, dtype=np.float64)
        
        end_y = min(start_y + height, gray.shape[0] - 1)
        end_x = min(start_x + width, gray.shape[1] - 1)
        
        for y in range(max(1, start_y), end_y):
            for x in range(max(1, start_x), end_x):
                center = gray[y, x]
                
                # Census Transform: 3x3 neighborhood, 8 comparisons
                code = 0
                if gray[y-1, x-1] < center: code |= 1
                if gray[y-1, x  ] < center: code |= 2
                if gray[y-1, x+1] < center: code |= 4
                if gray[y,   x+1] < center: code |= 8
                if gray[y+1, x+1] < center: code |= 16
                if gray[y+1, x  ] < center: code |= 32
                if gray[y+1, x-1] < center: code |= 64
                if gray[y,   x-1] < center: code |= 128
                
                histogram[code] += 1
        
        # Normalize
        total = np.sum(histogram)
        if total > 0:
            histogram = histogram / total * 255
        
        return histogram
    
    def get_feature_vector(self) -> np.ndarray:
        """Return the Centrist histogram."""
        if self._histogram is None:
            dim = 5 * 256 if self.spatial_pyramid else 256
            return np.zeros(dim, dtype=np.float64)
        return self._histogram
    
    @property
    def name(self) -> str:
        return "centrist" if not self.spatial_pyramid else "spatial_pyramid_centrist"
    
    def get_distance(self, other: 'Centrist') -> float:
        """Use L1 distance for comparison."""
        return dist_l1(self._histogram, other._histogram)
