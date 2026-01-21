"""
PHOG (Pyramid Histogram of Oriented Gradients) feature extractor.

PHOG combines edge orientation histograms at multiple spatial pyramid levels.

Reference:
Bosch, A., Zisserman, A., & Munoz, X.: Representing shape with a spatial pyramid kernel. CVIR 2007
"""

import numpy as np
from imagefeatures.base import GlobalFeature, register_feature
from imagefeatures.utils.metrics import dist_l1


@register_feature("phog")
class PHOG(GlobalFeature):
    """
    Pyramid Histogram of Oriented Gradients.
    
    Combines edge orientation histograms at multiple spatial pyramid levels:
    - Level 0: 1 histogram (whole image)
    - Level 1: 4 histograms (2x2 grid)
    - Level 2: 16 histograms (4x4 grid)
    
    Total: 21 histograms x 30 bins = 630 dimensions
    
    Example:
        >>> from imagefeatures.features import PHOG
        >>> from imagefeatures.utils import load_image
        >>> 
        >>> phog = PHOG()
        >>> phog.extract(load_image('image.jpg'))
        >>> features = phog.get_feature_vector()  # 630-dim vector
    """
    
    BINS = 30  # Number of orientation bins
    
    def __init__(self, levels: int = 3):
        """
        Args:
            levels: Number of pyramid levels (default: 3 for levels 0, 1, 2)
        """
        self.levels = levels
        self._histogram = None
    
    def extract(self, image: np.ndarray) -> None:
        """Extract PHOG features from image."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = (0.299 * image[:, :, 0] + 
                    0.587 * image[:, :, 1] + 
                    0.114 * image[:, :, 2]).astype(np.float64)
        else:
            gray = image.astype(np.float64)
        
        h, w = gray.shape
        
        # Compute gradients using Sobel
        gx, gy = self._sobel_filter(gray)
        
        # Compute gradient magnitude and direction
        gm = np.sqrt(gx**2 + gy**2)
        gd = np.arctan2(gy, gx)  # Range: [-pi, pi]
        
        # Create edge mask using simple thresholding
        threshold_low = 15
        edge_mask = gm > threshold_low
        
        # Initialize histogram: 1 + 4 + 16 = 21 regions x 30 bins
        total_histograms = 1 + 4 + 16  # levels 0, 1, 2
        self._histogram = np.zeros(total_histograms * self.BINS, dtype=np.float64)
        
        idx = 0
        
        # Level 0: whole image
        hist = self._get_histogram(0, 0, w, h, edge_mask, gd)
        self._histogram[idx:idx + self.BINS] = hist
        idx += self.BINS
        
        # Level 1: 2x2 grid
        for i in range(2):
            for j in range(2):
                hist = self._get_histogram(
                    j * w // 2, i * h // 2,
                    w // 2, h // 2,
                    edge_mask, gd
                )
                self._histogram[idx:idx + self.BINS] = hist
                idx += self.BINS
        
        # Level 2: 4x4 grid
        for i in range(4):
            for j in range(4):
                hist = self._get_histogram(
                    j * w // 4, i * h // 4,
                    w // 4, h // 4,
                    edge_mask, gd
                )
                self._histogram[idx:idx + self.BINS] = hist
                idx += self.BINS
    
    def _sobel_filter(self, gray: np.ndarray) -> tuple:
        """Apply Sobel filter to compute gradients (vectorized)."""
        h, w = gray.shape
        
        # Use numpy operations for efficiency
        gx = np.zeros((h, w), dtype=np.float64)
        gy = np.zeros((h, w), dtype=np.float64)
        
        # Sobel X gradient (vectorized)
        gx[1:-1, 1:-1] = (
            -gray[:-2, :-2] + gray[:-2, 2:] +
            -2*gray[1:-1, :-2] + 2*gray[1:-1, 2:] +
            -gray[2:, :-2] + gray[2:, 2:]
        )
        
        # Sobel Y gradient (vectorized)
        gy[1:-1, 1:-1] = (
            -gray[:-2, :-2] - 2*gray[:-2, 1:-1] - gray[:-2, 2:] +
            gray[2:, :-2] + 2*gray[2:, 1:-1] + gray[2:, 2:]
        )
        
        return gx, gy
    
    def _get_histogram(self, start_x: int, start_y: int, 
                       width: int, height: int,
                       edge_mask: np.ndarray, gd: np.ndarray) -> np.ndarray:
        """Compute orientation histogram for a region."""
        hist = np.zeros(self.BINS, dtype=np.float64)
        
        end_y = min(start_y + height, edge_mask.shape[0])
        end_x = min(start_x + width, edge_mask.shape[1])
        
        for y in range(start_y, end_y):
            for x in range(start_x, end_x):
                if edge_mask[y, x]:
                    # Map direction from [-pi, pi] to [0, 1)
                    angle = (gd[y, x] / np.pi + 0.5)  # Now in [0, 1)
                    bin_idx = int(angle * self.BINS)
                    if bin_idx >= self.BINS:
                        bin_idx = 0
                    hist[bin_idx] += 1
        
        # Normalize to max
        max_val = np.max(hist)
        if max_val > 0:
            hist = hist / max_val * 15  # Quantize to 0-15
        
        return hist
    
    def get_feature_vector(self) -> np.ndarray:
        """Return the PHOG histogram."""
        if self._histogram is None:
            return np.zeros(630, dtype=np.float64)
        return self._histogram
    
    @property
    def name(self) -> str:
        return "phog"
    
    def get_distance(self, other: 'PHOG') -> float:
        """Use L1 distance for PHOG comparison."""
        return dist_l1(self._histogram, other._histogram)
