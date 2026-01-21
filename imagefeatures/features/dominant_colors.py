"""
Dominant Color Descriptor.

Extracts K dominant colors from an image using simple clustering.
"""

import numpy as np
from imagefeatures.base import GlobalFeature, register_feature
from imagefeatures.utils.metrics import dist_l2


@register_feature("dominant_colors")
class DominantColors(GlobalFeature):
    """
    Dominant Color Descriptor.
    
    Extracts K dominant colors from the image. For each dominant color,
    stores the RGB values and percentage.
    
    Returns K * 4 dimensions (R, G, B, percentage for each color).
    
    Example:
        >>> from imagefeatures.features import DominantColors
        >>> from imagefeatures.utils import load_image
        >>> 
        >>> dc = DominantColors(k=5)
        >>> dc.extract(load_image('image.jpg'))
        >>> features = dc.get_feature_vector()  # 20-dim vector (5 colors x 4)
    """
    
    def __init__(self, k: int = 5, max_iter: int = 10):
        """
        Args:
            k: Number of dominant colors to extract
            max_iter: Maximum iterations for clustering
        """
        self.k = k
        self.max_iter = max_iter
        self._colors = None
    
    def extract(self, image: np.ndarray) -> None:
        """Extract dominant colors from image."""
        h, w = image.shape[:2]
        
        # Subsample for speed
        step = max(1, int(np.sqrt(h * w / 10000)))
        pixels = image[::step, ::step].reshape(-1, 3).astype(np.float64)
        
        # Simple k-means clustering
        centers, labels = self._kmeans(pixels, self.k, self.max_iter)
        
        # Compute percentage for each cluster
        counts = np.bincount(labels, minlength=self.k)
        percentages = counts / len(labels)
        
        # Sort by percentage (most dominant first)
        sorted_idx = np.argsort(percentages)[::-1]
        
        # Build feature vector: [R1, G1, B1, %1, R2, G2, B2, %2, ...]
        self._colors = np.zeros(self.k * 4, dtype=np.float64)
        for i, idx in enumerate(sorted_idx):
            self._colors[i*4] = centers[idx, 0]     # R
            self._colors[i*4+1] = centers[idx, 1]   # G
            self._colors[i*4+2] = centers[idx, 2]   # B
            self._colors[i*4+3] = percentages[idx] * 100  # %
    
    def _kmeans(self, data: np.ndarray, k: int, max_iter: int) -> tuple:
        """Simple k-means clustering."""
        n = len(data)
        
        # Initialize centers randomly
        idx = np.random.choice(n, k, replace=False)
        centers = data[idx].copy()
        
        labels = np.zeros(n, dtype=np.int32)
        
        for _ in range(max_iter):
            # Assign labels
            for i in range(n):
                distances = np.sum((centers - data[i])**2, axis=1)
                labels[i] = np.argmin(distances)
            
            # Update centers
            new_centers = np.zeros_like(centers)
            for j in range(k):
                mask = labels == j
                if np.any(mask):
                    new_centers[j] = np.mean(data[mask], axis=0)
                else:
                    new_centers[j] = centers[j]
            
            # Check convergence
            if np.allclose(centers, new_centers):
                break
            centers = new_centers
        
        return centers, labels
    
    def get_feature_vector(self) -> np.ndarray:
        """Return the dominant colors."""
        if self._colors is None:
            return np.zeros(self.k * 4, dtype=np.float64)
        return self._colors
    
    @property
    def name(self) -> str:
        return "dominant_colors"
    
    def get_distance(self, other: 'DominantColors') -> float:
        """Use L2 distance for comparison."""
        return dist_l2(self._colors, other._colors)
