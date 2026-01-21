"""
AutoColorCorrelogram feature extractor.

Captures spatial correlation of colors at multiple distances.

Reference:
Huang, J., Kumar, S.R., Mitra, M., Zhu, W., & Zabih, R.: 
Image Indexing Using Color Correlograms. CVPR 1997
"""

import numpy as np
from imagefeatures.base import GlobalFeature, register_feature
from imagefeatures.utils.metrics import dist_l1


@register_feature("auto_color_correlogram")
class AutoColorCorrelogram(GlobalFeature):
    """
    Auto Color Correlogram.
    
    Measures the probability of finding a pixel of the same color at 
    distance d from a given pixel. Uses quantized colors and multiple distances.
    
    Parameters:
    - 64 quantized colors (4 levels per RGB channel)
    - 4 distance values: 1, 3, 5, 7
    
    Total: 64 colors x 4 distances = 256 dimensions
    
    Example:
        >>> from imagefeatures.features import AutoColorCorrelogram
        >>> from imagefeatures.utils import load_image
        >>> 
        >>> acc = AutoColorCorrelogram()
        >>> acc.extract(load_image('image.jpg'))
        >>> features = acc.get_feature_vector()  # 256-dim vector
    """
    
    NUM_COLORS = 64  # 4x4x4 RGB quantization
    DISTANCES = [1, 3, 5, 7]  # Pixel distances to check
    
    def __init__(self):
        self._correlogram = None
    
    def extract(self, image: np.ndarray) -> None:
        """Extract auto color correlogram from image."""
        h, w = image.shape[:2]
        
        # Quantize colors to 64 bins (4 levels per channel)
        quantized = self._quantize_image(image)
        
        # Initialize correlogram: 64 colors x 4 distances
        correlogram = np.zeros((self.NUM_COLORS, len(self.DISTANCES)), dtype=np.float64)
        color_counts = np.zeros(self.NUM_COLORS, dtype=np.float64)
        
        # Count color occurrences
        for y in range(h):
            for x in range(w):
                color = quantized[y, x]
                color_counts[color] += 1
        
        # Sample pixels for efficiency (don't check every pixel)
        step = max(1, min(h, w) // 50)
        
        # For each sampled pixel, check neighbors at each distance
        for y in range(0, h, step):
            for x in range(0, w, step):
                center_color = quantized[y, x]
                
                for d_idx, dist in enumerate(self.DISTANCES):
                    match_count = 0
                    neighbor_count = 0
                    
                    # Check pixels at distance 'dist' (8 directions)
                    for dy, dx in [(-dist, 0), (dist, 0), (0, -dist), (0, dist),
                                   (-dist, -dist), (-dist, dist), (dist, -dist), (dist, dist)]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            neighbor_count += 1
                            if quantized[ny, nx] == center_color:
                                match_count += 1
                    
                    if neighbor_count > 0:
                        correlogram[center_color, d_idx] += match_count / neighbor_count
        
        # Normalize by color frequency
        for c in range(self.NUM_COLORS):
            if color_counts[c] > 0:
                correlogram[c, :] /= (color_counts[c] / (h * w / (step * step)))
        
        # Flatten and normalize
        self._correlogram = correlogram.flatten()
        max_val = np.max(self._correlogram)
        if max_val > 0:
            self._correlogram = self._correlogram / max_val
    
    def _quantize_image(self, image: np.ndarray) -> np.ndarray:
        """Quantize RGB image to 64 color indices."""
        h, w = image.shape[:2]
        result = np.zeros((h, w), dtype=np.int32)
        
        # 4 levels per channel: 0-63, 64-127, 128-191, 192-255
        r_q = (image[:, :, 0] // 64).astype(np.int32)
        g_q = (image[:, :, 1] // 64).astype(np.int32)
        b_q = (image[:, :, 2] // 64).astype(np.int32)
        
        # Combine into single index
        result = r_q * 16 + g_q * 4 + b_q
        
        return np.clip(result, 0, 63)
    
    def get_feature_vector(self) -> np.ndarray:
        """Return the correlogram."""
        if self._correlogram is None:
            return np.zeros(self.NUM_COLORS * len(self.DISTANCES), dtype=np.float64)
        return self._correlogram
    
    @property
    def name(self) -> str:
        return "auto_color_correlogram"
    
    def get_distance(self, other: 'AutoColorCorrelogram') -> float:
        """Use L1 distance for correlogram comparison."""
        return dist_l1(self._correlogram, other._correlogram)
