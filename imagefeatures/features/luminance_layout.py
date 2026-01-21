"""
LuminanceLayout feature extractor.

A simplified Color Layout variant using only luminance (Y) channel.
"""

import numpy as np
from imagefeatures.base import GlobalFeature, register_feature
from imagefeatures.utils.metrics import dist_l2


@register_feature("luminance_layout")
class LuminanceLayout(GlobalFeature):
    """
    Luminance Layout Descriptor.
    
    Similar to Color Layout but uses only the luminance channel.
    Divides image into 8x8 grid and applies DCT.
    
    Default: 64 DCT coefficients
    
    Example:
        >>> from imagefeatures.features import LuminanceLayout
        >>> from imagefeatures.utils import load_image
        >>> 
        >>> ll = LuminanceLayout()
        >>> ll.extract(load_image('image.jpg'))
        >>> features = ll.get_feature_vector()  # 64-dim vector
    """
    
    def __init__(self, num_coeffs: int = 64):
        """
        Args:
            num_coeffs: Number of DCT coefficients to keep (1-64)
        """
        self.num_coeffs = min(64, max(1, num_coeffs))
        self._coeffs = None
    
    def extract(self, image: np.ndarray) -> None:
        """Extract luminance layout from image."""
        h, w = image.shape[:2]
        
        # Convert to grayscale (luminance)
        gray = (0.299 * image[:, :, 0] + 
                0.587 * image[:, :, 1] + 
                0.114 * image[:, :, 2])
        
        # Create 8x8 representative luminance array
        lum_grid = np.zeros((8, 8), dtype=np.float64)
        
        block_h = h // 8
        block_w = w // 8
        
        for by in range(8):
            for bx in range(8):
                y_start = by * block_h
                y_end = y_start + block_h if by < 7 else h
                x_start = bx * block_w
                x_end = x_start + block_w if bx < 7 else w
                
                lum_grid[by, bx] = np.mean(gray[y_start:y_end, x_start:x_end])
        
        # Apply DCT
        dct = self._dct_8x8(lum_grid)
        
        # Extract coefficients in zigzag order
        self._coeffs = self._zigzag_scan(dct, self.num_coeffs)
    
    def _dct_8x8(self, block: np.ndarray) -> np.ndarray:
        """Apply 2D DCT to an 8x8 block."""
        # Precompute DCT matrix
        dct_matrix = np.zeros((8, 8), dtype=np.float64)
        for i in range(8):
            for j in range(8):
                if i == 0:
                    dct_matrix[i, j] = 1.0 / np.sqrt(8)
                else:
                    dct_matrix[i, j] = np.sqrt(2.0 / 8) * np.cos((2 * j + 1) * i * np.pi / 16)
        
        return dct_matrix @ block @ dct_matrix.T
    
    def _zigzag_scan(self, block: np.ndarray, num_coef: int) -> np.ndarray:
        """Extract coefficients in zigzag order."""
        zigzag = [
            (0,0), (0,1), (1,0), (2,0), (1,1), (0,2), (0,3), (1,2),
            (2,1), (3,0), (4,0), (3,1), (2,2), (1,3), (0,4), (0,5),
            (1,4), (2,3), (3,2), (4,1), (5,0), (6,0), (5,1), (4,2),
            (3,3), (2,4), (1,5), (0,6), (0,7), (1,6), (2,5), (3,4),
            (4,3), (5,2), (6,1), (7,0), (7,1), (6,2), (5,3), (4,4),
            (3,5), (2,6), (1,7), (2,7), (3,6), (4,5), (5,4), (6,3),
            (7,2), (7,3), (6,4), (5,5), (4,6), (3,7), (4,7), (5,6),
            (6,5), (7,4), (7,5), (6,6), (5,7), (6,7), (7,6), (7,7)
        ]
        
        coeffs = np.zeros(num_coef, dtype=np.float64)
        for i in range(min(num_coef, 64)):
            y, x = zigzag[i]
            coeffs[i] = block[y, x]
        
        return coeffs
    
    def get_feature_vector(self) -> np.ndarray:
        """Return the luminance layout coefficients."""
        if self._coeffs is None:
            return np.zeros(self.num_coeffs, dtype=np.float64)
        return self._coeffs
    
    @property
    def name(self) -> str:
        return "luminance_layout"
    
    def get_distance(self, other: 'LuminanceLayout') -> float:
        """Use L2 distance for comparison."""
        return dist_l2(self._coeffs, other._coeffs)
