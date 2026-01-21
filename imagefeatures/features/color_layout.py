"""
Color Layout Descriptor (MPEG-7).

Uses DCT coefficients of downsampled image to capture color distribution.
"""

import numpy as np
from imagefeatures.base import GlobalFeature, register_feature
from imagefeatures.utils.metrics import dist_l2


@register_feature("color_layout")
class ColorLayout(GlobalFeature):
    """
    MPEG-7 Color Layout Descriptor.
    
    Divides the image into an 8x8 grid, computes representative colors,
    and applies DCT to capture the spatial color distribution. Returns
    DCT coefficients from Y, Cb, Cr channels.
    
    Default: 6 Y coefficients + 3 Cb + 3 Cr = 12 dimensions.
    
    Example:
        >>> from imagefeatures.features import ColorLayout
        >>> from imagefeatures.utils import load_image
        >>> 
        >>> cl = ColorLayout()
        >>> cl.extract(load_image('image.jpg'))
        >>> features = cl.get_feature_vector()
    """
    
    def __init__(self, num_y_coef: int = 6, num_c_coef: int = 3):
        """
        Args:
            num_y_coef: Number of Y (luminance) DCT coefficients (1-64)
            num_c_coef: Number of Cb, Cr DCT coefficients each (1-64)
        """
        self.num_y_coef = min(64, max(1, num_y_coef))
        self.num_c_coef = min(64, max(1, num_c_coef))
        self._y_coeffs = None
        self._cb_coeffs = None
        self._cr_coeffs = None
    
    def extract(self, image: np.ndarray) -> None:
        """Extract color layout from image."""
        h, w = image.shape[:2]
        
        # Create 8x8 representative color array
        y_grid = np.zeros((8, 8), dtype=np.float64)
        cb_grid = np.zeros((8, 8), dtype=np.float64)
        cr_grid = np.zeros((8, 8), dtype=np.float64)
        
        block_h = h // 8
        block_w = w // 8
        
        for by in range(8):
            for bx in range(8):
                # Get block
                y_start = by * block_h
                y_end = y_start + block_h if by < 7 else h
                x_start = bx * block_w
                x_end = x_start + block_w if bx < 7 else w
                
                block = image[y_start:y_end, x_start:x_end]
                
                # Average color
                avg_r = np.mean(block[:, :, 0])
                avg_g = np.mean(block[:, :, 1])
                avg_b = np.mean(block[:, :, 2])
                
                # Convert to YCbCr
                y, cb, cr = self._rgb_to_ycbcr(avg_r, avg_g, avg_b)
                
                y_grid[by, bx] = y
                cb_grid[by, bx] = cb
                cr_grid[by, bx] = cr
        
        # Apply DCT to each channel
        y_dct = self._dct_8x8(y_grid)
        cb_dct = self._dct_8x8(cb_grid)
        cr_dct = self._dct_8x8(cr_grid)
        
        # Extract coefficients in zigzag order
        self._y_coeffs = self._zigzag_scan(y_dct, self.num_y_coef)
        self._cb_coeffs = self._zigzag_scan(cb_dct, self.num_c_coef)
        self._cr_coeffs = self._zigzag_scan(cr_dct, self.num_c_coef)
        
        # Quantize to 0-255 range
        self._y_coeffs = np.clip(self._y_coeffs, 0, 255)
        self._cb_coeffs = np.clip(self._cb_coeffs + 128, 0, 255)
        self._cr_coeffs = np.clip(self._cr_coeffs + 128, 0, 255)
    
    def _rgb_to_ycbcr(self, r: float, g: float, b: float) -> tuple:
        """Convert RGB to YCbCr."""
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = -0.169 * r - 0.331 * g + 0.500 * b
        cr = 0.500 * r - 0.419 * g - 0.081 * b
        return y, cb, cr
    
    def _dct_8x8(self, block: np.ndarray) -> np.ndarray:
        """Apply 2D DCT to an 8x8 block."""
        result = np.zeros((8, 8), dtype=np.float64)
        
        # Precompute DCT matrix
        dct_matrix = np.zeros((8, 8), dtype=np.float64)
        for i in range(8):
            for j in range(8):
                if i == 0:
                    dct_matrix[i, j] = 1.0 / np.sqrt(8)
                else:
                    dct_matrix[i, j] = np.sqrt(2.0 / 8) * np.cos((2 * j + 1) * i * np.pi / 16)
        
        # Apply DCT: result = dct_matrix @ block @ dct_matrix.T
        result = dct_matrix @ block @ dct_matrix.T
        
        return result
    
    def _zigzag_scan(self, block: np.ndarray, num_coef: int) -> np.ndarray:
        """Extract coefficients in zigzag order."""
        # Zigzag indices for 8x8 block
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
        """Return the color layout descriptor."""
        if self._y_coeffs is None:
            return np.zeros(self.num_y_coef + 2 * self.num_c_coef, dtype=np.float64)
        
        return np.concatenate([self._y_coeffs, self._cb_coeffs, self._cr_coeffs])
    
    @property
    def name(self) -> str:
        return "color_layout"
    
    def get_distance(self, other: 'ColorLayout') -> float:
        """
        Weighted distance for color layout.
        Y coefficients are weighted more heavily.
        """
        if self._y_coeffs is None or other._y_coeffs is None:
            return float('inf')
        
        # Weighted Euclidean distance
        # DC coefficients have higher weight
        y_weights = np.array([2.0] + [1.0] * (self.num_y_coef - 1))
        c_weights = np.array([2.0] + [1.0] * (self.num_c_coef - 1))
        
        y_dist = np.sum(y_weights * (self._y_coeffs - other._y_coeffs) ** 2)
        cb_dist = np.sum(c_weights * (self._cb_coeffs - other._cb_coeffs) ** 2)
        cr_dist = np.sum(c_weights * (self._cr_coeffs - other._cr_coeffs) ** 2)
        
        return np.sqrt(y_dist + cb_dist + cr_dist)
