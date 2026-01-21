"""
FCTH (Fuzzy Color and Texture Histogram) feature extractor.

FCTH uses fuzzy color quantization combined with texture information from
Haar wavelet decomposition to create a 192-dimensional descriptor.

Reference:
Chatzichristofis, S.A., Boutalis, Y.S.: FCTH: Fuzzy Color and Texture Histogram - 
A Low Level Feature for Accurate Image Retrieval. WIAMIS 2008
"""

import numpy as np
from imagefeatures.base import GlobalFeature, register_feature
from imagefeatures.utils.metrics import tanimoto


@register_feature("fcth")
class FCTH(GlobalFeature):
    """
    Fuzzy Color and Texture Histogram.
    
    Combines fuzzy 10-bin color quantization with 8 texture types from
    Haar wavelet analysis, producing a 192-dimensional descriptor (8 textures x 24 colors).
    
    Texture types are derived from the relationship between wavelet subbands.
    
    Example:
        >>> from imagefeatures.features import FCTH
        >>> from imagefeatures.utils import load_image
        >>> 
        >>> fcth = FCTH()
        >>> fcth.extract(load_image('image.jpg'))
        >>> features = fcth.get_feature_vector()  # 192-dim vector
    """
    
    def __init__(self):
        self._histogram = None
    
    def extract(self, image: np.ndarray) -> None:
        """Extract FCTH features from image."""
        h, w = image.shape[:2]
        
        # Convert to grayscale for texture analysis
        gray = (0.299 * image[:, :, 0] + 
                0.587 * image[:, :, 1] + 
                0.114 * image[:, :, 2]).astype(np.float64)
        
        # Determine block size
        min_dim = min(h, w)
        if min_dim >= 80:
            step = max(2, min_dim // 40)
        else:
            step = 2
        
        # Initialize 192-bin histogram (8 textures x 24 colors)
        histogram = np.zeros(192, dtype=np.float64)
        
        # Process image in blocks
        for y in range(0, h - step * 2, step):
            for x in range(0, w - step * 2, step):
                # Get color block
                block_rgb = image[y:y+step, x:x+step]
                
                # Get texture block (2x2 of step blocks for Haar)
                tex_block = gray[y:y+step*2, x:x+step*2]
                
                # Compute mean color and HSV
                mean_r = np.mean(block_rgb[:, :, 0])
                mean_g = np.mean(block_rgb[:, :, 1])
                mean_b = np.mean(block_rgb[:, :, 2])
                
                color_bin = self._get_color_bin(mean_r, mean_g, mean_b)
                
                # Compute texture type using simplified Haar analysis
                texture_type = self._get_texture_type(tex_block)
                
                # Accumulate
                histogram[texture_type * 24 + color_bin] += 1
        
        # Normalize
        total = np.sum(histogram)
        if total > 0:
            histogram = histogram / total
        
        # Quantize
        self._histogram = np.clip(histogram * 255 * 5, 0, 15).astype(np.float64)
    
    def _get_color_bin(self, r: float, g: float, b: float) -> int:
        """Get fuzzy color bin (0-23) from RGB."""
        # Convert to HSV
        max_c = max(r, g, b)
        min_c = min(r, g, b)
        
        v = max_c / 255.0
        s = 0 if max_c == 0 else (max_c - min_c) / max_c
        
        if max_c == min_c:
            h = 0
        elif max_c == r:
            h = 60 * ((g - b) / (max_c - min_c)) % 360
        elif max_c == g:
            h = 60 * ((b - r) / (max_c - min_c)) + 120
        else:
            h = 60 * ((r - g) / (max_c - min_c)) + 240
        
        # Quantize to 24 bins based on HSV
        # 8 hue bins x 3 saturation/value combinations
        if s < 0.2 and v < 0.3:
            return 0  # Black
        elif s < 0.2 and v > 0.8:
            return 1  # White
        elif s < 0.2:
            return 2  # Gray
        
        hue_bin = int(h / 45) % 8  # 8 hue bins
        sv_bin = 0 if v < 0.5 else (1 if s > 0.5 else 2)
        
        return 3 + hue_bin * 21 // 8 + sv_bin
    
    def _get_texture_type(self, block: np.ndarray) -> int:
        """
        Get texture type (0-7) using simplified Haar wavelet analysis.
        
        The 8 texture types represent different combinations of:
        - Smooth vs textured
        - Horizontal vs vertical orientation
        - Coarse vs fine texture
        """
        h, w = block.shape
        half_h, half_w = h // 2, w // 2
        
        # Compute 2x2 Haar approximation
        ll = np.mean(block[:half_h, :half_w])  # Low-low (approximation)
        lh = np.mean(block[:half_h, half_w:])  # Low-high (vertical detail)
        hl = np.mean(block[half_h:, :half_w])  # High-low (horizontal detail)
        hh = np.mean(block[half_h:, half_w:])  # High-high (diagonal detail)
        
        # Compute energy in detail subbands
        e_lh = abs(ll - lh)  # Vertical energy
        e_hl = abs(ll - hl)  # Horizontal energy
        e_hh = abs(ll - hh)  # Diagonal energy
        
        # Thresholds
        t_low = 10
        t_high = 30
        
        # Classify texture type
        if e_lh < t_low and e_hl < t_low and e_hh < t_low:
            return 0  # Smooth
        elif e_lh > t_high and e_hl < t_low:
            return 1  # Vertical edges
        elif e_hl > t_high and e_lh < t_low:
            return 2  # Horizontal edges
        elif e_hh > t_high:
            return 3  # Diagonal texture
        elif e_lh > t_low and e_hl > t_low:
            return 4  # Complex texture
        elif e_lh > e_hl:
            return 5  # Slightly vertical
        elif e_hl > e_lh:
            return 6  # Slightly horizontal
        else:
            return 7  # Isotropic texture
    
    def get_feature_vector(self) -> np.ndarray:
        """Return the FCTH histogram."""
        if self._histogram is None:
            return np.zeros(192, dtype=np.float64)
        return self._histogram
    
    @property
    def name(self) -> str:
        return "fcth"
    
    def get_distance(self, other: 'FCTH') -> float:
        """Use Tanimoto coefficient for FCTH comparison."""
        return tanimoto(self._histogram, other._histogram) * 100
