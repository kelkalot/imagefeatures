"""
CEDD (Color and Edge Directivity Descriptor) feature extractor.

CEDD combines fuzzy color quantization with texture (edge) information
to create a compact 144-dimensional descriptor.

Reference:
Chatzichristofis, S.A., Boutalis, Y.S.: CEDD: Color and Edge Directivity Descriptor. 
A Compact Descriptor for Image Indexing and Retrieval. ICVS 2008
"""

import numpy as np
from imagefeatures.base import GlobalFeature, register_feature
from imagefeatures.utils.metrics import tanimoto


def rgb_to_hsv_cedd(r: int, g: int, b: int) -> tuple:
    """Convert RGB to HSV using CEDD's specific method."""
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    
    # Value
    v = max_c
    
    # Saturation
    s = 0 if max_c == 0 else int(255 - 255 * (min_c / max_c))
    
    # Hue
    if max_c == min_c:
        h = 0
    elif max_c == r and g >= b:
        h = int(60 * (g - b) / (max_c - min_c))
    elif max_c == r and g < b:
        h = int(359 + 60 * (g - b) / (max_c - min_c))
    elif max_c == g:
        h = int(119 + 60 * (b - r) / (max_c - min_c))
    else:  # max_c == b
        h = int(239 + 60 * (r - g) / (max_c - min_c))
    
    return h, s, v


class Fuzzy10Bin:
    """Fuzzy 10-bin color quantizer for CEDD."""
    
    # Membership function boundaries for triangular fuzzy sets
    HUE_TRIANGLES = [
        (0, 0, 5, 10), (5, 10, 35, 50), (35, 50, 70, 85), (70, 85, 150, 165),
        (150, 165, 195, 205), (195, 205, 265, 280), (265, 280, 315, 330), (315, 330, 360, 360)
    ]
    SAT_TRIANGLES = [(0, 0, 10, 75), (10, 75, 255, 255)]
    VAL_TRIANGLES = [(0, 0, 10, 75), (10, 75, 180, 220), (180, 220, 255, 255)]
    
    # Fuzzy rules: (hue_idx, sat_idx, val_idx) -> output_bin
    RULES = [
        (0, 0, 0, 2), (0, 1, 0, 2), (0, 0, 2, 0), (0, 0, 1, 1),
        (1, 0, 0, 2), (1, 1, 0, 2), (1, 0, 2, 0), (1, 0, 1, 1),
        (2, 0, 0, 2), (2, 1, 0, 2), (2, 0, 2, 0), (2, 0, 1, 1),
        (3, 0, 0, 2), (3, 1, 0, 2), (3, 0, 2, 0), (3, 0, 1, 1),
        (4, 0, 0, 2), (4, 1, 0, 2), (4, 0, 2, 0), (4, 0, 1, 1),
        (5, 0, 0, 2), (5, 1, 0, 2), (5, 0, 2, 0), (5, 0, 1, 1),
        (6, 0, 0, 2), (6, 1, 0, 2), (6, 0, 2, 0), (6, 0, 1, 1),
        (7, 0, 0, 2), (7, 1, 0, 2), (7, 0, 2, 0), (7, 0, 1, 1),
        (0, 1, 1, 3), (0, 1, 2, 3), (1, 1, 1, 4), (1, 1, 2, 4),
        (2, 1, 1, 5), (2, 1, 2, 5), (3, 1, 1, 6), (3, 1, 2, 6),
        (4, 1, 1, 7), (4, 1, 2, 7), (5, 1, 1, 8), (5, 1, 2, 8),
        (6, 1, 1, 9), (6, 1, 2, 9), (7, 1, 1, 3), (7, 1, 2, 3),
    ]
    
    def __init__(self):
        pass
    
    def _triangular_membership(self, value: float, triangles: list) -> np.ndarray:
        """Compute membership values for triangular fuzzy sets."""
        result = np.zeros(len(triangles))
        for i, (a, b, c, d) in enumerate(triangles):
            if b <= value <= c:
                result[i] = 1.0
            elif a <= value < b and b != a:
                result[i] = (value - a) / (b - a)
            elif c < value <= d and c != d:
                result[i] = (value - c) / (c - d) + 1
        return result
    
    def apply(self, h: float, s: float, v: float) -> np.ndarray:
        """Apply fuzzy quantization and return 10-bin histogram contribution."""
        hue_act = self._triangular_membership(h, self.HUE_TRIANGLES)
        sat_act = self._triangular_membership(s, self.SAT_TRIANGLES)
        val_act = self._triangular_membership(v, self.VAL_TRIANGLES)
        
        result = np.zeros(10)
        for h_idx, s_idx, v_idx, out_bin in self.RULES:
            if hue_act[h_idx] > 0 and sat_act[s_idx] > 0 and val_act[v_idx] > 0:
                activation = min(hue_act[h_idx], sat_act[s_idx], val_act[v_idx])
                result[out_bin] += activation
        
        return result


@register_feature("cedd")
class CEDD(GlobalFeature):
    """
    Color and Edge Directivity Descriptor.
    
    Combines fuzzy color quantization (10 bins) with edge directivity (6 types)
    to produce a 144-dimensional feature vector (actually uses compact 60-dim).
    
    The 6 edge types are: no edge, horizontal, vertical, 45°, 135°, non-directional
    Combined with 24 fuzzy color bins = 144 dimensions (or 10 colors x 6 edges = 60 compact)
    
    Example:
        >>> from imagefeatures.features import CEDD
        >>> from imagefeatures.utils import load_image
        >>> 
        >>> cedd = CEDD()
        >>> cedd.extract(load_image('image.jpg'))
        >>> features = cedd.get_feature_vector()  # 144-dim vector
    """
    
    # Edge detection thresholds
    T0 = 14.0  # No edge threshold
    T1 = 0.68  # Non-directional threshold
    T2 = 0.98  # Directional threshold
    T3 = 0.98  # Diagonal threshold
    
    def __init__(self, compact: bool = False):
        """
        Args:
            compact: If True, use 60-dim output, otherwise 144-dim
        """
        self.compact = compact
        self._histogram = None
        self._fuzzy = Fuzzy10Bin()
    
    def extract(self, image: np.ndarray) -> None:
        """Extract CEDD features from image."""
        h, w = image.shape[:2]
        
        # Determine block size based on image size
        min_dim = min(h, w)
        if min_dim >= 80:
            num_blocks = 1600
        elif min_dim >= 40:
            num_blocks = 400
        else:
            num_blocks = 100
        
        blocks_per_side = int(np.sqrt(num_blocks))
        step_y = (h // blocks_per_side) // 2 * 2  # Ensure even
        step_x = (w // blocks_per_side) // 2 * 2
        
        if step_x < 2:
            step_x = 2
        if step_y < 2:
            step_y = 2
        
        # Convert to grayscale for edge detection
        gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        
        # Initialize histogram (6 edge types x 24 color bins = 144)
        # Or compact: 6 edge types x 10 color bins = 60
        n_colors = 10 if self.compact else 24
        histogram = np.zeros(6 * n_colors, dtype=np.float64)
        
        max_y = step_y * blocks_per_side
        max_x = step_x * blocks_per_side
        
        for y in range(0, min(max_y, h - step_y), step_y):
            for x in range(0, min(max_x, w - step_x), step_x):
                # Get block
                block_rgb = image[y:y+step_y, x:x+step_x]
                block_gray = gray[y:y+step_y, x:x+step_x]
                
                # Compute edge type
                edge_type = self._detect_edge(block_gray, step_x, step_y)
                
                # Compute mean color
                mean_r = int(np.mean(block_rgb[:, :, 0]))
                mean_g = int(np.mean(block_rgb[:, :, 1]))
                mean_b = int(np.mean(block_rgb[:, :, 2]))
                
                # Convert to HSV
                hsv = rgb_to_hsv_cedd(mean_r, mean_g, mean_b)
                
                # Get fuzzy color bins
                color_bins = self._fuzzy.apply(hsv[0], hsv[1], hsv[2])
                
                # Accumulate in histogram
                if self.compact:
                    for j in range(10):
                        if color_bins[j] > 0:
                            histogram[n_colors * edge_type + j] += color_bins[j]
                else:
                    # Use 24-bin extension (simplified - just repeat 10 bins)
                    for j in range(10):
                        if color_bins[j] > 0:
                            # Map to 24 bins (simplified mapping)
                            bin_24 = (j * 24) // 10
                            histogram[n_colors * edge_type + bin_24] += color_bins[j]
        
        # Normalize
        total = np.sum(histogram)
        if total > 0:
            histogram = histogram / total
        
        # Quantize to 0-15 range (4 bits per bin)
        self._histogram = np.clip(histogram * 15 * 10, 0, 15).astype(np.float64)
    
    def _detect_edge(self, block: np.ndarray, step_x: int, step_y: int) -> int:
        """
        Detect edge type in a block.
        
        Returns:
            0: No edge
            1: Non-directional
            2: Horizontal
            3: Vertical
            4: 45° diagonal
            5: 135° diagonal
        """
        half_x = step_x // 2
        half_y = step_y // 2
        
        # Compute quadrant averages
        q1 = np.mean(block[:half_y, :half_x])  # Top-left
        q2 = np.mean(block[:half_y, half_x:])  # Top-right
        q3 = np.mean(block[half_y:, :half_x])  # Bottom-left
        q4 = np.mean(block[half_y:, half_x:])  # Bottom-right
        
        # Compute mask responses
        m1 = abs(2*q1 - 2*q2 - 2*q3 + 2*q4)  # Non-directional
        m2 = abs(q1 + q2 - q3 - q4)  # Horizontal
        m3 = abs(q1 - q2 + q3 - q4)  # Vertical
        m4 = abs(np.sqrt(2)*q1 - np.sqrt(2)*q4)  # 45° diagonal
        m5 = abs(np.sqrt(2)*q2 - np.sqrt(2)*q3)  # 135° diagonal
        
        max_response = max(m1, m2, m3, m4, m5)
        
        if max_response < self.T0:
            return 0  # No edge
        
        # Normalize
        m1 /= max_response
        m2 /= max_response
        m3 /= max_response
        m4 /= max_response
        m5 /= max_response
        
        # Find dominant edge
        edges = []
        if m1 > self.T1:
            edges.append((m1, 1))
        if m2 > self.T2:
            edges.append((m2, 2))
        if m3 > self.T2:
            edges.append((m3, 3))
        if m4 > self.T3:
            edges.append((m4, 4))
        if m5 > self.T3:
            edges.append((m5, 5))
        
        if not edges:
            return 0
        
        return max(edges, key=lambda x: x[0])[1]
    
    def get_feature_vector(self) -> np.ndarray:
        """Return the CEDD histogram."""
        if self._histogram is None:
            n_colors = 10 if self.compact else 24
            return np.zeros(6 * n_colors, dtype=np.float64)
        return self._histogram
    
    @property
    def name(self) -> str:
        return "cedd"
    
    def get_distance(self, other: 'CEDD') -> float:
        """Use Tanimoto coefficient for CEDD comparison."""
        return tanimoto(self._histogram, other._histogram) * 100
