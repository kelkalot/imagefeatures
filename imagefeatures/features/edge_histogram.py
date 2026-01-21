"""
Edge Histogram Descriptor (MPEG-7).

Computes a histogram of edge directions in image sub-blocks.
"""

import numpy as np
from imagefeatures.base import GlobalFeature, register_feature
from imagefeatures.utils.metrics import dist_l1


@register_feature("edge_histogram")
class EdgeHistogram(GlobalFeature):
    """
    MPEG-7 Edge Histogram Descriptor.
    
    Divides the image into 4x4 sub-images and computes edge direction 
    histograms for each. Returns an 80-dimensional feature vector
    (16 blocks x 5 edge types).
    
    Edge types:
    - Vertical
    - Horizontal  
    - 45° diagonal
    - 135° diagonal
    - Non-directional (isotropic)
    
    Example:
        >>> from imagefeatures.features import EdgeHistogram
        >>> from imagefeatures.utils import load_image
        >>> 
        >>> eh = EdgeHistogram()
        >>> eh.extract(load_image('image.jpg'))
        >>> features = eh.get_feature_vector()  # 80-dim vector
    """
    
    # Edge detection filters (2x2 blocks)
    FILTERS = {
        'vertical': np.array([[1, -1], [1, -1]], dtype=np.float64),
        'horizontal': np.array([[1, 1], [-1, -1]], dtype=np.float64),
        'diag_45': np.array([[np.sqrt(2), 0], [0, -np.sqrt(2)]], dtype=np.float64),
        'diag_135': np.array([[0, np.sqrt(2)], [-np.sqrt(2), 0]], dtype=np.float64),
        'isotropic': np.array([[2, -2], [-2, 2]], dtype=np.float64),
    }
    
    # Quantization bins (MPEG-7 standard)
    QUANT_BINS = np.array([0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50])
    
    def __init__(self):
        self._histogram = None
    
    def extract(self, image: np.ndarray) -> None:
        """Extract edge histogram from image."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = (0.299 * image[:, :, 0] + 
                    0.587 * image[:, :, 1] + 
                    0.114 * image[:, :, 2]).astype(np.float64)
        else:
            gray = image.astype(np.float64)
        
        h, w = gray.shape
        
        # Initialize 80-bin histogram (16 blocks x 5 edge types)
        self._histogram = np.zeros(80, dtype=np.float64)
        
        # Divide into 4x4 sub-images
        block_h = h // 4
        block_w = w // 4
        
        for block_y in range(4):
            for block_x in range(4):
                # Extract sub-image
                y_start = block_y * block_h
                y_end = y_start + block_h
                x_start = block_x * block_w
                x_end = x_start + block_w
                
                sub_img = gray[y_start:y_end, x_start:x_end]
                
                # Compute edge histogram for this block
                block_hist = self._compute_block_histogram(sub_img)
                
                # Store in global histogram
                block_idx = block_y * 4 + block_x
                self._histogram[block_idx * 5:(block_idx + 1) * 5] = block_hist
        
        # Quantize histogram values
        self._quantize()
    
    def _compute_block_histogram(self, block: np.ndarray) -> np.ndarray:
        """Compute edge histogram for a single block."""
        h, w = block.shape
        hist = np.zeros(5, dtype=np.float64)
        
        # Process in 2x2 sub-blocks
        count = 0
        for y in range(0, h - 1, 2):
            for x in range(0, w - 1, 2):
                patch = block[y:y+2, x:x+2]
                
                # Compute filter responses
                responses = []
                for filter_mat in self.FILTERS.values():
                    response = np.sum(patch * filter_mat)
                    responses.append(abs(response))
                
                # Find strongest edge type
                max_response = max(responses)
                if max_response > 11:  # Threshold
                    max_idx = responses.index(max_response)
                    hist[max_idx] += 1
                    count += 1
        
        # Normalize
        if count > 0:
            hist /= count
        
        return hist
    
    def _quantize(self) -> None:
        """Quantize histogram values to 0-7 range."""
        for i in range(len(self._histogram)):
            val = self._histogram[i]
            # Find bin index
            bin_idx = np.searchsorted(self.QUANT_BINS, val)
            self._histogram[i] = min(bin_idx, 7)
    
    def get_feature_vector(self) -> np.ndarray:
        """Return the edge histogram."""
        if self._histogram is None:
            return np.zeros(80, dtype=np.float64)
        return self._histogram
    
    @property
    def name(self) -> str:
        return "edge_histogram"
    
    def get_distance(self, other: 'EdgeHistogram') -> float:
        """Use L1 distance for edge histogram comparison."""
        return dist_l1(self._histogram, other._histogram)
