"""
HOG (Histogram of Oriented Gradients) feature extractor.

Classic descriptor for object detection and recognition.
"""

import numpy as np
from imagefeatures.base import GlobalFeature, register_feature
from imagefeatures.utils.metrics import dist_l2


@register_feature("hog")
class HOG(GlobalFeature):
    """
    Histogram of Oriented Gradients.
    
    Divides image into cells and computes histogram of gradient orientations
    in each cell. Normalizes over blocks of cells.
    
    Default configuration:
    - 8x8 pixels per cell
    - 2x2 cells per block
    - 9 orientation bins
    - Image resized to 128x128
    
    Results in (16-1)x(16-1)x2x2x9 = 8100 dimensions (with default params)
    Simplified version: 4x4 cells = 144 dimensions
    
    Example:
        >>> from imagefeatures.features import HOG
        >>> from imagefeatures.utils import load_image
        >>> 
        >>> hog = HOG()
        >>> hog.extract(load_image('image.jpg'))
        >>> features = hog.get_feature_vector()
    """
    
    def __init__(self, cells_per_side: int = 4, orientations: int = 9):
        """
        Args:
            cells_per_side: Number of cells per image side (4 = 4x4 grid)
            orientations: Number of orientation bins (typically 9)
        """
        self.cells_per_side = cells_per_side
        self.orientations = orientations
        self._histogram = None
    
    def extract(self, image: np.ndarray) -> None:
        """Extract HOG features from image."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = (0.299 * image[:, :, 0] + 
                    0.587 * image[:, :, 1] + 
                    0.114 * image[:, :, 2]).astype(np.float64)
        else:
            gray = image.astype(np.float64)
        
        h, w = gray.shape
        
        # Resize to standard size for consistent feature dimensions
        target_size = self.cells_per_side * 8  # 8 pixels per cell
        if h != target_size or w != target_size:
            gray = self._resize(gray, target_size, target_size)
        
        h, w = gray.shape
        
        # Compute gradients
        gx = np.zeros_like(gray)
        gy = np.zeros_like(gray)
        
        gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
        gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
        
        # Compute magnitude and orientation
        magnitude = np.sqrt(gx**2 + gy**2)
        orientation = np.arctan2(gy, gx)  # Range: [-pi, pi]
        orientation = (orientation + np.pi) * self.orientations / (2 * np.pi)  # Map to [0, orientations)
        
        # Compute cell histograms
        cell_size = h // self.cells_per_side
        n_cells = self.cells_per_side
        
        cell_hists = np.zeros((n_cells, n_cells, self.orientations), dtype=np.float64)
        
        for i in range(n_cells):
            for j in range(n_cells):
                cell_mag = magnitude[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
                cell_ori = orientation[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
                
                # Build histogram with magnitude weighting
                for oi in range(self.orientations):
                    mask = ((cell_ori >= oi) & (cell_ori < oi + 1)) | \
                           ((oi == self.orientations - 1) & (cell_ori >= oi))
                    cell_hists[i, j, oi] = np.sum(cell_mag[mask])
        
        # Block normalization (L2-norm)
        # For simplicity, normalize each cell independently
        for i in range(n_cells):
            for j in range(n_cells):
                norm = np.linalg.norm(cell_hists[i, j])
                if norm > 0:
                    cell_hists[i, j] /= norm
        
        self._histogram = cell_hists.flatten()
    
    def _resize(self, image: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
        """Simple bilinear interpolation resize."""
        h, w = image.shape
        result = np.zeros((new_h, new_w), dtype=np.float64)
        
        for i in range(new_h):
            for j in range(new_w):
                src_y = i * h / new_h
                src_x = j * w / new_w
                
                y0, x0 = int(src_y), int(src_x)
                y1, x1 = min(y0 + 1, h - 1), min(x0 + 1, w - 1)
                
                dy, dx = src_y - y0, src_x - x0
                
                result[i, j] = (1-dy) * (1-dx) * image[y0, x0] + \
                               (1-dy) * dx * image[y0, x1] + \
                               dy * (1-dx) * image[y1, x0] + \
                               dy * dx * image[y1, x1]
        
        return result
    
    def get_feature_vector(self) -> np.ndarray:
        """Return the HOG histogram."""
        if self._histogram is None:
            return np.zeros(self.cells_per_side**2 * self.orientations, dtype=np.float64)
        return self._histogram
    
    @property
    def name(self) -> str:
        return "hog"
    
    def get_distance(self, other: 'HOG') -> float:
        """Use L2 distance for comparison."""
        return dist_l2(self._histogram, other._histogram)
