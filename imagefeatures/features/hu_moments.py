"""
Hu Moments feature extractor.

Seven invariant moments for shape description.
"""

import numpy as np
from imagefeatures.base import GlobalFeature, register_feature
from imagefeatures.utils.metrics import dist_l2


@register_feature("hu_moments")
class HuMoments(GlobalFeature):
    """
    Hu Moments Descriptor.
    
    Computes 7 Hu moments that are invariant to:
    - Translation
    - Scale
    - Rotation
    
    Based on normalized central moments of the image.
    
    Example:
        >>> from imagefeatures.features import HuMoments
        >>> from imagefeatures.utils import load_image
        >>> 
        >>> hu = HuMoments()
        >>> hu.extract(load_image('image.jpg'))
        >>> features = hu.get_feature_vector()  # 7-dim vector
    """
    
    def __init__(self):
        self._moments = None
    
    def extract(self, image: np.ndarray) -> None:
        """Extract Hu moments from image."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = (0.299 * image[:, :, 0] + 
                    0.587 * image[:, :, 1] + 
                    0.114 * image[:, :, 2]).astype(np.float64)
        else:
            gray = image.astype(np.float64)
        
        # Compute raw moments
        m = self._raw_moments(gray)
        
        # Compute central moments
        x_bar = m['m10'] / m['m00'] if m['m00'] != 0 else 0
        y_bar = m['m01'] / m['m00'] if m['m00'] != 0 else 0
        
        mu = self._central_moments(gray, x_bar, y_bar)
        
        # Compute normalized central moments
        nu = self._normalized_moments(mu, m['m00'])
        
        # Compute Hu moments
        self._moments = self._hu_moments(nu)
    
    def _raw_moments(self, img: np.ndarray) -> dict:
        """Compute raw moments m_pq."""
        h, w = img.shape
        y, x = np.mgrid[0:h, 0:w]
        
        m = {}
        m['m00'] = np.sum(img)
        m['m10'] = np.sum(x * img)
        m['m01'] = np.sum(y * img)
        m['m20'] = np.sum(x**2 * img)
        m['m02'] = np.sum(y**2 * img)
        m['m11'] = np.sum(x * y * img)
        m['m30'] = np.sum(x**3 * img)
        m['m03'] = np.sum(y**3 * img)
        m['m21'] = np.sum(x**2 * y * img)
        m['m12'] = np.sum(x * y**2 * img)
        
        return m
    
    def _central_moments(self, img: np.ndarray, x_bar: float, y_bar: float) -> dict:
        """Compute central moments mu_pq."""
        h, w = img.shape
        y, x = np.mgrid[0:h, 0:w]
        x = x - x_bar
        y = y - y_bar
        
        mu = {}
        mu['mu00'] = np.sum(img)
        mu['mu20'] = np.sum(x**2 * img)
        mu['mu02'] = np.sum(y**2 * img)
        mu['mu11'] = np.sum(x * y * img)
        mu['mu30'] = np.sum(x**3 * img)
        mu['mu03'] = np.sum(y**3 * img)
        mu['mu21'] = np.sum(x**2 * y * img)
        mu['mu12'] = np.sum(x * y**2 * img)
        
        return mu
    
    def _normalized_moments(self, mu: dict, m00: float) -> dict:
        """Compute scale-normalized central moments."""
        nu = {}
        if m00 == 0:
            return {k: 0 for k in ['nu20', 'nu02', 'nu11', 'nu30', 'nu03', 'nu21', 'nu12']}
        
        nu['nu20'] = mu['mu20'] / (m00 ** 2)
        nu['nu02'] = mu['mu02'] / (m00 ** 2)
        nu['nu11'] = mu['mu11'] / (m00 ** 2)
        nu['nu30'] = mu['mu30'] / (m00 ** 2.5)
        nu['nu03'] = mu['mu03'] / (m00 ** 2.5)
        nu['nu21'] = mu['mu21'] / (m00 ** 2.5)
        nu['nu12'] = mu['mu12'] / (m00 ** 2.5)
        
        return nu
    
    def _hu_moments(self, nu: dict) -> np.ndarray:
        """Compute 7 Hu invariant moments."""
        hu = np.zeros(7, dtype=np.float64)
        
        # Hu moment 1
        hu[0] = nu['nu20'] + nu['nu02']
        
        # Hu moment 2
        hu[1] = (nu['nu20'] - nu['nu02'])**2 + 4*nu['nu11']**2
        
        # Hu moment 3
        hu[2] = (nu['nu30'] - 3*nu['nu12'])**2 + (3*nu['nu21'] - nu['nu03'])**2
        
        # Hu moment 4
        hu[3] = (nu['nu30'] + nu['nu12'])**2 + (nu['nu21'] + nu['nu03'])**2
        
        # Hu moment 5
        hu[4] = ((nu['nu30'] - 3*nu['nu12']) * (nu['nu30'] + nu['nu12']) * 
                 ((nu['nu30'] + nu['nu12'])**2 - 3*(nu['nu21'] + nu['nu03'])**2) +
                 (3*nu['nu21'] - nu['nu03']) * (nu['nu21'] + nu['nu03']) *
                 (3*(nu['nu30'] + nu['nu12'])**2 - (nu['nu21'] + nu['nu03'])**2))
        
        # Hu moment 6
        hu[5] = ((nu['nu20'] - nu['nu02']) * 
                 ((nu['nu30'] + nu['nu12'])**2 - (nu['nu21'] + nu['nu03'])**2) +
                 4*nu['nu11'] * (nu['nu30'] + nu['nu12']) * (nu['nu21'] + nu['nu03']))
        
        # Hu moment 7 (skew invariant)
        hu[6] = ((3*nu['nu21'] - nu['nu03']) * (nu['nu30'] + nu['nu12']) *
                 ((nu['nu30'] + nu['nu12'])**2 - 3*(nu['nu21'] + nu['nu03'])**2) -
                 (nu['nu30'] - 3*nu['nu12']) * (nu['nu21'] + nu['nu03']) *
                 (3*(nu['nu30'] + nu['nu12'])**2 - (nu['nu21'] + nu['nu03'])**2))
        
        # Log transform for better numerical properties
        for i in range(7):
            if hu[i] != 0:
                hu[i] = -np.sign(hu[i]) * np.log10(np.abs(hu[i]) + 1e-10)
        
        return hu
    
    def get_feature_vector(self) -> np.ndarray:
        """Return the Hu moments."""
        if self._moments is None:
            return np.zeros(7, dtype=np.float64)
        return self._moments
    
    @property
    def name(self) -> str:
        return "hu_moments"
    
    def get_distance(self, other: 'HuMoments') -> float:
        """Use L2 distance for comparison."""
        return dist_l2(self._moments, other._moments)
