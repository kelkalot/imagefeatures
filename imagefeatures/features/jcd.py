"""
JCD (Joint Composite Descriptor) feature extractor.

JCD combines CEDD and FCTH descriptors into a single 168-dimensional descriptor.

Reference:
Zagoris K., Chatzichristofis S.A., Papamarkos N., Boutalis Y.S.: Automatic Image Annotation 
and Retrieval Using the Joint Composite Descriptor. ISDA 2010
"""

import numpy as np
from imagefeatures.base import GlobalFeature, register_feature
from imagefeatures.utils.metrics import tanimoto


@register_feature("jcd")
class JCD(GlobalFeature):
    """
    Joint Composite Descriptor.
    
    Combines CEDD and FCTH features into a unified 168-dimensional descriptor.
    The combination preserves complementary information from both descriptors.
    
    Example:
        >>> from imagefeatures.features import JCD
        >>> from imagefeatures.utils import load_image
        >>> 
        >>> jcd = JCD()
        >>> jcd.extract(load_image('image.jpg'))
        >>> features = jcd.get_feature_vector()  # 168-dim vector
    """
    
    def __init__(self):
        self._histogram = None
        self._cedd = None
        self._fcth = None
    
    def extract(self, image: np.ndarray) -> None:
        """Extract JCD features by combining CEDD and FCTH."""
        # Import here to avoid circular imports
        from imagefeatures.features.cedd import CEDD
        from imagefeatures.features.fcth import FCTH
        
        # Extract CEDD (144-dim or 60-dim compact)
        cedd = CEDD(compact=False)
        cedd.extract(image)
        cedd_hist = cedd.get_feature_vector()
        
        # Extract FCTH (192-dim)
        fcth = FCTH()
        fcth.extract(image)
        fcth_hist = fcth.get_feature_vector()
        
        # Combine into JCD (168-dim)
        self._histogram = self._join_histograms(cedd_hist, fcth_hist)
    
    def _join_histograms(self, cedd: np.ndarray, fcth: np.ndarray) -> np.ndarray:
        """
        Join CEDD and FCTH histograms into JCD.
        
        The joining follows the original LIRE implementation logic:
        - Creates a 168-dimensional descriptor
        - Combines corresponding texture+color bins from both descriptors
        """
        jcd = np.zeros(168, dtype=np.float64)
        
        # FCTH has 8 textures x 24 colors = 192
        # CEDD has 6 edges x 24 colors = 144
        # JCD has 7 x 24 = 168
        
        # Create temporary tables from FCTH (combine pairs of texture bins)
        temp1 = np.zeros(24)  # fcth[0:24] + fcth[96:120]
        temp2 = np.zeros(24)  # fcth[24:48] + fcth[120:144]
        temp3 = np.zeros(24)  # fcth[48:72] + fcth[144:168]
        temp4 = np.zeros(24)  # fcth[72:96] + fcth[168:192]
        
        for i in range(24):
            if i < len(fcth) - 96:
                temp1[i] = fcth[i] + fcth[96 + i] if 96 + i < len(fcth) else fcth[i]
            if i < len(fcth) - 120:
                temp2[i] = fcth[24 + i] + fcth[120 + i] if 120 + i < len(fcth) else fcth[24 + i]
            if i < len(fcth) - 144:
                temp3[i] = fcth[48 + i] + fcth[144 + i] if 144 + i < len(fcth) else fcth[48 + i]
            if i < len(fcth) - 168:
                temp4[i] = fcth[72 + i] + fcth[168 + i] if 168 + i < len(fcth) else fcth[72 + i]
        
        # Combine with CEDD
        for i in range(24):
            # Average/combine corresponding bins
            if i < len(cedd):
                jcd[i] = (temp1[i] + cedd[i]) / 2
            if 48 + i < len(cedd):
                jcd[24 + i] = (temp2[i] + cedd[48 + i]) / 2
            if 96 + i < len(cedd):
                jcd[48 + i] = cedd[96 + i]  # Pure CEDD
            if 72 + i < len(cedd):
                jcd[72 + i] = (temp3[i] + cedd[72 + i]) / 2
            if 120 + i < len(cedd):
                jcd[96 + i] = cedd[120 + i]  # Pure CEDD
            jcd[120 + i] = temp4[i]  # Pure FCTH
            if 24 + i < len(cedd):
                jcd[144 + i] = cedd[24 + i]  # Pure CEDD
        
        return jcd
    
    def get_feature_vector(self) -> np.ndarray:
        """Return the JCD histogram."""
        if self._histogram is None:
            return np.zeros(168, dtype=np.float64)
        return self._histogram
    
    @property
    def name(self) -> str:
        return "jcd"
    
    def get_distance(self, other: 'JCD') -> float:
        """Use Tanimoto coefficient for JCD comparison."""
        h1 = self._histogram
        h2 = other._histogram
        
        # Normalize
        s1 = np.sum(h1)
        s2 = np.sum(h2)
        
        if s1 == 0 and s2 == 0:
            return 0.0
        if s1 == 0 or s2 == 0:
            return 100.0
        
        h1_norm = h1 / s1
        h2_norm = h2 / s2
        
        # Tanimoto coefficient
        dot = np.sum(h1_norm * h2_norm)
        norm1 = np.sum(h1_norm * h1_norm)
        norm2 = np.sum(h2_norm * h2_norm)
        
        return 100.0 - 100.0 * (dot / (norm1 + norm2 - dot))
