"""
Base classes for image feature extraction.

Provides the abstract GlobalFeature class that all feature extractors inherit from,
plus a registry system for easy feature discovery and extensibility.
"""

from abc import ABC, abstractmethod
from typing import Dict, Type, Optional
import numpy as np


# Global feature registry
_FEATURE_REGISTRY: Dict[str, Type['GlobalFeature']] = {}


def register_feature(name: str):
    """
    Decorator to register a feature class in the global registry.
    
    Usage:
        @register_feature("color_histogram")
        class ColorHistogram(GlobalFeature):
            ...
    """
    def decorator(cls: Type['GlobalFeature']) -> Type['GlobalFeature']:
        _FEATURE_REGISTRY[name.lower()] = cls
        return cls
    return decorator


def get_feature(name: str) -> Optional[Type['GlobalFeature']]:
    """Get a registered feature class by name."""
    return _FEATURE_REGISTRY.get(name.lower())


def list_features() -> Dict[str, Type['GlobalFeature']]:
    """Return all registered features."""
    return dict(_FEATURE_REGISTRY)


class GlobalFeature(ABC):
    """
    Abstract base class for all global image feature extractors.
    
    A global feature extracts a fixed-size feature vector from an entire image.
    Subclasses must implement:
        - extract(image): Compute the feature from an image
        - get_feature_vector(): Return the computed feature as a numpy array
        - name: Property returning the feature's name
    
    Example:
        class MyFeature(GlobalFeature):
            def __init__(self):
                self._vector = None
            
            def extract(self, image: np.ndarray) -> None:
                # Compute feature from RGB image (H, W, 3)
                self._vector = np.zeros(64)  # placeholder
            
            def get_feature_vector(self) -> np.ndarray:
                return self._vector
            
            @property
            def name(self) -> str:
                return "my_feature"
    """
    
    @abstractmethod
    def extract(self, image: np.ndarray) -> None:
        """
        Extract the feature from an image.
        
        Args:
            image: RGB image as numpy array with shape (H, W, 3), dtype uint8
        """
        pass
    
    @abstractmethod
    def get_feature_vector(self) -> np.ndarray:
        """
        Return the extracted feature vector.
        
        Returns:
            1D numpy array containing the feature values
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this feature."""
        pass
    
    @property
    def dim(self) -> int:
        """Return the dimensionality of the feature vector."""
        vec = self.get_feature_vector()
        return len(vec) if vec is not None else 0
    
    def get_distance(self, other: 'GlobalFeature') -> float:
        """
        Compute distance to another feature of the same type.
        
        Default implementation uses L2 (Euclidean) distance.
        Subclasses can override for feature-specific distance metrics.
        
        Args:
            other: Another GlobalFeature instance of the same type
            
        Returns:
            Distance value (lower = more similar)
        """
        v1 = self.get_feature_vector()
        v2 = other.get_feature_vector()
        return float(np.sqrt(np.sum((v1 - v2) ** 2)))
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dim={self.dim})"
