"""
imagefeatures - Classic image feature extraction for Python

A minimal, easy-to-use package for extracting classic image features
like color histograms, texture descriptors, and edge features.
"""

from imagefeatures.base import GlobalFeature, register_feature, get_feature
from imagefeatures.extractor import FeatureExtractor

__version__ = "0.1.0"
__all__ = [
    "GlobalFeature",
    "register_feature", 
    "get_feature",
    "FeatureExtractor",
]
