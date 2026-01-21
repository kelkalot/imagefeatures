"""
Feature module - contains all feature extractors.
"""

# Import all feature classes to register them
from imagefeatures.features.color_histogram import ColorHistogram
from imagefeatures.features.lbp import LocalBinaryPatterns
from imagefeatures.features.gabor import Gabor
from imagefeatures.features.tamura import Tamura
from imagefeatures.features.edge_histogram import EdgeHistogram
from imagefeatures.features.color_layout import ColorLayout
from imagefeatures.features.cedd import CEDD
from imagefeatures.features.fcth import FCTH
from imagefeatures.features.jcd import JCD
from imagefeatures.features.phog import PHOG
from imagefeatures.features.auto_color_correlogram import AutoColorCorrelogram
from imagefeatures.features.opponent_histogram import OpponentHistogram
from imagefeatures.features.luminance_layout import LuminanceLayout
from imagefeatures.features.rotation_invariant_lbp import RotationInvariantLBP
from imagefeatures.features.centrist import Centrist
from imagefeatures.features.fuzzy_color_histogram import FuzzyColorHistogram
from imagefeatures.features.color_moments import ColorMoments
from imagefeatures.features.hu_moments import HuMoments
from imagefeatures.features.hog import HOG
from imagefeatures.features.haralick import Haralick
from imagefeatures.features.dominant_colors import DominantColors
from imagefeatures.features.scalable_color import ScalableColor

__all__ = [
    "ColorHistogram",
    "LocalBinaryPatterns",
    "Gabor",
    "Tamura",
    "EdgeHistogram",
    "ColorLayout",
    "CEDD",
    "FCTH",
    "JCD",
    "PHOG",
    "AutoColorCorrelogram",
    "OpponentHistogram",
    "LuminanceLayout",
    "RotationInvariantLBP",
    "Centrist",
    "FuzzyColorHistogram",
    "ColorMoments",
    "HuMoments",
    "HOG",
    "Haralick",
    "DominantColors",
    "ScalableColor",
]
