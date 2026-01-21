# imagefeatures 🖼️

A minimal Python package for extracting **classic image features** with only NumPy + Pillow.

Perfect for image retrieval, classification, and similarity search.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kelkalot/imagefeatures/blob/main/examples/demo.ipynb) **Basic Demo** |
[![Similarity](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kelkalot/imagefeatures/blob/main/examples/demo_similarity.ipynb) **Similarity Search** |
[![Classification](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kelkalot/imagefeatures/blob/main/examples/demo_classification.ipynb) **Classification**

## ✨ Features

**22 feature extractors** totaling **3,058 dimensions**:

### Color Features (741 dims)
| Feature | Dims | Description |
|---------|------|-------------|
| `ColorHistogram` | 64 | RGB/HSV/Luminance histogram |
| `ColorMoments` | 9 | Mean, std, skewness per channel |
| `OpponentHistogram` | 512 | Opponent color space histogram |
| `FuzzyColorHistogram` | 72 | Fuzzy HSV quantization |
| `DominantColors` | 20 | K-means extracted dominant colors |
| `ScalableColor` | 64 | MPEG-7 Haar-based color descriptor |

### Texture Features (620 dims)
| Feature | Dims | Description |
|---------|------|-------------|
| `LocalBinaryPatterns` | 256 | Classic LBP histogram |
| `RotationInvariantLBP` | 36 | Rotation-invariant LBP |
| `Gabor` | 48 | Multi-scale Gabor wavelets |
| `Tamura` | 18 | Coarseness, contrast, directionality |
| `Haralick` | 6 | GLCM texture features |
| `Centrist` | 256 | Census transform histogram |

### Shape Features (861 dims)
| Feature | Dims | Description |
|---------|------|-------------|
| `EdgeHistogram` | 80 | MPEG-7 edge directions |
| `PHOG` | 630 | Pyramid histogram of oriented gradients |
| `HOG` | 144 | Histogram of oriented gradients |
| `HuMoments` | 7 | Shape-invariant Hu moments |

### Layout Features (76 dims)
| Feature | Dims | Description |
|---------|------|-------------|
| `ColorLayout` | 12 | MPEG-7 DCT color layout |
| `LuminanceLayout` | 64 | DCT luminance descriptor |

### Combined Features (760 dims)
| Feature | Dims | Description |
|---------|------|-------------|
| `CEDD` | 144 | Color + edge directivity |
| `FCTH` | 192 | Fuzzy color + texture |
| `JCD` | 168 | Joint CEDD + FCTH |
| `AutoColorCorrelogram` | 256 | Spatial color correlation |

## 📦 Installation

```bash
# From GitHub
pip install git+https://github.com/kelkalot/imagefeatures.git

# Or clone and install locally
git clone https://github.com/kelkalot/imagefeatures.git
cd imagefeatures
pip install -e .
```

**Dependencies:** `numpy>=1.20`, `pillow>=8.0` (no OpenCV required!)

## 🚀 Quick Start

### Extract from a Single Image

```python
from imagefeatures import FeatureExtractor
from imagefeatures.features import ColorHistogram, LocalBinaryPatterns, CEDD

# Create extractor with specific features
extractor = FeatureExtractor([ColorHistogram(), LocalBinaryPatterns(), CEDD()])

# Extract features
features = extractor.extract("image.jpg")

for name, vector in features.items():
    print(f"{name}: {len(vector)} dimensions")
```

### Extract from a Folder

```python
from imagefeatures import FeatureExtractor

# Use all 22 features
extractor = FeatureExtractor()

# Extract and save to CSV
result = extractor.extract_folder("./images/", output="features.csv")
print(f"Extracted {result['features'].shape[1]} dimensions from {len(result['filenames'])} images")

# Or save as sklearn-compatible pickle
extractor.extract_folder("./images/", output="features.pkl")
```

### Use with scikit-learn

```python
import pickle
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

# Load extracted features
X, filenames = pickle.load(open("features.pkl", "rb"))

# Cluster images
kmeans = KMeans(n_clusters=10).fit(X)

# Find similar images
nn = NearestNeighbors(n_neighbors=5).fit(X)
distances, indices = nn.kneighbors(X[0:1])
```

## 🔧 Creating Custom Features

The package uses a decorator-based registration system. Here's how to create your own feature:

### Basic Example

```python
from imagefeatures import GlobalFeature, register_feature
import numpy as np

@register_feature("mean_rgb")
class MeanRGB(GlobalFeature):
    """Simple feature: mean RGB values."""
    
    def __init__(self):
        self._vector = None
    
    def extract(self, image: np.ndarray) -> None:
        """
        Extract feature from image.
        
        Args:
            image: RGB numpy array with shape (H, W, 3), dtype uint8
        """
        # Compute mean of each channel
        self._vector = np.mean(image, axis=(0, 1))
    
    def get_feature_vector(self) -> np.ndarray:
        """Return the computed feature vector."""
        if self._vector is None:
            return np.zeros(3)
        return self._vector
    
    @property
    def name(self) -> str:
        """Feature name (used in output columns)."""
        return "mean_rgb"
```

### Advanced Example with Distance Metric

```python
from imagefeatures import GlobalFeature, register_feature
from imagefeatures.utils.metrics import jsd  # Jensen-Shannon divergence
import numpy as np

@register_feature("color_distribution")
class ColorDistribution(GlobalFeature):
    """Normalized color distribution with custom distance."""
    
    def __init__(self, bins: int = 32):
        self.bins = bins
        self._histogram = None
    
    def extract(self, image: np.ndarray) -> None:
        h, w = image.shape[:2]
        self._histogram = np.zeros(self.bins * 3)
        
        for c in range(3):
            channel = image[:, :, c].flatten()
            hist, _ = np.histogram(channel, bins=self.bins, range=(0, 256))
            self._histogram[c * self.bins:(c + 1) * self.bins] = hist
        
        # Normalize to probability distribution
        self._histogram = self._histogram / (h * w)
    
    def get_feature_vector(self) -> np.ndarray:
        if self._histogram is None:
            return np.zeros(self.bins * 3)
        return self._histogram
    
    @property
    def name(self) -> str:
        return f"color_dist_{self.bins}"
    
    def get_distance(self, other: 'ColorDistribution') -> float:
        """Use Jensen-Shannon divergence for probability distributions."""
        return jsd(self._histogram, other._histogram)
```

### Using Your Custom Feature

```python
from imagefeatures import FeatureExtractor
from my_features import MeanRGB, ColorDistribution

# Your features are automatically registered
extractor = FeatureExtractor([
    MeanRGB(),
    ColorDistribution(bins=16),
])

result = extractor.extract_folder("./images/")
```

### Available Distance Metrics

The `imagefeatures.utils.metrics` module provides:

| Function | Description | Best for |
|----------|-------------|----------|
| `dist_l1` | Manhattan distance | Histograms |
| `dist_l2` | Euclidean distance | General purpose |
| `cosine_distance` | 1 - cosine similarity | Sparse vectors |
| `jsd` | Jensen-Shannon divergence | Probability distributions |
| `tanimoto` | Tanimoto coefficient | Binary/sparse features |

## 📁 Output Formats

| Format | Extension | Description |
|--------|-----------|-------------|
| CSV | `.csv` | One row per image, columns are feature values |
| Pickle | `.pkl` | Tuple of `(X, filenames)` for sklearn |

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Test specific feature
python -c "from imagefeatures.features import CEDD; print(CEDD().dim)"
```

## 📚 References

This package is inspired by the [LIRE](https://github.com/dermotte/lire) Java library. Key papers:
- **CEDD/FCTH**: Chatzichristofis & Boutalis, ICVS 2008
- **PHOG**: Bosch, Zisserman & Munoz, CVIR 2007
- **Color Correlogram**: Huang et al., CVPR 1997
- **LBP**: Ojala, Pietikäinen & Mäenpää, PAMI 2002

## 📄 License & Contributors

MIT

Michael A. Riegler 
Mathias Lux
