Extract image features, compare image similarity, or analyze images using the imagefeatures library.

This skill provides access to 22 classical image feature extractors (3,058 total dimensions) covering color, texture, shape, and layout — all powered by NumPy and Pillow with zero deep learning dependencies.

## Usage

The user will provide one of:
- An image path (or paths) to extract features from
- Two image paths to compare for similarity
- A folder path to batch-process
- A question about which features to use for a task

Arguments: $ARGUMENTS

## Instructions

1. **Determine the task** from the user's arguments:
   - If a single image path → extract features and summarize
   - If two image paths → compare similarity using multiple metrics
   - If a folder → batch extract and report statistics
   - If "list" or "features" → show available features
   - If a question → recommend features and demonstrate

2. **Run feature extraction** using Python:

```python
import numpy as np
from imagefeatures import FeatureExtractor
from imagefeatures.base import list_features
from imagefeatures.utils import load_image
```

3. **For single image analysis**, extract all features and present a readable summary including:
   - Dominant colors (RGB values and percentages)
   - Texture characteristics (coarseness, contrast, directionality from Tamura)
   - Shape complexity (edge types, Hu moment magnitudes)
   - Layout description (from ColorLayout DCT coefficients)

4. **For image comparison**, compute distances using each feature's native metric:

```python
from imagefeatures.utils.metrics import dist_l1, dist_l2, cosine_distance, jsd, tanimoto

# Extract features from both images, then:
for name, feat in features.items():
    distance = feat.get_distance(vec1, vec2)
```

Present results as a ranked similarity table.

5. **For batch processing**, use:

```python
extractor = FeatureExtractor()
result = extractor.extract_folder(folder_path)
```

6. **Always show concrete numbers** — feature dimensions, distances, dominant colors, etc. Don't just say "features extracted"; show what was found.

7. **For selecting specific features**, the user can name them:
   - Color: color_histogram, color_moments, opponent_histogram, fuzzy_color_histogram, dominant_colors, scalable_color
   - Texture: lbp, rotation_invariant_lbp, gabor, tamura, haralick, centrist
   - Shape: edge_histogram, phog, hog, hu_moments
   - Layout: color_layout, luminance_layout
   - Combined: cedd, fcth, jcd, auto_color_correlogram

## Output Format

Present results clearly with tables or structured output. Include:
- Feature name and dimensionality
- Key values (not full vectors — summarize meaningfully)
- For comparisons: distance values and interpretation (lower = more similar)
