Analyze images, compare visual similarity, find duplicates, or group images by appearance.

This skill wraps the `imagefeatures` library (22 classical CV descriptors, NumPy + Pillow only) to answer **user-level questions** about images — not to dump raw feature vectors.

Arguments: $ARGUMENTS

## Task Detection

Determine what the user actually wants from their input:

| User intent | Example inputs | What to do |
|---|---|---|
| **"Are these similar?"** | Two image paths | Compare and give a clear verdict |
| **"Find duplicates"** | A folder path | Find near-duplicate pairs |
| **"Group these"** | A folder path + "cluster"/"group" | Cluster by visual similarity |
| **"Describe this image"** | Single image path | Summarize visual properties in plain language |
| **"Which images match X?"** | A folder + a query image | Rank folder images by similarity to query |

If the input doesn't match any of these, ask the user what they want to do with the image(s).

## Core Approach

**Never dump raw feature vectors or distance tables at the user.** Instead:
1. Run the analysis
2. Interpret the numbers
3. Give a clear answer in plain language
4. Show supporting numbers only when they add value

## How to Compare Two Images

Use a curated subset of features that are reliable and complementary. Do NOT use all 22 — that creates noise.

```python
from imagefeatures.utils import load_image
from imagefeatures.features import (
    CEDD, ColorLayout, ColorHistogram,
    EdgeHistogram, Tamura, AutoColorCorrelogram
)

img1 = load_image(path1)
img2 = load_image(path2)

# These 6 features cover color, texture, shape, and layout well
features = [CEDD(), ColorLayout(), ColorHistogram(),
            EdgeHistogram(), Tamura(), AutoColorCorrelogram()]

scores = {}
for feat in features:
    f1, f2 = type(feat)(), type(feat)()
    f1.extract(img1)
    f2.extract(img2)
    scores[feat.name] = f1.get_distance(f2)
```

### Interpreting Distances

Each feature uses its own metric and scale. Use these thresholds to interpret:

| Feature | Metric | Very Similar | Somewhat Similar | Different |
|---|---|---|---|---|
| **CEDD** | Tanimoto×100 | < 15 | 15–40 | > 40 |
| **ColorLayout** | Weighted L2 | < 10 | 10–25 | > 25 |
| **ColorHistogram** | JSD | < 0.05 | 0.05–0.15 | > 0.15 |
| **EdgeHistogram** | L1 | < 30 | 30–70 | > 70 |
| **Tamura** | L2 | < 0.5 | 0.5–1.2 | > 1.2 |
| **AutoColorCorrelogram** | L1 | < 50 | 50–120 | > 120 |

### Producing a Verdict

Count how many features say "very similar", "somewhat similar", and "different":
- **Majority "very similar"** → "These images look very similar (likely near-duplicates or minor variations)"
- **Majority "somewhat similar"** → "These images share some visual characteristics but are clearly different images"
- **Majority "different"** → "These images are visually different"
- **Mixed signals** → Explain what's shared and what differs: "Similar color palette but different structure" (if color features agree but edge/shape don't)

Always state the verdict first, then optionally show the breakdown.

## How to Find Duplicates in a Folder

```python
from imagefeatures import FeatureExtractor
from imagefeatures.features import CEDD, ColorLayout
from imagefeatures.utils import load_image, get_image_files
from imagefeatures.utils.metrics import cosine_distance
from pathlib import Path
import numpy as np

folder = Path(folder_path)
image_files = get_image_files(folder)

# Use CEDD — best single feature for general similarity
extractor = FeatureExtractor([CEDD()])
vectors = {}
for f in image_files:
    result = extractor.extract(f)
    vectors[f.name] = list(result.values())[0]

# Compare all pairs
duplicates = []
names = list(vectors.keys())
for i in range(len(names)):
    for j in range(i+1, len(names)):
        dist = cosine_distance(vectors[names[i]], vectors[names[j]])
        if dist < 0.05:  # Very similar threshold
            duplicates.append((names[i], names[j], dist))

duplicates.sort(key=lambda x: x[2])
```

Report as: "Found N pairs of near-duplicate images" with a list of pairs, or "No duplicates found."

## How to Describe a Single Image

Extract a few interpretable features and translate to plain language:

```python
from imagefeatures.features import DominantColors, Tamura, EdgeHistogram, HuMoments
from imagefeatures.utils import load_image

img = load_image(path)

dc = DominantColors(k=5)
dc.extract(img)
colors = dc.get_feature_vector()  # [R,G,B,%,R,G,B,%, ...]

tamura = Tamura()
tamura.extract(img)

eh = EdgeHistogram()
eh.extract(img)
```

Translate to language like:
- **Dominant colors**: "Mostly warm tones — 45% orange-brown (RGB 180,120,60), 25% cream (RGB 240,230,210)..."
- **Texture**: Use `tamura.coarseness`, `.contrast`, `.directionality`:
  - Coarseness > 0.6 → "coarse/grainy texture", < 0.3 → "smooth/fine texture"
  - Contrast > 30 → "high contrast", < 10 → "low contrast/flat"
  - Directionality > 0.3 → "strong directional patterns (stripes/lines)", < 0.15 → "no dominant direction"
- **Edges**: Summarize the 5 edge types (vertical, horizontal, 45°, 135°, non-directional) — "Dominated by vertical edges" or "Mix of edge orientations"

## How to Group/Cluster Images

```python
from imagefeatures import FeatureExtractor
from imagefeatures.features import CEDD
from imagefeatures.utils import get_image_files
from imagefeatures.utils.metrics import cosine_distance
from pathlib import Path
import numpy as np

folder = Path(folder_path)
image_files = get_image_files(folder)

extractor = FeatureExtractor([CEDD()])
vectors = {}
for f in image_files:
    result = extractor.extract(f)
    vectors[f.name] = list(result.values())[0]

# Simple agglomerative clustering using distance matrix
names = list(vectors.keys())
n = len(names)
dist_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(i+1, n):
        d = cosine_distance(vectors[names[i]], vectors[names[j]])
        dist_matrix[i][j] = d
        dist_matrix[j][i] = d

# Assign groups using threshold-based clustering
threshold = 0.15
visited = [False] * n
groups = []
for i in range(n):
    if visited[i]:
        continue
    group = [names[i]]
    visited[i] = True
    for j in range(i+1, n):
        if not visited[j] and dist_matrix[i][j] < threshold:
            group.append(names[j])
            visited[j] = True
    groups.append(group)
```

Report as: "Found N visual groups" with representative descriptions like "Group 1 (3 images): warm-toned outdoor scenes" etc.

## Output Rules

1. **Lead with the answer.** "These images are visually similar." not "I extracted 22 features and computed distances..."
2. **Use plain language.** "Similar colors but different layout" not "ColorHistogram JSD=0.03, ColorLayout L2=28.4"
3. **Show numbers only as evidence**, not as the primary output. Put them in a small supporting table if the user would find them useful.
4. **Name colors by name**, not RGB. Say "teal" not "RGB(0,128,128)". Include RGB in parentheses only if relevant.
5. **For errors** (file not found, not an image, etc.), say what went wrong clearly and suggest a fix.
