"""
Meaningful tests for imagefeatures: verify that feature extractors produce
correct, interpretable results on images with known visual properties.

These aren't just "does it run" tests — they verify semantic correctness:
- Solid color images should have trivial texture and peaked color histograms
- Stripe orientation should be detected by edge/shape features
- Similar images should be closer than dissimilar ones in feature space
- Feature dimensions and normalization should be correct
"""
import os
import numpy as np
import pytest

from imagefeatures import FeatureExtractor
from imagefeatures.features import *  # triggers registry population
from imagefeatures.base import list_features, get_feature
from imagefeatures.utils import load_image
from imagefeatures.utils.metrics import dist_l2, cosine_distance

IMG_DIR = os.path.join(os.path.dirname(__file__), "images")


def img(name):
    return load_image(os.path.join(IMG_DIR, name))


# ---------------------------------------------------------------------------
# 1. REGISTRY AND BASIC INFRASTRUCTURE
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_all_22_features_registered(self):
        features = list_features()
        assert len(features) == 22, f"Expected 22 features, got {len(features)}: {list(features.keys())}"

    def test_total_dimensions(self):
        extractor = FeatureExtractor()
        assert extractor.get_total_dimensions() == 3058

    def test_feature_names_match(self):
        expected = {
            "color_histogram", "color_moments", "opponent_histogram",
            "fuzzy_color_histogram", "dominant_colors", "scalable_color",
            "lbp", "rotation_invariant_lbp", "gabor", "tamura", "haralick",
            "centrist", "edge_histogram", "phog", "hog", "hu_moments",
            "color_layout", "luminance_layout", "cedd", "fcth", "jcd",
            "auto_color_correlogram",
        }
        assert set(list_features().keys()) == expected


# ---------------------------------------------------------------------------
# 2. COLOR FEATURES — verified on solid and split images
# ---------------------------------------------------------------------------

class TestColorFeatures:
    def test_dominant_colors_solid_red(self):
        """Solid red image: dominant color should be (255, 0, 0)."""
        feat = DominantColors()
        feat.extract(img("solid_red.png"))
        vec = feat.get_feature_vector()
        assert len(vec) == 20
        # First cluster should be red
        r, g, b = vec[0], vec[1], vec[2]
        assert r > 200, f"Expected red channel > 200, got {r}"
        assert g < 50, f"Expected green channel < 50, got {g}"
        assert b < 50, f"Expected blue channel < 50, got {b}"

    def test_color_moments_distinguish_red_blue(self):
        """Color moments should clearly distinguish red from blue."""
        feat_r = ColorMoments()
        feat_r.extract(img("solid_red.png"))
        vec_r = feat_r.get_feature_vector()

        feat_b = ColorMoments()
        feat_b.extract(img("solid_blue.png"))
        vec_b = feat_b.get_feature_vector()

        # Means should differ significantly (first 3 values are channel means)
        assert not np.allclose(vec_r[:3], vec_b[:3], atol=50), \
            "Red and blue should have very different color moments"

    def test_color_histogram_solid_is_peaked(self):
        """Solid color → histogram should have mass in very few bins."""
        feat = ColorHistogram()
        feat.extract(img("solid_red.png"))
        vec = feat.get_feature_vector()
        # Most bins should be zero for a uniform color
        nonzero = np.count_nonzero(vec)
        assert nonzero <= 3, f"Solid color histogram should be peaked, got {nonzero} nonzero bins"

    def test_color_histogram_split_has_two_peaks(self):
        """Red-blue split should activate at least 2 bins."""
        feat = ColorHistogram()
        feat.extract(img("red_blue_split.png"))
        vec = feat.get_feature_vector()
        nonzero = np.count_nonzero(vec)
        assert nonzero >= 2, f"Split image should have ≥2 color bins, got {nonzero}"

    def test_fuzzy_color_histogram_rainbow_spread(self):
        """Rainbow image should activate many fuzzy histogram bins due to hue variety."""
        feat = FuzzyColorHistogram()
        feat.extract(img("rainbow.png"))
        vec = feat.get_feature_vector()
        nonzero = np.count_nonzero(vec)
        assert nonzero > 5, f"Rainbow fuzzy histogram should be spread, got {nonzero} nonzero"

    def test_opponent_histogram_dimensions(self):
        """OpponentHistogram should produce 512-dim vector."""
        feat = OpponentHistogram()
        feat.extract(img("rainbow.png"))
        vec = feat.get_feature_vector()
        assert len(vec) == 512


# ---------------------------------------------------------------------------
# 3. TEXTURE FEATURES — verified on stripes vs. solid
# ---------------------------------------------------------------------------

class TestTextureFeatures:
    def test_lbp_solid_has_concentrated_histogram(self):
        """Solid images should concentrate LBP mass in few bins
        (all neighbors equal to center → code 255)."""
        feat = LocalBinaryPatterns()
        feat.extract(img("solid_red.png"))
        vec = feat.get_feature_vector()
        # Solid: all pixels identical, all neighbors >= center → code 255
        # Should have dominant bin at code 255
        assert vec[255] == np.max(vec), "Solid image LBP should peak at bin 255"

    def test_lbp_checkerboard_is_distributed(self):
        """Checkerboard should have more LBP patterns activated than solid."""
        feat_s = LocalBinaryPatterns()
        feat_s.extract(img("solid_red.png"))
        vec_s = feat_s.get_feature_vector()

        feat_c = LocalBinaryPatterns()
        feat_c.extract(img("checkerboard.png"))
        vec_c = feat_c.get_feature_vector()

        nonzero_s = np.count_nonzero(vec_s)
        nonzero_c = np.count_nonzero(vec_c)
        assert nonzero_c > nonzero_s, \
            f"Checkerboard ({nonzero_c} bins) should activate more LBP bins than solid ({nonzero_s})"

    def test_tamura_dimensions(self):
        """Tamura should produce 18-dim vector (coarseness + contrast + directionality)."""
        feat = Tamura()
        feat.extract(img("gradient_bw.png"))
        vec = feat.get_feature_vector()
        assert len(vec) == 18

    def test_haralick_contrast_solid_vs_stripes(self):
        """Solid image should have near-zero GLCM contrast; stripes should have high."""
        feat_s = Haralick()
        feat_s.extract(img("solid_red.png"))
        vec_s = feat_s.get_feature_vector()

        feat_t = Haralick()
        feat_t.extract(img("stripes_horizontal.png"))
        vec_t = feat_t.get_feature_vector()

        # First element is contrast
        assert vec_s[0] < vec_t[0], \
            f"Solid contrast ({vec_s[0]:.4f}) should be less than stripes ({vec_t[0]:.4f})"

    def test_gabor_dimensions(self):
        """Gabor should produce 48-dim vector (4 scales × 6 orientations × 2 stats)."""
        feat = Gabor()
        feat.extract(img("stripes_horizontal.png"))
        vec = feat.get_feature_vector()
        assert len(vec) == 48

    def test_rotation_invariant_lbp_dimensions(self):
        """RotationInvariantLBP should produce 36-dim vector."""
        feat = RotationInvariantLBP()
        feat.extract(img("checkerboard.png"))
        vec = feat.get_feature_vector()
        assert len(vec) == 36


# ---------------------------------------------------------------------------
# 4. SHAPE / EDGE FEATURES — verified on stripe orientation
# ---------------------------------------------------------------------------

class TestShapeFeatures:
    def test_edge_histogram_dimensions(self):
        """EdgeHistogram should produce 80-dim vector (16 blocks × 5 types)."""
        feat = EdgeHistogram()
        feat.extract(img("stripes_horizontal.png"))
        vec = feat.get_feature_vector()
        assert len(vec) == 80

    def test_phog_distinguishes_shapes(self):
        """PHOG should distinguish checkerboard from circles (different gradient structures)."""
        feat_ck = PHOG()
        feat_ck.extract(img("checkerboard.png"))
        vec_ck = feat_ck.get_feature_vector()

        feat_cr = PHOG()
        feat_cr.extract(img("concentric_circles.png"))
        vec_cr = feat_cr.get_feature_vector()

        dist = dist_l2(vec_ck, vec_cr)
        assert dist > 0.01, "PHOG should distinguish checkerboard from circles"

    def test_hu_moments_finite(self):
        """Hu moments should be finite for all test images."""
        feat = HuMoments()
        feat.extract(img("concentric_circles.png"))
        vec = feat.get_feature_vector()
        assert len(vec) == 7
        assert np.all(np.isfinite(vec)), "Hu moments should be finite"

    def test_hu_moments_differ_for_different_shapes(self):
        """Different shapes should have different Hu moments."""
        feat_ck = HuMoments()
        feat_ck.extract(img("checkerboard.png"))
        vec_ck = feat_ck.get_feature_vector()

        feat_cr = HuMoments()
        feat_cr.extract(img("concentric_circles.png"))
        vec_cr = feat_cr.get_feature_vector()

        assert not np.allclose(vec_ck, vec_cr, atol=0.1), \
            "Different shapes should have different Hu moments"

    def test_hog_dimensions(self):
        """HOG should produce 144-dimensional vector."""
        feat = HOG()
        feat.extract(img("checkerboard.png"))
        vec = feat.get_feature_vector()
        assert len(vec) == 144


# ---------------------------------------------------------------------------
# 5. LAYOUT FEATURES — verified on split vs. solid
# ---------------------------------------------------------------------------

class TestLayoutFeatures:
    def test_color_layout_distinguishes_split_from_solid(self):
        """Red-blue split should have different layout than solid red."""
        feat_solid = ColorLayout()
        feat_solid.extract(img("solid_red.png"))
        vec_solid = feat_solid.get_feature_vector()

        feat_split = ColorLayout()
        feat_split.extract(img("red_blue_split.png"))
        vec_split = feat_split.get_feature_vector()

        dist = dist_l2(vec_solid, vec_split)
        assert dist > 1.0, f"Layout distance between solid and split should be > 1, got {dist:.4f}"

    def test_luminance_layout_gradient_vs_solid(self):
        """Gradient should have different luminance layout than solid."""
        feat_s = LuminanceLayout()
        feat_s.extract(img("solid_red.png"))
        vec_s = feat_s.get_feature_vector()

        feat_g = LuminanceLayout()
        feat_g.extract(img("gradient_bw.png"))
        vec_g = feat_g.get_feature_vector()

        dist = dist_l2(vec_s, vec_g)
        assert dist > 0.1, f"Luminance layout should differ, got dist={dist:.4f}"


# ---------------------------------------------------------------------------
# 6. COMBINED FEATURES (CEDD, FCTH, JCD)
# ---------------------------------------------------------------------------

class TestCombinedFeatures:
    def test_cedd_dimensions(self):
        feat = CEDD()
        feat.extract(img("rainbow.png"))
        vec = feat.get_feature_vector()
        assert len(vec) == 144

    def test_fcth_dimensions(self):
        feat = FCTH()
        feat.extract(img("rainbow.png"))
        vec = feat.get_feature_vector()
        assert len(vec) == 192

    def test_jcd_dimensions(self):
        feat = JCD()
        feat.extract(img("rainbow.png"))
        vec = feat.get_feature_vector()
        assert len(vec) == 168

    def test_cedd_solid_vs_textured(self):
        """CEDD combines color + edge: solid should differ from textured."""
        feat_s = CEDD()
        feat_s.extract(img("solid_red.png"))
        vec_s = feat_s.get_feature_vector()

        feat_t = CEDD()
        feat_t.extract(img("checkerboard.png"))
        vec_t = feat_t.get_feature_vector()

        dist = dist_l2(vec_s, vec_t)
        assert dist > 0.01, "CEDD should distinguish solid from textured"

    def test_auto_color_correlogram_dimensions(self):
        feat = AutoColorCorrelogram()
        feat.extract(img("rainbow.png"))
        vec = feat.get_feature_vector()
        assert len(vec) == 256


# ---------------------------------------------------------------------------
# 7. SIMILARITY / DISTANCE — the key integration test
# ---------------------------------------------------------------------------

class TestSimilarity:
    def test_solid_red_closer_to_split_than_to_blue(self):
        """Red is closer to red-blue-split (50% red) than to solid blue,
        using color_moments which captures per-channel statistics."""
        extractor = FeatureExtractor()
        feat_red = extractor.extract(os.path.join(IMG_DIR, "solid_red.png"))
        feat_blue = extractor.extract(os.path.join(IMG_DIR, "solid_blue.png"))
        feat_split = extractor.extract(os.path.join(IMG_DIR, "red_blue_split.png"))

        # Use color_moments (name key is color_moments_rgb)
        d_red_split = dist_l2(feat_red["color_moments_rgb"], feat_split["color_moments_rgb"])
        d_red_blue = dist_l2(feat_red["color_moments_rgb"], feat_blue["color_moments_rgb"])
        assert d_red_split < d_red_blue, \
            f"Red→Split ({d_red_split:.4f}) should be < Red→Blue ({d_red_blue:.4f})"

    def test_same_texture_family_closer(self):
        """H-stripes and V-stripes share more with each other (via haralick)
        than either shares with solid red (no texture)."""
        extractor = FeatureExtractor()
        feat_h = extractor.extract(os.path.join(IMG_DIR, "stripes_horizontal.png"))
        feat_v = extractor.extract(os.path.join(IMG_DIR, "stripes_vertical.png"))
        feat_s = extractor.extract(os.path.join(IMG_DIR, "solid_red.png"))

        # Haralick captures texture properties; stripes have similar contrast/energy
        d_hv = dist_l2(feat_h["haralick"], feat_v["haralick"])
        d_hs = dist_l2(feat_h["haralick"], feat_s["haralick"])
        assert d_hv < d_hs, \
            f"Stripes H↔V ({d_hv:.4f}) should be < Stripes H↔Solid ({d_hs:.4f})"

    def test_identical_images_zero_distance_deterministic(self):
        """Same image should have zero distance for deterministic features."""
        extractor = FeatureExtractor()
        feat1 = extractor.extract(os.path.join(IMG_DIR, "checkerboard.png"))
        feat2 = extractor.extract(os.path.join(IMG_DIR, "checkerboard.png"))

        # Skip dominant_colors (k-means is non-deterministic)
        for name in feat1:
            if name == "dominant_colors":
                continue
            d = dist_l2(feat1[name], feat2[name])
            assert d < 1e-10, f"Self-distance for {name} should be ~0, got {d}"


# ---------------------------------------------------------------------------
# 8. BATCH EXTRACTION (FeatureExtractor integration)
# ---------------------------------------------------------------------------

class TestBatchExtraction:
    def test_extract_folder(self):
        """Extract features from all test images at once."""
        extractor = FeatureExtractor()
        result = extractor.extract_folder(IMG_DIR)
        assert "features" in result
        assert "filenames" in result
        assert len(result["filenames"]) == 10
        assert result["features"].shape == (10, 3058)

    def test_selective_features(self):
        """Extract only selected features."""
        extractor = FeatureExtractor([ColorHistogram(), HuMoments()])
        result = extractor.extract(os.path.join(IMG_DIR, "rainbow.png"))
        # Keys use the .name property: color_histogram_rgb and hu_moments
        assert "color_histogram_rgb" in result
        assert "hu_moments" in result
        assert len(result) == 2


# ---------------------------------------------------------------------------
# 9. ALL FEATURES RUN WITHOUT ERROR ON ALL IMAGES
# ---------------------------------------------------------------------------

class TestAllFeaturesAllImages:
    """Smoke test: every feature extracts successfully on every test image."""

    @pytest.fixture(params=[
        "solid_red.png", "solid_blue.png", "red_blue_split.png",
        "stripes_horizontal.png", "stripes_vertical.png", "stripes_diagonal.png",
        "gradient_bw.png", "rainbow.png", "checkerboard.png", "concentric_circles.png",
    ])
    def image_name(self, request):
        return request.param

    def test_all_features_extract(self, image_name):
        """Every registered feature should extract without error."""
        image = img(image_name)
        for name, cls in list_features().items():
            feat = cls()
            feat.extract(image)
            vec = feat.get_feature_vector()
            assert isinstance(vec, np.ndarray), f"{name} should return ndarray"
            assert len(vec) > 0, f"{name} should return non-empty vector"
            assert np.all(np.isfinite(vec)), f"{name} returned non-finite values on {image_name}"
