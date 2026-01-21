"""
Color space conversion utilities.

All conversions work on numpy arrays.
"""

import numpy as np


def rgb_to_hsv(r: int, g: int, b: int) -> tuple:
    """
    Convert RGB to HSV color space.
    
    Args:
        r, g, b: RGB values in range [0, 255]
        
    Returns:
        Tuple (h, s, v) where:
        - h: hue in range [0, 360)
        - s: saturation in range [0, 100]
        - v: value in range [0, 100]
    """
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    diff = max_c - min_c
    
    # Hue calculation
    if diff == 0:
        h = 0
    elif max_c == r:
        h = 60 * (((g - b) / diff) % 6)
    elif max_c == g:
        h = 60 * (((b - r) / diff) + 2)
    else:
        h = 60 * (((r - g) / diff) + 4)
    
    # Saturation
    s = 0 if max_c == 0 else (diff / max_c) * 100
    
    # Value
    v = max_c * 100
    
    return (h, s, v)


def rgb_to_hsv_array(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to HSV.
    
    Args:
        image: RGB image (H, W, 3), dtype uint8
        
    Returns:
        HSV image (H, W, 3), where H is [0-360], S and V are [0-100]
    """
    img = image.astype(np.float32) / 255.0
    
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    diff = max_c - min_c
    
    # Hue
    h = np.zeros_like(max_c)
    
    mask = diff != 0
    mask_r = mask & (max_c == r)
    mask_g = mask & (max_c == g)
    mask_b = mask & (max_c == b)
    
    h[mask_r] = 60 * (((g[mask_r] - b[mask_r]) / diff[mask_r]) % 6)
    h[mask_g] = 60 * (((b[mask_g] - r[mask_g]) / diff[mask_g]) + 2)
    h[mask_b] = 60 * (((r[mask_b] - g[mask_b]) / diff[mask_b]) + 4)
    
    # Saturation
    s = np.zeros_like(max_c)
    mask_s = max_c != 0
    s[mask_s] = (diff[mask_s] / max_c[mask_s]) * 100
    
    # Value
    v = max_c * 100
    
    return np.stack([h, s, v], axis=-1)


def rgb_to_yuv(r: int, g: int, b: int) -> tuple:
    """
    Convert RGB to YUV color space.
    
    Args:
        r, g, b: RGB values in range [0, 255]
        
    Returns:
        Tuple (y, u, v)
    """
    y = int(0.299 * r + 0.587 * g + 0.114 * b)
    u = int((b - y) * 0.492)
    v = int((r - y) * 0.877)
    return (y, u, v)


def rgb_to_hmmd(r: int, g: int, b: int) -> tuple:
    """
    Convert RGB to HMMD color space (Hue-Max-Min-Diff).
    
    Used in MPEG-7 color descriptors.
    
    Args:
        r, g, b: RGB values in range [0, 255]
        
    Returns:
        Tuple (hue, max, min, diff, sum)
    """
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    diff = max_c - min_c
    sum_c = (max_c + min_c) / 2.0
    
    # Hue calculation
    if diff == 0:
        hue = 0
    elif max_c == r and (g - b) > 0:
        hue = 60 * (g - b) / diff
    elif max_c == r and (g - b) <= 0:
        hue = 60 * (g - b) / diff + 360
    elif max_c == g:
        hue = 60 * (2 + (b - r) / diff)
    else:  # max_c == b
        hue = 60 * (4 + (r - g) / diff)
    
    return (int(hue), int(max_c), int(min_c), int(diff / 2), int(sum_c))
