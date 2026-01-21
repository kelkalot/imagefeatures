"""
Utility functions for image loading and processing.
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union


def load_image(path: Union[str, Path]) -> np.ndarray:
    """
    Load an image file and return as RGB numpy array.
    
    Args:
        path: Path to image file
        
    Returns:
        RGB image as numpy array with shape (H, W, 3), dtype uint8
    """
    img = Image.open(path)
    # Convert to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return np.array(img)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to grayscale.
    
    Uses standard luminosity formula: 0.299*R + 0.587*G + 0.114*B
    
    Args:
        image: RGB image as numpy array (H, W, 3)
        
    Returns:
        Grayscale image as numpy array (H, W), dtype uint8
    """
    if len(image.shape) == 2:
        return image  # Already grayscale
    
    # Standard luminosity formula
    gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    return gray.astype(np.uint8)


def resize_image(image: np.ndarray, max_size: int = 256) -> np.ndarray:
    """
    Resize image so the larger dimension is at most max_size.
    
    Args:
        image: Input image as numpy array
        max_size: Maximum size for the larger dimension
        
    Returns:
        Resized image as numpy array
    """
    h, w = image.shape[:2]
    if max(h, w) <= max_size:
        return image
    
    scale = max_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Use PIL for high-quality resizing
    if len(image.shape) == 3:
        img = Image.fromarray(image)
    else:
        img = Image.fromarray(image, mode='L')
    
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return np.array(img)


def get_image_files(folder: Union[str, Path]) -> list:
    """
    Get all image files in a folder.
    
    Args:
        folder: Path to folder
        
    Returns:
        List of Path objects for image files
    """
    folder = Path(folder)
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    
    files = []
    for ext in extensions:
        files.extend(folder.glob(f'*{ext}'))
        files.extend(folder.glob(f'*{ext.upper()}'))
    
    return sorted(files)
