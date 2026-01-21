"""
Distance metrics for comparing feature vectors.
"""

import numpy as np


def dist_l1(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    L1 (Manhattan) distance between two vectors.
    
    Args:
        v1, v2: Feature vectors as numpy arrays
        
    Returns:
        L1 distance
    """
    return float(np.sum(np.abs(v1 - v2)))


def dist_l2(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    L2 (Euclidean) distance between two vectors.
    
    Args:
        v1, v2: Feature vectors as numpy arrays
        
    Returns:
        L2 distance
    """
    return float(np.sqrt(np.sum((v1 - v2) ** 2)))


def tanimoto(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Tanimoto distance (1 - Tanimoto coefficient).
    
    Also known as Jaccard distance for binary vectors.
    
    Args:
        v1, v2: Feature vectors as numpy arrays
        
    Returns:
        Tanimoto distance in range [0, 1]
    """
    dot = np.dot(v1, v2)
    norm1 = np.dot(v1, v1)
    norm2 = np.dot(v2, v2)
    
    denom = norm1 + norm2 - dot
    if denom == 0:
        return 0.0
    
    return 1.0 - (dot / denom)


def jsd(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Jensen-Shannon Divergence between two distributions.
    
    Args:
        v1, v2: Histogram vectors (will be normalized to sum to 1)
        
    Returns:
        JSD value (lower = more similar)
    """
    # Normalize to probability distributions
    s1 = np.sum(v1)
    s2 = np.sum(v2)
    
    if s1 == 0 or s2 == 0:
        return 1.0  # Maximum distance if one is empty
    
    p = v1.astype(np.float64) / s1
    q = v2.astype(np.float64) / s2
    
    # Mean distribution
    m = (p + q) / 2
    
    # KL divergence helper with log2
    def kl(a, b):
        # Avoid log(0) by adding small epsilon where needed
        mask = (a > 0) & (b > 0)
        result = np.zeros_like(a)
        result[mask] = a[mask] * np.log2(a[mask] / b[mask])
        return np.sum(result)
    
    return float((kl(p, m) + kl(q, m)) / 2)


def cosine_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Cosine distance (1 - cosine similarity).
    
    Args:
        v1, v2: Feature vectors as numpy arrays
        
    Returns:
        Cosine distance in range [0, 2]
    """
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 1.0
    
    similarity = np.dot(v1, v2) / (norm1 * norm2)
    return 1.0 - similarity
