"""General utility functions used across the pipeline."""

import re
import logging
from typing import List, Optional
import numpy as np
from scipy.spatial.distance import cosine as cosine_distance

from .alias_types import FeatureVector # Use relative import if utils.py is in the same package

logger = logging.getLogger(__name__)

def sorted_alphanumeric(data: List[str]) -> List[str]:
    """Sorts a list of strings alphanumerically."""
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(data, key=alphanum_key)

def calculate_cosine_similarity(feat1: Optional[FeatureVector], feat2: Optional[FeatureVector]) -> float:
    """Calculates cosine similarity between two feature vectors, handling potential issues."""
    if feat1 is None or feat2 is None:
        return 0.0
    feat1 = feat1.flatten()
    feat2 = feat2.flatten()

    # Basic validity checks
    if feat1.shape != feat2.shape or feat1.size == 0:
        # logger.debug(f"Shape mismatch or zero size: {feat1.shape} vs {feat2.shape}")
        return 0.0
    if not np.any(feat1) or not np.any(feat2): # Check if either vector is all zeros
        # logger.debug("One or both features are all zeros.")
        return 0.0
    if not np.isfinite(feat1).all() or not np.isfinite(feat2).all():
        # logger.warning(f"Non-finite values detected in features.")
        return 0.0

    try:
        # Ensure float64 for precision in distance calculation
        distance = cosine_distance(feat1.astype(np.float64), feat2.astype(np.float64))
        # Clamp distance to [0, 2] range expected by cosine distance
        similarity = 1.0 - np.clip(float(distance), 0.0, 2.0)
        # Final clip for safety, although 1 - clipped_distance should be in [0, 1]
        return float(np.clip(similarity, 0.0, 1.0))
    except ValueError as ve:
        # logger.error(f"Cosine distance ValueError: {ve}. Feat1 norm: {np.linalg.norm(feat1)}, Feat2 norm: {np.linalg.norm(feat2)}")
        return 0.0
    except Exception as e:
        logger.error(f"Unexpected error during cosine distance calculation: {e}", exc_info=False)
        return 0.0

def normalize_embedding(embedding: FeatureVector) -> FeatureVector:
    """Normalizes a feature vector (L2 normalization)."""
    norm = np.linalg.norm(embedding)
    if norm < 1e-6: # Avoid division by zero or near-zero
        return embedding # Return original if norm is too small
    return embedding / norm