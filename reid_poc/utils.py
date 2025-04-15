"""General utility functions used across the pipeline."""

import re
import logging
from typing import List, Optional, Tuple, Set
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
    # --- Function remains the same as original ---
    if feat1 is None or feat2 is None: return 0.0
    feat1 = feat1.flatten()
    feat2 = feat2.flatten()

    if feat1.shape != feat2.shape or feat1.size == 0: return 0.0
    if not np.any(feat1) or not np.any(feat2): return 0.0
    if not np.isfinite(feat1).all() or not np.isfinite(feat2).all(): return 0.0

    try:
        # scipy's cosine distance is 1 - similarity
        distance = cosine_distance(feat1.astype(np.float64), feat2.astype(np.float64))
        # Clamp distance to handle potential float inaccuracies -> [0, 2]
        similarity = 1.0 - np.clip(float(distance), 0.0, 2.0)
        # Final clip for safety -> [0, 1]
        return float(np.clip(similarity, 0.0, 1.0))
    except ValueError as ve:
        # logger.debug(f"Cosine distance ValueError: {ve}. Norms: {np.linalg.norm(feat1):.2f}, {np.linalg.norm(feat2):.2f}")
        return 0.0
    except Exception as e:
        logger.error(f"Unexpected error during cosine distance: {e}", exc_info=False)
        return 0.0

def normalize_embedding(embedding: FeatureVector) -> FeatureVector:
    """Normalizes a feature vector (L2 normalization)."""
    # --- Function remains the same as original ---
    if embedding is None: return embedding # Should not happen but safe check
    norm = np.linalg.norm(embedding)
    if norm < 1e-6: # Avoid division by zero or near-zero
        # logger.debug("Embedding norm near zero, returning original.")
        return embedding
    return embedding / norm

def normalize_overlap_set(overlap_set: Set[Tuple[str, str]]) -> Set[Tuple[str, str]]:
    """Ensures camera pairs in the set are stored consistently (sorted tuple)."""
    normalized = set()
    for c1, c2 in overlap_set:
        # Sort the tuple elements alphabetically before adding to the set
        normalized.add(tuple(sorted((c1, c2))))
    return normalized