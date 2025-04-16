"""Constants used across the pipeline module."""

# Default Config Values for Pruning
DEFAULT_PRUNE_INTERVAL = 500 # Frames between gallery prune checks
DEFAULT_PRUNE_THRESHOLD_MULTIPLIER = 2.0 # Prune if unseen for X * lost_track_buffer_frames