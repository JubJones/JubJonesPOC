# -*- coding: utf-8 -*-
"""Initialization file for the pipeline_module package."""

# Expose the main pipeline class for easier import
from .core import MultiCameraPipeline

__all__ = ["MultiCameraPipeline"]

# Optionally, configure logging for the module here if desired
# import logging
# logging.getLogger(__name__).addHandler(logging.NullHandler())