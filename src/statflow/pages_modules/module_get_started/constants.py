"""
Constants for the get started module.

This module contains enums and constants used across the get started functionality.
"""

from enum import Enum


class DatasetParamMode(str, Enum):
    EXPERIMENT_NAMES_AS_DATASETS = "EXPERIMENT_NAMES_AS_DATASETS"
    SINGLE_DATASET_MODE = "SINGLE_DATASET_MODE"