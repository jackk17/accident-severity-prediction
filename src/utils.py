"""
Utility functions
"""

import os
import yaml
from pathlib import Path


def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def get_project_root():
    """Get the root directory of the project"""
    return Path(__file__).parent.parent


def ensure_dir_exists(directory):
    """Ensure a directory exists"""
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_model_path():
    """Get the path where models are stored"""
    root = get_project_root()
    model_dir = root / "models"
    ensure_dir_exists(model_dir)
    return model_dir


def get_data_path(data_type="raw"):
    """Get the path for data storage"""
    root = get_project_root()
    data_dir = root / "data" / data_type
    ensure_dir_exists(data_dir)
    return data_dir
