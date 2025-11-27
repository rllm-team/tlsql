"""Configuration module 

Contains configuration files and utilities for TL-SQL.
"""

import json
from typing import Dict, Any

try:
    # Py3.9+ 
    from importlib import resources
except ImportError:  # pragma: no cover
    # Py3.8 fallback 
    import importlib_resources as resources  # type: ignore

_CONFIG_PACKAGE = __package__
_DEFAULT_CONFIG_FILE = "dataset_config.json"


def _get_resource(name: str):
    """Return resource handle"""
    return resources.files(_CONFIG_PACKAGE).joinpath(name)


def load_dataset_config(filename: str = _DEFAULT_CONFIG_FILE) -> Dict[str, Any]:
    """Load dataset configuration 
    
    Args:
        filename: Config file name 
    
    Returns:
        Dictionary containing dataset configurations 
    """
    resource = _get_resource(filename)
    with resource.open('r', encoding='utf-8') as fp:
        try:
            return json.load(fp)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON config: {resource}") from exc

