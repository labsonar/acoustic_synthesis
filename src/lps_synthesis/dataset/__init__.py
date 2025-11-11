"""Basic __init__.py
Allows to import the __all__ by folder name.
"""
from .details import ToyCatalog, OlocumCatalog

__all__ = [
    "ToyCatalog",
    "OlocumCatalog"
]
