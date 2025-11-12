"""Basic __init__.py
Allows to import the __all__ by folder name.
"""
from .database import ToyDatabase, OlocumDatabase

__all__ = [
    "ToyDatabase",
    "OlocumDatabase"
]
