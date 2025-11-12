"""
Catalog Module

Define Catalog and CatalogEntry abstraction.
"""
import abc
import typing

import pandas as pd

class CatalogEntry(abc.ABC):
    """Abstract base class for elements stored in a Catalog."""

    @abc.abstractmethod
    def as_dict(self) -> dict[str, typing.Any]:
        """Return a dictionary representation of the object."""

T = typing.TypeVar("T", bound="CatalogEntry")

class Catalog(typing.Generic[T]):
    """Generic container that holds and serializes CatalogEntry instances."""

    def __init__(self, entries: typing.List[T]):
        super().__init__()
        self.entries = entries

    def __iter__(self) -> typing.Iterator[T]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> T:
        return self.entries[index]

    def to_df(self) -> pd.DataFrame:
        """Return a pandas DataFrame with the full catalog information."""
        return pd.DataFrame([entry.as_dict() for entry in self.entries])
