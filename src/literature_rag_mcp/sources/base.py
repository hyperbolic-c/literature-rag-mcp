"""Abstract base class for data sources."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SourceItem:
    """Represents an item from a data source."""
    key: str
    title: str
    item_type: str
    creators: str = ""
    abstract: str = ""
    doi: str = ""
    date: str = ""


@dataclass
class SourceAttachment:
    """Represents an attachment (PDF) for an item."""
    key: str
    parent_key: str
    filename: str
    content_type: str
    path: str = ""


class AbstractSource(ABC):
    """Abstract interface for data sources."""

    @abstractmethod
    def get_items(self, limit: int | None = None) -> list[SourceItem]:
        """Get items from the source."""
        pass

    @abstractmethod
    def get_item_by_key(self, key: str) -> SourceItem | None:
        """Get a single item by key."""
        pass

    @abstractmethod
    def get_attachments(self, item_key: str) -> list[SourceAttachment]:
        """Get attachments for an item."""
        pass
