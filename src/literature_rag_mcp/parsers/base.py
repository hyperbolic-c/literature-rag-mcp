"""Abstract base class for document parsers."""

from abc import ABC, abstractmethod


class AbstractParser(ABC):
    """Abstract interface for document parsers."""

    @abstractmethod
    def get_markdown(self, attachment_key: str) -> str:
        """Get markdown content for an attachment."""
        pass

    @abstractmethod
    def parse(self, pdf_path: str) -> str:
        """Parse a PDF file and return markdown."""
        pass
