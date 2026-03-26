"""
Abstract storage interface.
Swap Local → Firebase/S3 by implementing this interface without touching Services.
"""

from abc import ABC, abstractmethod


class AbstractStorage(ABC):

    @abstractmethod
    async def save(self, file_bytes: bytes, filename: str) -> str:
        """
        Save file bytes and return the stored path / URL.

        Args:
            file_bytes: Raw bytes of the uploaded file.
            filename:   Target filename (already sanitized, UUID-based).

        Returns:
            Stored path string — used as `image_path` in the DB.
        """
        ...

    @abstractmethod
    async def delete(self, path: str) -> None:
        """Delete a file by its stored path. Silently ignore if not found."""
        ...

    @abstractmethod
    def get_url(self, path: str) -> str:
        """Return a publicly accessible URL for the given path."""
        ...
