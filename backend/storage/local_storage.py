"""
Local filesystem storage implementation.
Stores uploaded images in the configured upload directory.
"""

import os
import aiofiles

from backend.storage.base_storage import AbstractStorage
from backend.core.config import get_settings
from backend.core.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class LocalStorage(AbstractStorage):
    """
    Saves files to the local filesystem inside `uploads/`.
    To switch to Firebase/S3, implement AbstractStorage and swap this class
    in `core/dependencies.py`.
    """

    def __init__(self, upload_dir: str | None = None) -> None:
        self._upload_dir = upload_dir or settings.resolved_upload_dir
        os.makedirs(self._upload_dir, exist_ok=True)

    async def save(self, file_bytes: bytes, filename: str) -> str:
        file_path = os.path.join(self._upload_dir, filename)
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(file_bytes)
        logger.info("File saved: %s", file_path)
        return file_path

    async def delete(self, path: str) -> None:
        try:
            os.remove(path)
            logger.info("File deleted: %s", path)
        except FileNotFoundError:
            logger.warning("File not found for deletion: %s", path)

    def get_url(self, path: str) -> str:
        """Return relative URL served by FastAPI StaticFiles."""
        filename = os.path.basename(path)
        return f"/uploads/{filename}"
