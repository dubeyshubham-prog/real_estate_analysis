from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from config.settings import Config
from src.common.exceptions import UploadValidationError


async def validate_upload(
    uploaded_file: UploadFile,
    allowed_content_types: set[str],
    allowed_extensions: set[str],
) -> bytes:
    """Validate upload metadata and enforce the configured size limit."""
    extension = Path(uploaded_file.filename or "").suffix.lower()
    if extension not in allowed_extensions:
        raise UploadValidationError(
            f"Unsupported file extension: {extension or 'missing'}"
        )
    if uploaded_file.content_type not in allowed_content_types:
        raise UploadValidationError(
            f"Unsupported content type: {uploaded_file.content_type}"
        )

    maximum_bytes = Config.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    content = await uploaded_file.read(maximum_bytes + 1)
    await uploaded_file.seek(0)
    if len(content) > maximum_bytes:
        raise UploadValidationError(
            f"Upload exceeds {Config.MAX_UPLOAD_SIZE_MB} MB"
        )
    if not content:
        raise UploadValidationError("Uploaded file is empty")
    return content


def safe_upload_path(directory: Path, original_filename: str) -> Path:
    """Generate a server-controlled filename while preserving the extension."""
    extension = Path(original_filename).suffix.lower()
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{uuid4().hex}{extension}"
