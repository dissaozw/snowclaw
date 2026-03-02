"""Model weight download utility with progress bar and caching."""

from __future__ import annotations

import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path


DEFAULT_CACHE_DIR = Path.home() / ".cache" / "snowclaw" / "models"


def download_model(
    url: str,
    filename: str,
    cache_dir: Path | None = None,
    expected_sha256: str | None = None,
) -> Path:
    """
    Download a model file with progress indicator and local caching.

    Args:
        url: URL to download the model from.
        filename: Filename to save the model as.
        cache_dir: Directory to cache models in. Defaults to ~/.cache/snowclaw/models/.
        expected_sha256: Optional SHA-256 hash to verify the download.

    Returns:
        Path to the cached model file.

    Raises:
        RuntimeError: If download fails or hash mismatch.
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / filename

    if model_path.exists():
        if expected_sha256 and _sha256(model_path) != expected_sha256:
            model_path.unlink()  # Re-download if hash mismatch
        else:
            return model_path

    tmp_path = model_path.with_suffix(".tmp")
    try:
        _download_with_progress(url, tmp_path)
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download model from {url}: {e}") from e

    if expected_sha256:
        actual = _sha256(tmp_path)
        if actual != expected_sha256:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"SHA-256 mismatch for {filename}: expected {expected_sha256}, got {actual}"
            )

    shutil.move(str(tmp_path), str(model_path))
    return model_path


def _download_with_progress(url: str, dest: Path) -> None:
    """Download a file with a progress bar to stderr."""
    req = urllib.request.urlopen(url)  # noqa: S310
    total = int(req.headers.get("Content-Length", 0))
    downloaded = 0
    block_size = 8192

    with open(dest, "wb") as f:
        while True:
            chunk = req.read(block_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded / total * 100
                bar_len = 30
                filled = int(bar_len * downloaded / total)
                bar = "=" * filled + "-" * (bar_len - filled)
                sys.stderr.write(
                    f"\rDownloading {dest.name}: [{bar}] {pct:.1f}%"
                )
                sys.stderr.flush()

    if total > 0:
        sys.stderr.write("\n")


def _sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
