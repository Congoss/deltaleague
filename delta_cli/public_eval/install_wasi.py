"""Install the hash-pinned WASI interpreter during image construction."""
import hashlib
import io
from pathlib import Path
import sys
from urllib.request import urlopen
import zipfile
from .wasi import ARCHIVE_URL, ARCHIVE_SHA256


def install(root):
    raw = urlopen(ARCHIVE_URL, timeout=120).read(64 * 1024**2)
    if hashlib.sha256(raw).hexdigest() != ARCHIVE_SHA256:
        raise ValueError('WASI archive integrity mismatch')
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for member in archive.infolist():
            if not (root / member.filename).resolve().is_relative_to(root.resolve()):
                raise ValueError('Unsafe WASI archive path')
        archive.extractall(root)


if __name__ == '__main__':
    install(Path(sys.argv[1]))
