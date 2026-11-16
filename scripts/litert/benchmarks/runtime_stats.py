"""Process and runtime-cache measurements shared by the LiteRT-LM benchmark scripts."""

import platform
import resource
from pathlib import Path


def directory_bytes(path: Path) -> int:
    """Return the bytes used by regular files under a directory, such as a LiteRT-LM runtime cache."""
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def peak_rss_bytes() -> int:
    """Return this process's peak resident set size in bytes.

    `ru_maxrss` is in bytes on macOS and in KiB on Linux.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak * 1024 if platform.system() == "Linux" else peak
