"""Normalize user folder paths (Windows, WSL, quoted, file://)."""

from __future__ import annotations

import os
import re
from pathlib import Path


_WIN_DRIVE = re.compile(r"^([A-Za-z]):[\\/](.*)$")
_WIN_DRIVE_SLASH = re.compile(r"^/([A-Za-z]):[\\/]?(.*)$")  # /C:/Users/...
_WSL_UNC = re.compile(r"^//wsl(?:\.localhost)?/[^/]+/(.*)$", re.I)


def normalize_user_path(raw: str) -> Path:
    """Turn UI/CLI path text into a usable local Path.

    Accepts:
      /home/dev/...
      /mnt/c/Users/...
      C:\\Users\\...
      C:/Users/...
      /C:/Users/...
      file:///C:/Users/...
      \\\\wsl$\\...
      ~/Documents
    """
    s = (raw or "").strip().strip('"').strip("'")
    if not s:
        raise ValueError("empty path")

    # file URI
    if s.lower().startswith("file:"):
        # file:///C:/Users/... or file:///mnt/c/...
        s = re.sub(r"^file://", "", s, flags=re.I)
        # leading slash before drive: /C:/...
        s = s.replace("%20", " ")

    s = s.replace("\0", "")

    # UNC \\wsl$\Ubuntu\home\... → leave as-is if exists, else try strip
    if s.startswith("\\\\") or s.startswith("//"):
        s2 = s.replace("\\", "/")
        m = _WSL_UNC.match(s2)
        if m:
            s = "/" + m.group(1)
        else:
            s = s2

    # Normalize slashes for drive detection
    mixed = s.replace("\\", "/")

    # /C:/Users/... or /C:/Users...
    m = _WIN_DRIVE_SLASH.match(mixed)
    if m:
        drive, rest = m.group(1).lower(), m.group(2)
        s = f"/mnt/{drive}/{rest}"
        mixed = s

    # C:/Users/... or C:\Users\...
    m = _WIN_DRIVE.match(s.replace("/", "\\") if "\\" in s else mixed)
    # Prefer matching on original with either slash
    m = re.match(r"^([A-Za-z]):[\\/](.*)$", s)
    if not m:
        m = re.match(r"^([A-Za-z]):[\\/](.*)$", mixed)
    if m:
        drive, rest = m.group(1).lower(), m.group(2).replace("\\", "/")
        s = f"/mnt/{drive}/{rest}"

    # Collapse duplicate slashes (not leading //)
    if s.startswith("/"):
        s = "/" + re.sub(r"/+", "/", s.lstrip("/"))

    path = Path(s).expanduser()
    # resolve without requiring existence first (clearer errors)
    try:
        path = path.resolve(strict=False)
    except OSError:
        path = Path(os.path.abspath(os.path.expanduser(s)))

    return path


def describe_path_help(failed: str | Path) -> str:
    p = str(failed)
    return (
        f"not a directory: {p}\n"
        "Tip (WSL): use /mnt/c/Users/<you>/... or C:\\Users\\<you>\\... "
        "(Windows paths are converted automatically)."
    )
