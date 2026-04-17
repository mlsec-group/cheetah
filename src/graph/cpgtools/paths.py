"""Filesystem paths for CPG tooling (no setuptools package_data)."""

from __future__ import annotations

from pathlib import Path


def joern_scripts_dir() -> Path:
    return Path(__file__).resolve().parent / "joern_scripts"
