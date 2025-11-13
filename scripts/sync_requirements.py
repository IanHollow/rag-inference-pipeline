#!/usr/bin/env python3
"""
Sync requirements.txt with dependencies from pyproject.toml.

This script extracts the direct dependencies from pyproject.toml
and writes them to requirements.txt, preserving the original version
specifiers (like >=) without pinning or including transitive dependencies.
"""

from pathlib import Path

import tomli as tomllib


def sync_requirements() -> None:
    # Get project root (parent of scripts folder)
    project_root = Path(__file__).parent.parent
    pyproject_path = project_root / "pyproject.toml"
    requirements_path = project_root / "requirements.txt"

    # Read pyproject.toml
    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)

    # Extract dependencies
    deps = data["project"]["dependencies"]

    # Write to requirements.txt
    with requirements_path.open("w") as f:
        f.write("\n".join(deps) + "\n")

    print(f"Synced {len(deps)} dependencies to requirements.txt")


if __name__ == "__main__":
    sync_requirements()
