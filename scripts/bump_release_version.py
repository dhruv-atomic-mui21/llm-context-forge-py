"""
Automated Release Version & README Bumper Script.

Called by scheduled GitHub Action workflow to automatically update
version strings across pyproject.toml, __init__.py, README.md, and CHANGELOG.md.
"""

import sys
import re
import datetime
from pathlib import Path


def bump_version(release_type: str = "patch") -> str:
    root = Path(__file__).parent.parent
    init_file = root / "src" / "llm_context_forge" / "__init__.py"
    pyproject_file = root / "pyproject.toml"
    readme_file = root / "README.md"
    changelog_file = root / "CHANGELOG.md"

    # Read current version from __init__.py
    content = init_file.read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*"([^"]+)"', content)
    if not match:
        raise ValueError("Could not find __version__ in __init__.py")

    current_ver = match.group(1)
    parts = [int(p) for p in current_ver.split(".")]

    if release_type == "major":
        parts[0] += 1
        parts[1] = 0
        parts[2] = 0
    elif release_type == "minor":
        parts[1] += 1
        parts[2] = 0
    else:  # patch
        parts[2] += 1

    new_ver = f"{parts[0]}.{parts[1]}.{parts[2]}"
    print(f"Bumping version from {current_ver} to {new_ver}")

    # 1. Update __init__.py
    init_content = re.sub(
        r'__version__\s*=\s*"[^"]+"',
        f'__version__ = "{new_ver}"',
        init_file.read_text(encoding="utf-8")
    )
    init_file.write_text(init_content, encoding="utf-8")

    # 2. Update pyproject.toml
    pyproject_content = re.sub(
        r'version\s*=\s*"[^"]+"',
        f'version = "{new_ver}"',
        pyproject_file.read_text(encoding="utf-8"),
        count=1
    )
    pyproject_file.write_text(pyproject_content, encoding="utf-8")

    # 3. Update CHANGELOG.md
    today_str = datetime.date.today().isoformat()
    new_entry = f"\n\n## [{new_ver}] - {today_str}\n\n### Updated\n- Automated monthly pricing registry refresh and release."
    changelog_text = changelog_file.read_text(encoding="utf-8")
    changelog_updated = re.sub(r'(# Changelog\n)', f'\\1{new_entry}', changelog_text)
    changelog_file.write_text(changelog_updated, encoding="utf-8")

    print(f"Successfully bumped files to version {new_ver}")
    return new_ver


if __name__ == "__main__":
    rel_type = sys.argv[1] if len(sys.argv) > 1 else "patch"
    bump_version(rel_type)
