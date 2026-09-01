#!/usr/bin/env python
"""Dependency-free integrity checks for the development repository."""

from __future__ import annotations

import ast
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    ".github/workflows/tests.yml",
    "LICENSE",
    "README.md",
    "pyproject.toml",
    "src/sarflood/data/dataset.py",
    "tests/test_dataset.py",
    "tests/test_smoke.py",
}


def tracked_files() -> set[str]:
    result = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return set(result.stdout.splitlines())


def validate() -> list[str]:
    errors: list[str] = []
    tracked = tracked_files()
    missing = REQUIRED - tracked
    if missing:
        errors.append(f"Missing required files: {sorted(missing)}")

    forbidden_suffixes = {".pt", ".pth", ".ckpt", ".onnx", ".pyc"}
    artifacts = sorted(path for path in tracked if Path(path).suffix in forbidden_suffixes)
    if artifacts:
        errors.append(f"Generated artifacts are tracked: {artifacts}")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    if "<your-fork-url>" in readme:
        errors.append("README contains the placeholder clone URL")
    if not re.search(r"under active\s*>?\s*development", readme, flags=re.IGNORECASE):
        errors.append("README must state that the project is under active development")
    if readme.count("```") % 2:
        errors.append("README has an unclosed fenced code block")
    for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", readme):
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        relative = target.split("#", 1)[0]
        if relative and not (ROOT / relative).exists():
            errors.append(f"README links to a missing path: {relative}")

    gitignore_lines = {
        line.strip() for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    }
    if "data/" in gitignore_lines:
        errors.append("Use /data/ in .gitignore; data/ also hides src/sarflood/data")

    for path in sorted(ROOT.rglob("*.py")):
        if any(part.startswith(".") for part in path.relative_to(ROOT).parts):
            continue
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            errors.append(f"Invalid Python in {path.relative_to(ROOT)}: {exc}")

    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for number, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") == "code" and cell.get("outputs"):
                errors.append(f"{path.relative_to(ROOT)} cell {number} contains outputs")
            if cell.get("cell_type") == "code":
                source = "".join(cell.get("source", []))
                source = "\n".join(
                    (line[: len(line) - len(line.lstrip())] + "pass")
                    if line.lstrip().startswith(("%", "!")) and line != line.lstrip()
                    else line
                    for line in source.splitlines()
                    if not (line.lstrip().startswith(("%", "!")) and line == line.lstrip())
                )
                try:
                    ast.parse(source)
                except SyntaxError as exc:
                    errors.append(
                        f"Invalid Python in {path.relative_to(ROOT)} cell {number}: {exc}"
                    )

    secret_pattern = re.compile(
        r"(?i)(api[_-]?key|client[_-]?secret|password|access[_-]?token)\s*=\s*['\"][^'\"]+"
    )
    for path in sorted(ROOT.rglob("*")):
        relative_parts = path.relative_to(ROOT).parts
        if (
            not path.is_file()
            or any(part.startswith(".") for part in relative_parts)
            or path.suffix in {".png", ".jpg"}
        ):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if secret_pattern.search(text):
            errors.append(f"Potential credential assignment in {path.relative_to(ROOT)}")
    return errors


def main() -> None:
    errors = validate()
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print("Repository validation passed")


if __name__ == "__main__":
    main()
