"""Гейт: весь src обязан разбираться питоном 3.13 (Kaggle-совместимость)."""

import ast
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"


def test_every_source_file_parses_under_python_313() -> None:
    problems = []
    for path in sorted(_SRC.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        try:
            ast.parse(source, feature_version=(3, 13))
        except SyntaxError as exc:
            problems.append(f"{path.relative_to(_SRC)}: {exc.msg} (строка {exc.lineno})")
    assert not problems, "не 3.13-совместимо:\n" + "\n".join(problems)
