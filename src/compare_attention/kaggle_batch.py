"""План последовательных запусков YAML-экспериментов на Kaggle."""

import re
from pathlib import Path

import yaml

from compare_attention.cli.train import load_experiment


def plan_experiments(config: Path | None, base_slug: str) -> list[tuple[Path | None, str]]:
    """Развернуть папку в запуски; отдельный файл сохраняет исходный slug."""
    if config is None:
        return [(None, base_slug)]
    if not config.exists():
        raise ValueError(f"--config: путь не существует: {config}")
    batch = config.is_dir()
    files = (
        sorted(
            (
                path
                for path in config.iterdir()
                if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}
            ),
            key=lambda path: path.name,
        )
        if batch
        else [config]
    )
    if not files:
        raise ValueError(f"--config: в папке {config} нет файлов .yaml/.yml")
    for path in files:
        try:
            load_experiment(path)
        except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
            raise ValueError(f"некорректный YAML {path}: {exc}") from exc
    experiments: list[tuple[Path | None, str]] = [
        (path, _batch_slug(base_slug, path) if batch else base_slug) for path in files
    ]
    seen: dict[str, Path | None] = {}
    for path, slug in experiments:
        if slug in seen:
            raise ValueError(f"коллизия slug {slug!r}: {seen[slug]} и {path}")
        seen[slug] = path
    return experiments


def _batch_slug(base_slug: str, path: Path) -> str:
    """Имя файла в lowercase ASCII, разделители в дефисы; не более 50 символов."""
    name = re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-")
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", base_slug) or len(base_slug) > 48:
        raise ValueError(
            f"недопустимый slug {base_slug!r}: база должна содержать 1–48 символов a-z, 0-9 "
            "и одиночные дефисы между ними"
        )
    slug = f"{base_slug}-{name}"[:50].rstrip("-")
    if not name or len(slug) < 5:
        raise ValueError(f"недопустимый slug для {path}: имя должно дать ASCII slug длиной 5–50")
    return slug
