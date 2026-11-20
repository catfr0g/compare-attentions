"""Preserve only the bootstrap's revalidated single driver root, never ambient paths."""

import importlib.util
from pathlib import Path


def validated_driver_environment(package, environment):
    """Call the identity-authenticated existing validator; do not weaken its policy."""
    incoming = environment.get("LD_LIBRARY_PATH")
    if incoming is None:
        return {}
    validator = Path(package) / "inherited/tools/study_environment.py"
    spec = importlib.util.spec_from_file_location("_normal_driver_validator", validator)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    driver = module.driver_directory()
    if (
        type(driver) is not str
        or not driver
        or not Path(driver).is_absolute()
        or incoming != driver
        or any(character in driver for character in ("\x00", "\n", "\r"))
    ):
        raise ValueError("NORMAL_VALIDATED_DRIVER_REQUIRED")
    return {"LD_LIBRARY_PATH": driver}
