"""Finite authenticated NORMAL releases, not arbitrary source overrides."""

from pathlib import Path

from unified_normal_contract import digest, read, require

LEGACY = "REPAIRED_NORMAL_V1"
REPAIRED = "REPAIRED_NORMAL_707_V1"
REMAINING = "REMAINING_NORMAL_FIXED_V1"
FRESH_RESTART = "SORTING_DELTA303_FRESH_RESTART_V1"
REMAINING_SIX = "REMAINING_SIX_ELASTIC_FIXED_V1"
PROFILES = {
    REMAINING_SIX: {
        "source": "115df61b5390f324fd06fc81bb091d04a0bce74dfa8587be5aa560b01ed27e2e",
        "archive": "a283729bc9c21aca47c9d5fd66515cfd2ec3b4c35bbd231576668a21f932114d",
        "prefix": "normal-package/",
        "review_policy": "EXTERNAL_TYPED_NORMAL_SOURCE_AND_PLAN",
    },
    FRESH_RESTART: {
        "source": "115df61b5390f324fd06fc81bb091d04a0bce74dfa8587be5aa560b01ed27e2e",
        "archive": "a283729bc9c21aca47c9d5fd66515cfd2ec3b4c35bbd231576668a21f932114d",
        "prefix": "normal-package/",
        "review_policy": "EXTERNAL_TYPED_NORMAL_SOURCE_AND_PLAN",
    },
    REMAINING: {
        "source": "d7e97c813df77ee30b613c77aaf0b8a47934211446116e9bd58ec48b6f8250a0",
        "archive": "3128c670a9a86ad2a01a1f31367f0d64eccbe191a71611fdd911a66df31a3117",
        "prefix": "normal-package/",
        "review_policy": "EXTERNAL_TYPED_NORMAL_SOURCE_AND_PLAN",
    },
    LEGACY: {
        "source": "89b7e3ac5b42716e24d0d2a1994d13c32845e0c93a668f284e3cb341fe301cb8",
        "archive": "67861b9b2e62081c4b486ac918c0716c32257f3c5af5fe6b5bf01aa5bb62cf1c",
        "prefix": "normal-package/",
    },
    REPAIRED: {
        "source": "707e957e95be410aed329235fbaedc72459d854accf1962c73349bc630f2b3d7",
        "archive": "1977b149125adef3fea8604c6b7f61b4cec68c8edf5873b50d4b55b3e4f9e9cc",
        "prefix": "payload/normal-package/",
        "technical_review": "03675a71760db3c21a51c58779e562dc99b834b36188de3f646af6c800dccd1e",
    },
}


def is_remaining_profile(name):
    """Finite old/new producer coexistence; never arbitrary source profiles."""
    return name in (REMAINING, FRESH_RESTART, REMAINING_SIX)


def selected(name=LEGACY):
    require(type(name) is str and name in PROFILES, "UNIFIED_SUPPORTED_SOURCE_PROFILE")
    return PROFILES[name]


def authenticate_package(package, name):
    """Authenticate the exact identity, closed member set and every payload byte."""
    expected = selected(name)
    root = Path(package)
    require(not root.is_symlink() and not root.is_junction(), "UNIFIED_NORMAL_ROOT_CLOSURE")
    require(
        digest(root / "normal-source-identity.json") == expected["source"],
        "UNIFIED_NORMAL_IDENTITY",
    )
    identity = read(root / "normal-source-identity.json")
    paths = list(root.rglob("*"))
    require(
        {p.relative_to(root).as_posix() for p in paths if p.is_file()}
        == set(identity["files"]) | {"normal-source-identity.json"}
        and all(not p.is_symlink() and not p.is_junction() for p in paths),
        "UNIFIED_NORMAL_ROOT_CLOSURE",
    )
    for member, pin in identity["files"].items():
        require(
            bool(member)
            and not Path(member).is_absolute()
            and ".." not in member.split("/")
            and "\\" not in member
            and ":" not in member,
            "UNIFIED_NORMAL_SAFE_SOURCE",
        )
        path = root / member
        require(
            path.is_file()
            and path.stat().st_size == pin["bytes"]
            and digest(path) == pin["sha256"],
            "UNIFIED_NORMAL_SOURCE_BYTES",
        )
    return identity


def authenticate_review(path, name):
    """Source technical provenance is distinct from the new plan's typed review."""
    expected = selected(name)
    require(
        "technical_review" in expected and digest(path) == expected["technical_review"],
        "UNIFIED_SOURCE_TECHNICAL_REVIEW",
    )
    review = read(path)
    require(
        review["passed"] is True
        and review["security_concerns"] == []
        and review["logic_errors"] == [],
        "UNIFIED_SOURCE_TECHNICAL_REVIEW",
    )
