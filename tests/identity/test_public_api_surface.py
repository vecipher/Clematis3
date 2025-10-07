

from __future__ import annotations

import types

# Frozen public surfaces for v3 (scoped to clematis and clematis.errors)
EXPECTED_CLEMATIS = {
    "__version__",
    # Config (v1)
    "CONFIG_VERSION",
    "validate_config",
    "validate_config_verbose",
    "validate_config_api",
    # Snapshot (v1)
    "SCHEMA_VERSION",
    # Submodule
    "errors",
}

EXPECTED_ERRORS = {
    "ClematisError",
    "ConfigError",
    "SnapshotError",
    "IdentityError",
    "QualityShadowError",
    "ReflectionError",
    "ParallelError",
    "CLIError",
}

# Some repos expose this helper publicly; allow it without failing the freeze.
OPTIONAL_ERRORS = {"format_error"}


def _public_only(names: set[str]) -> set[str]:
    # Treat __version__ as part of the public surface even though it starts with "_"
    return {n for n in names if n == "__version__" or not n.startswith("_")}


def test_clematis_star_is_exact():
    ns: dict[str, object] = {}
    exec("from clematis import *", {}, ns)
    got = _public_only(set(ns.keys()))
    assert got == EXPECTED_CLEMATIS, (
        f"Unexpected clematis surface:\n"
        f"  extra: {sorted(got - EXPECTED_CLEMATIS)}\n"
        f"  missing: {sorted(EXPECTED_CLEMATIS - got)}"
    )


def test_errors_star_is_exact_with_optionals():
    ns: dict[str, object] = {}
    exec("from clematis.errors import *", {}, ns)
    got = _public_only(set(ns.keys()))
    missing = EXPECTED_ERRORS - got
    # Allow only OPTIONAL_ERRORS beyond the frozen set
    extras = got - (EXPECTED_ERRORS | OPTIONAL_ERRORS)
    assert not missing and not extras, (
        f"Unexpected errors surface:\n"
        f"  extra (disallowed): {sorted(extras)}\n"
        f"  missing: {sorted(missing)}"
    )


def test_internal_modules_not_leaked_by_star():
    ns: dict[str, object] = {}
    exec("from clematis import *", {}, ns)
    forbidden = {"engine", "io", "scripts", "configs"}
    leaked = forbidden & set(ns)
    assert not leaked, f"Internal modules leaked via star: {sorted(leaked)}"


def test_all_is_sorted_and_matches_surface():
    import clematis
    import clematis.errors as cerr

    # __all__ exists and is sorted deterministically
    assert hasattr(clematis, "__all__")
    assert hasattr(cerr, "__all__")
    assert list(clematis.__all__) == sorted(clematis.__all__)
    assert list(cerr.__all__) == sorted(cerr.__all__)

    # clematis top-level must match exactly
    assert set(clematis.__all__) == EXPECTED_CLEMATIS

    # errors must contain the frozen set; allow only OPTIONAL_ERRORS as extras
    err_all = set(cerr.__all__)
    missing = EXPECTED_ERRORS - err_all
    extras = err_all - (EXPECTED_ERRORS | OPTIONAL_ERRORS)
    assert not missing and not extras, (
        f"errors.__all__ mismatch:\n"
        f"  extra (disallowed): {sorted(extras)}\n"
        f"  missing: {sorted(missing)}"
    )


def test_errors_module_object_is_exposed_at_top_level():
    import clematis

    assert isinstance(clematis.errors, types.ModuleType), (
        "`clematis.errors` must be a submodule exposed at the top level"
    )
