#!/usr/bin/env python3
"""Check concrete structural guardrails for Logismos's declared runtime scope.

This guard does not infer program semantics from prose or source-code tokens. Reviewers remain
responsible for deciding whether new behavior stays inside the declared product boundary.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "contracts/runtime-scope.toml"
EXPECTED_CAPABILITIES = ["load", "quantize", "infer", "serve"]
EXPECTED_EXCLUSIONS = [
    "general-model-formation",
    "general-training",
    "model-release",
]
EXPECTED_RETIRED = ["autograd", "optim", "data", "melete"]
EXPECTED_LICENSE = "LicenseRef-PolyForm-Noncommercial-1.0.0"
EXPECTED_ROOT_KEYS = frozenset({"schema", "scope", "bounded_adaptation"})
EXPECTED_SCOPE_KEYS = frozenset(
    {
        "owner",
        "capabilities",
        "excluded_authority",
        "retired_packages",
        "retired_paths",
    }
)
EXPECTED_ADAPTATION_KEYS = frozenset({"default", "admission", "requirements"})
CARGO_METADATA_COMMAND = (
    "cargo",
    "metadata",
    "--format-version",
    "1",
    "--no-deps",
    "--locked",
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def string_list(table: dict[str, object], key: str) -> list[str]:
    value = table.get(key)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        fail(f"{key} must be an array of strings")
    return value


def require_exact_keys(
    table: dict[str, object], expected: frozenset[str], label: str
) -> None:
    actual = set(table)
    missing = sorted(expected - actual)
    unexpected = sorted(str(key) for key in actual - expected)
    if not missing and not unexpected:
        return

    details = []
    if missing:
        details.append("missing " + ", ".join(missing))
    if unexpected:
        details.append("unexpected " + ", ".join(unexpected))
    fail(f"{label} keys are not exact ({'; '.join(details)})")


def validate_contract(raw: dict[str, object]) -> tuple[list[str], list[str]]:
    require_exact_keys(raw, EXPECTED_ROOT_KEYS, "runtime scope root")
    schema = raw.get("schema")
    if type(schema) is not int or schema != 1:
        fail("runtime scope schema must be the integer 1")

    scope = raw.get("scope")
    if not isinstance(scope, dict):
        fail("runtime scope must be a table")
    require_exact_keys(scope, EXPECTED_SCOPE_KEYS, "runtime scope")
    if scope.get("owner") != "logismos":
        fail("runtime scope must be owned by logismos")
    if string_list(scope, "capabilities") != EXPECTED_CAPABILITIES:
        fail("runtime capabilities must remain load, quantize, infer, serve")
    if string_list(scope, "excluded_authority") != EXPECTED_EXCLUSIONS:
        fail("general formation, training, and release authority must remain excluded")
    retired = string_list(scope, "retired_packages")
    if retired != EXPECTED_RETIRED:
        fail("retired package identities changed; amend the boundary deliberately")
    retired_paths = string_list(scope, "retired_paths")
    if retired_paths != [f"crates/{name}" for name in retired]:
        fail("retired paths must correspond exactly to retired package identities")

    adaptation = raw.get("bounded_adaptation")
    if not isinstance(adaptation, dict):
        fail("bounded adaptation must be a table")
    require_exact_keys(
        adaptation, EXPECTED_ADAPTATION_KEYS, "bounded adaptation"
    )
    if adaptation.get("default") != "absent":
        fail("bounded adaptation must remain absent by default")
    if adaptation.get("admission") != "named-consumer-contract":
        fail("bounded adaptation requires a named consumer contract")
    requirements = string_list(adaptation, "requirements")
    if requirements != [
        "persistent-output-owner",
        "retention-and-revocation",
        "rollback",
    ]:
        fail("bounded-adaptation admission requirements are incomplete")
    return retired, retired_paths


def load_contract() -> tuple[list[str], list[str]]:
    raw = tomllib.loads(CONTRACT.read_text(encoding="utf-8"))
    return validate_contract(raw)


def metadata_packages(metadata: dict[str, object]) -> list[dict[str, object]]:
    packages = metadata.get("packages")
    if not isinstance(packages, list) or any(
        not isinstance(package, dict) or not isinstance(package.get("name"), str)
        for package in packages
    ):
        fail("cargo metadata returned an invalid package array")
    return packages


def workspace_metadata() -> dict[str, object]:
    completed = subprocess.run(
        CARGO_METADATA_COMMAND,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no diagnostic emitted"
        fail(f"cargo metadata --no-deps --locked failed: {detail}")
    metadata = json.loads(completed.stdout)
    if not isinstance(metadata, dict):
        fail("cargo metadata returned a non-object document")
    metadata_packages(metadata)
    return metadata


def discover_retired_paths(root: Path, retired_paths: list[str]) -> set[str]:
    existing = set()
    for relative_path in retired_paths:
        try:
            (root / relative_path).lstat()
        except FileNotFoundError:
            continue
        existing.add(relative_path)
    return existing


def check_retired_inventory(
    retired: list[str],
    retired_paths: list[str],
    existing_paths: set[str],
    package_names: set[str],
    locked_names: set[object],
) -> None:
    present_paths = sorted(set(retired_paths) & existing_paths)
    if present_paths:
        fail(f"retired runtime path reappeared: {', '.join(present_paths)}")

    present_packages = sorted(set(retired) & package_names)
    if present_packages:
        fail(f"retired workspace packages reappeared: {', '.join(present_packages)}")

    present_locks = sorted(set(retired) & locked_names)
    if present_locks:
        fail(f"retired Cargo.lock packages remain: {', '.join(present_locks)}")


def validate_license_derivation(
    cargo: dict[str, object],
    fingerprint: dict[str, object],
    license_bytes: bytes,
    metadata: dict[str, object],
) -> None:
    workspace = cargo.get("workspace")
    if not isinstance(workspace, dict):
        fail("Cargo.toml has no workspace table")
    package = workspace.get("package")
    if not isinstance(package, dict):
        fail("Cargo.toml has no workspace.package table")
    spdx = package.get("license")
    if spdx != EXPECTED_LICENSE:
        fail("workspace license must remain PolyForm Noncommercial 1.0.0")

    hashes = fingerprint.get("sha256")
    if not isinstance(hashes, dict):
        fail("license fingerprint has no sha256 table")
    actual = hashlib.sha256(license_bytes).hexdigest()
    if hashes.get("LICENSE") != actual:
        fail("license fingerprint does not match LICENSE")
    if not license_bytes.startswith(b"# PolyForm Noncommercial License 1.0.0"):
        fail("LICENSE text does not match the workspace license identity")

    packages = metadata_packages(metadata)
    mismatched_packages = sorted(
        str(package.get("name"))
        for package in packages
        if package.get("license") != EXPECTED_LICENSE
    )
    if mismatched_packages:
        fail(
            "workspace packages do not inherit the declared license: "
            + ", ".join(mismatched_packages)
        )


def check_license_derivation(metadata: dict[str, object]) -> None:
    cargo = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))
    fingerprint = tomllib.loads(
        (ROOT / ".kanon-license-fingerprint.toml").read_text(encoding="utf-8")
    )
    validate_license_derivation(cargo, fingerprint, (ROOT / "LICENSE").read_bytes(), metadata)


def expect_failure(action: object, expected: str) -> None:
    if not callable(action):
        raise AssertionError("self-test action must be callable")
    try:
        action()
    except RuntimeError as error:
        if expected not in str(error):
            raise AssertionError(
                f"self-test expected {expected!r} in failure, got {str(error)!r}"
            ) from error
    else:
        raise AssertionError(f"self-test expected failure containing {expected!r}")


def replace_fixture_once(source: str, marker: str, replacement: str) -> str:
    if source.count(marker) != 1:
        raise AssertionError(f"TOML fixture marker must occur exactly once: {marker!r}")
    return source.replace(marker, replacement, 1)


def run_self_tests() -> None:
    contract_source = CONTRACT.read_text(encoding="utf-8")
    contract: dict[str, object] = tomllib.loads(contract_source)
    retired, retired_paths = validate_contract(contract)
    check_retired_inventory(retired, retired_paths, set(), set(), set())

    boolean_schema = tomllib.loads(
        replace_fixture_once(contract_source, "schema = 1", "schema = true")
    )
    expect_failure(
        lambda: validate_contract(boolean_schema), "schema must be the integer 1"
    )

    float_schema = tomllib.loads(
        replace_fixture_once(contract_source, "schema = 1", "schema = 1.0")
    )
    expect_failure(lambda: validate_contract(float_schema), "schema must be the integer 1")

    unknown_root = tomllib.loads(
        replace_fixture_once(
            contract_source,
            "\n[scope]\n",
            "\nroot_extension = true\n\n[scope]\n",
        )
    )
    expect_failure(lambda: validate_contract(unknown_root), "unexpected root_extension")

    unknown_scope = tomllib.loads(
        replace_fixture_once(
            contract_source,
            "\n[bounded_adaptation]\n",
            "\ntraining_authority = true\n\n[bounded_adaptation]\n",
        )
    )
    expect_failure(lambda: validate_contract(unknown_scope), "unexpected training_authority")

    unknown_adaptation = tomllib.loads(
        contract_source + "\nrelease_authority = true\n"
    )
    expect_failure(
        lambda: validate_contract(unknown_adaptation),
        "unexpected release_authority",
    )

    widened = copy.deepcopy(contract)
    widened_scope = widened["scope"]
    if not isinstance(widened_scope, dict):
        raise AssertionError("scope fixture is not a table")
    widened_scope["capabilities"] = [*EXPECTED_CAPABILITIES, "train"]
    expect_failure(lambda: validate_contract(widened), "runtime capabilities")

    weakened_adaptation = copy.deepcopy(contract)
    adaptation = weakened_adaptation["bounded_adaptation"]
    if not isinstance(adaptation, dict):
        raise AssertionError("bounded-adaptation fixture is not a table")
    adaptation["requirements"] = ["persistent-output-owner", "rollback"]
    expect_failure(
        lambda: validate_contract(weakened_adaptation),
        "bounded-adaptation admission requirements",
    )

    expect_failure(
        lambda: check_retired_inventory(
            retired, retired_paths, {retired_paths[0]}, set(), set()
        ),
        "retired runtime path",
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        fixture_root = Path(temp_dir)
        dangling_path = fixture_root / retired_paths[0]
        dangling_path.parent.mkdir(parents=True)
        dangling_path.symlink_to(
            fixture_root / "missing-target", target_is_directory=True
        )
        discovered_paths = discover_retired_paths(fixture_root, retired_paths)
        if discovered_paths != {retired_paths[0]}:
            raise AssertionError("retired-path discovery missed a dangling symlink")
        expect_failure(
            lambda: check_retired_inventory(
                retired, retired_paths, discovered_paths, set(), set()
            ),
            "retired runtime path",
        )
    expect_failure(
        lambda: check_retired_inventory(retired, retired_paths, set(), {retired[1]}, set()),
        "retired workspace packages",
    )
    expect_failure(
        lambda: check_retired_inventory(retired, retired_paths, set(), set(), {retired[2]}),
        "retired Cargo.lock packages",
    )

    license_bytes = b"# PolyForm Noncommercial License 1.0.0\nfixture\n"
    cargo = {
        "workspace": {"package": {"license": EXPECTED_LICENSE}},
    }
    fingerprint = {
        "sha256": {"LICENSE": hashlib.sha256(license_bytes).hexdigest()},
    }
    metadata = {"packages": [{"name": "fixture", "license": EXPECTED_LICENSE}]}
    validate_license_derivation(cargo, fingerprint, license_bytes, metadata)

    wrong_fingerprint = copy.deepcopy(fingerprint)
    sha256 = wrong_fingerprint["sha256"]
    if not isinstance(sha256, dict):
        raise AssertionError("fingerprint fixture is not a table")
    sha256["LICENSE"] = "0" * 64
    expect_failure(
        lambda: validate_license_derivation(
            cargo, wrong_fingerprint, license_bytes, metadata
        ),
        "fingerprint does not match",
    )

    wrong_package_license = {
        "packages": [{"name": "fixture", "license": "MIT"}],
    }
    expect_failure(
        lambda: validate_license_derivation(
            cargo, fingerprint, license_bytes, wrong_package_license
        ),
        "do not inherit",
    )

    if "--locked" not in CARGO_METADATA_COMMAND:
        raise AssertionError("cargo metadata command must enforce the committed lockfile")


def main() -> int:
    run_self_tests()
    retired, retired_paths = load_contract()
    metadata = workspace_metadata()
    packages = metadata_packages(metadata)
    package_names = {str(package.get("name")) for package in packages}

    lock = tomllib.loads((ROOT / "Cargo.lock").read_text(encoding="utf-8"))
    lock_packages = lock.get("package")
    if not isinstance(lock_packages, list) or any(
        not isinstance(package, dict) for package in lock_packages
    ):
        fail("Cargo.lock returned an invalid package array")
    locked_names = {package.get("name") for package in lock_packages}
    existing_paths = discover_retired_paths(ROOT, retired_paths)
    check_retired_inventory(
        retired, retired_paths, existing_paths, package_names, locked_names
    )

    check_license_derivation(metadata)
    print("runtime scope structure, locked metadata, and license derivation: ok")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (
        AssertionError,
        json.JSONDecodeError,
        OSError,
        RuntimeError,
        tomllib.TOMLDecodeError,
    ) as error:
        print(f"runtime scope check failed: {error}", file=sys.stderr)
        sys.exit(1)
