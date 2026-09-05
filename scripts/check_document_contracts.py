#!/usr/bin/env python3
"""Check authored navigation paths and generated golden-artifact inventory."""

from __future__ import annotations

import ast
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
from typing import Final


ROOT: Final = Path(__file__).resolve().parents[1]
AUTHORED_NAVIGATION_DOCUMENTS: Final = (
    Path("AGENTS.md"),
    Path("README.md"),
    Path("CLAUDE.md"),
    Path("llms.txt"),
    Path("ARCHITECTURE.md"),
    Path("_llm/glossary.md"),
)
GOLDEN_GENERATOR: Final = Path("phases/03-stella/golden/generate.py")
INLINE_CODE: Final = re.compile(r"`([^`]+)`")
MARKDOWN_TARGET: Final = re.compile(r"\]\(([^)]+)\)")
OUTPUT_HEADING: Final = re.compile(r"^Outputs at .+:$")
OUTPUT_BULLET: Final = re.compile(r"^\s*-\s+([A-Za-z0-9_.-]+)\s+—.*$")
WRITE_METHODS: Final = frozenset({"write_text", "write_bytes"})


def fail(message: str) -> None:
    raise RuntimeError(message)


def is_concrete_crate_path(candidate: str) -> bool:
    path = PurePosixPath(candidate.rstrip("/"))
    return (
        candidate.startswith("crates/")
        and len(path.parts) > 1
        and path.parts[0] == "crates"
        and not any(part in {".", ".."} or any(char in part for char in "<>*{}$") for part in path.parts)
    )


def navigation_paths(root: Path, document: Path) -> set[Path]:
    text = document.read_text(encoding="utf-8")
    candidates = [*INLINE_CODE.findall(text), *MARKDOWN_TARGET.findall(text)]
    local_crates = {
        path.name for path in (root / "crates").iterdir() if path.is_dir()
    }
    return {
        Path(candidate.rstrip("/"))
        for candidate in candidates
        if is_concrete_crate_path(candidate)
        and PurePosixPath(candidate.rstrip("/")).parts[1] in local_crates
    }


def check_navigation_paths(root: Path) -> None:
    missing_documents = [
        str(document) for document in AUTHORED_NAVIGATION_DOCUMENTS if not (root / document).is_file()
    ]
    if missing_documents:
        fail("authored navigation document missing: " + ", ".join(missing_documents))

    missing_paths = sorted(
        f"{document}: {path}"
        for document in AUTHORED_NAVIGATION_DOCUMENTS
        for path in navigation_paths(root, root / document)
        if not (root / path).exists()
    )
    if missing_paths:
        fail("authored crate navigation path missing: " + ", ".join(missing_paths))


def static_output_path(expression: ast.expr) -> str | None:
    if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Div):
        if isinstance(expression.left, ast.Name) and expression.left.id == "here":
            if isinstance(expression.right, ast.Constant) and isinstance(expression.right.value, str):
                return expression.right.value
    if (
        isinstance(expression, ast.Call)
        and isinstance(expression.func, ast.Name)
        and expression.func.id == "str"
        and len(expression.args) == 1
    ):
        return static_output_path(expression.args[0])
    return None


def open_is_writable(call: ast.Call) -> bool:
    mode = call.args[0] if call.args else next(
        (keyword.value for keyword in call.keywords if keyword.arg == "mode"), None
    )
    return isinstance(mode, ast.Constant) and isinstance(mode.value, str) and any(
        marker in mode.value for marker in "wax+"
    )


class GeneratorOutputVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.outputs: set[str] = set()

    def visit_Call(self, node: ast.Call) -> None:
        output_expression: ast.expr | None = None
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in WRITE_METHODS:
                output_expression = node.func.value
            elif node.func.attr == "open" and open_is_writable(node):
                output_expression = node.func.value
            elif node.func.attr == "save_file":
                if len(node.args) < 2:
                    fail("generator save_file call has no output path")
                output_expression = node.args[1]
        if output_expression is not None:
            output = static_output_path(output_expression)
            if output is None:
                fail("generator output path must be a static `here / filename` expression")
            path = PurePosixPath(output)
            if path.name != output or path.name == "":
                fail(f"generator output must be a filename, got {output!r}")
            self.outputs.add(output)
        self.generic_visit(node)


def generator_tree(generator: Path) -> ast.Module:
    try:
        return ast.parse(generator.read_text(encoding="utf-8"), filename=str(generator))
    except SyntaxError as error:
        fail(f"generator source is not valid Python: {error}")


def generator_outputs(tree: ast.Module) -> list[str]:
    visitor = GeneratorOutputVisitor()
    visitor.visit(tree)
    if not visitor.outputs:
        fail("generator output inventory is empty")
    return sorted(visitor.outputs)


def documented_outputs(tree: ast.Module) -> list[str]:
    docstring = ast.get_docstring(tree)
    if docstring is None:
        fail("generator has no output-inventory docstring")
    lines = docstring.splitlines()
    try:
        heading = next(index for index, line in enumerate(lines) if OUTPUT_HEADING.fullmatch(line))
    except StopIteration:
        fail("generator has no output-inventory heading")
    outputs = []
    for line in lines[heading + 1 :]:
        if not line.strip():
            continue
        match = OUTPUT_BULLET.fullmatch(line)
        if match is None:
            break
        outputs.append(match.group(1))
    outputs.sort()
    if not outputs:
        fail("generator output-inventory docstring has no filename bullets")
    if len(outputs) != len(set(outputs)):
        fail("generator output-inventory docstring has duplicate filenames")
    return outputs


def check_golden_generator_inventory(root: Path) -> None:
    generator = root / GOLDEN_GENERATOR
    if not generator.is_file():
        fail(f"golden generator missing: {GOLDEN_GENERATOR}")
    tree = generator_tree(generator)
    actual_outputs = generator_outputs(tree)
    declared_outputs = documented_outputs(tree)
    if declared_outputs != actual_outputs:
        fail(
            "generator docstring outputs do not match AST write targets: "
            f"documented {declared_outputs}, actual {actual_outputs}"
        )


def expect_failure(action: object, expected: str) -> None:
    if not callable(action):
        raise AssertionError("self-test action must be callable")
    try:
        action()
    except RuntimeError as error:
        if expected not in str(error):
            raise AssertionError(f"expected {expected!r}, got {error!s}") from error
    else:
        raise AssertionError(f"expected failure containing {expected!r}")


def write_navigation_fixture(root: Path, missing_path: bool = False) -> None:
    for document in AUTHORED_NAVIGATION_DOCUMENTS:
        (root / document).parent.mkdir(parents=True, exist_ok=True)
        (root / document).write_text("", encoding="utf-8")
    (root / "crates/core").mkdir(parents=True, exist_ok=True)
    (root / "AGENTS.md").write_text("`crates/core`\n", encoding="utf-8")
    if missing_path:
        (root / "llms.txt").write_text("`crates/core/missing`\n", encoding="utf-8")


def write_generator_fixture(root: Path, documented_provenance: str) -> None:
    golden = root / GOLDEN_GENERATOR.parent
    golden.mkdir(parents=True, exist_ok=True)
    (golden / GOLDEN_GENERATOR.name).write_text(
        f'''"""Outputs at `fixtures/`:
    - tokens.jsonl — token records
    - embeddings_dim1024.safetensors — embedding matrix
    - {documented_provenance} — generator receipt
"""

from pathlib import Path

here = Path(__file__).parent
(here / \"tokens.jsonl\").open(\"w\")
safetensors.torch.save_file({{}}, str(here / \"embeddings_dim1024.safetensors\"))
(here / \"PROVENANCE.json\").write_text(\"{{}}\")
''',
        encoding="utf-8",
    )


def run_self_tests() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        write_navigation_fixture(root)
        check_navigation_paths(root)
        write_navigation_fixture(root, missing_path=True)
        expect_failure(lambda: check_navigation_paths(root), "llms.txt: crates/core/missing")

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        write_generator_fixture(root, "PROVENANCE.json")
        check_golden_generator_inventory(root)
        write_generator_fixture(root, "PROVENANCE.md")
        expect_failure(
            lambda: check_golden_generator_inventory(root),
            "docstring outputs do not match AST write targets",
        )


def main() -> int:
    run_self_tests()
    check_navigation_paths(ROOT)
    check_golden_generator_inventory(ROOT)
    print("authored crate navigation and generator output inventory: ok")
    return 0


def self_test_main() -> int:
    run_self_tests()
    print("document contract self-tests: ok")
    return 0


if __name__ == "__main__":
    try:
        if sys.argv[1:] == ["--self-test"]:
            sys.exit(self_test_main())
        if len(sys.argv) != 1:
            fail("usage: check_document_contracts.py [--self-test]")
        sys.exit(main())
    except (AssertionError, OSError, RuntimeError) as error:
        print(f"document contract check failed: {error}", file=sys.stderr)
        sys.exit(1)
