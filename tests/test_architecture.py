import ast
from importlib.util import resolve_name
from pathlib import Path


PACKAGE_ROOT = Path(__file__).parents[1] / "src" / "rag_eval"


def _resolved_imports(package: str) -> set[str]:
    imports: set[str] = set()
    for path in (PACKAGE_ROOT / package).glob("*.py"):
        module = f"rag_eval.{package}.{path.stem}"
        current_package = module if path.stem == "__init__" else module.rpartition(".")[0]
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                target = "." * node.level + (node.module or "")
                imports.add(
                    resolve_name(target, current_package)
                    if node.level
                    else target
                )
    return imports


def test_component_packages_do_not_import_end_to_end():
    for package in ("generation", "retrieval"):
        forbidden = sorted(
            name
            for name in _resolved_imports(package)
            if name.startswith("rag_eval.end_to_end")
        )
        assert forbidden == []


def test_retrieval_does_not_import_generation_metrics():
    assert "rag_eval.generation.metrics" not in _resolved_imports("retrieval")


def test_shared_evaluation_does_not_import_domain_packages():
    domain_prefixes = (
        "rag_eval.generation",
        "rag_eval.retrieval",
        "rag_eval.end_to_end",
        "rag_eval.semantic",
    )
    forbidden = sorted(
        name
        for name in _resolved_imports("evaluation")
        if name.startswith(domain_prefixes)
    )
    assert forbidden == []
