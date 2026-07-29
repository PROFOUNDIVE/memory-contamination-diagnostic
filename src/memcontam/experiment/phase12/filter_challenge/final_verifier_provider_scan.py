from __future__ import annotations

import ast


FORBIDDEN_PROVIDER_TARGETS = frozenset(
    {
        "memcontam.clients.factory.build_llm_client",
        "memcontam.clients.openai_compatible.OpenAICompatibleClient",
        "memcontam.clients.openai_responses.OpenAIResponsesClient",
    }
)
_SHORT_TARGETS = {target.rsplit(".", 1)[-1]: target for target in FORBIDDEN_PROVIDER_TARGETS}


def provider_construction_found(tree: ast.Module) -> bool:
    scanner = _ProviderConstructionScanner()
    scanner.visit(tree)
    return scanner.found


class _ProviderConstructionScanner(ast.NodeVisitor):
    def __init__(self) -> None:
        self._scopes: list[dict[str, str]] = [{}]
        self.found = False

    @property
    def _aliases(self) -> dict[str, str]:
        return self._scopes[-1]

    def visit_Import(self, node: ast.Import) -> None:
        for item in node.names:
            self._aliases[item.asname or item.name.split(".")[0]] = item.name

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module is None:
            return
        for item in node.names:
            self._aliases[item.asname or item.name] = f"{node.module}.{item.name}"

    def visit_Assign(self, node: ast.Assign) -> None:
        self._bind(node.targets, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self._bind((node.target,), node.value)
        self.generic_visit(node)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self._bind((node.target,), node.value)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_nested(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_nested(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_nested(node)

    def visit_Call(self, node: ast.Call) -> None:
        if self._resolved_name(node.func) in FORBIDDEN_PROVIDER_TARGETS:
            self.found = True
        self.generic_visit(node)

    def _visit_nested(self, node: ast.AST) -> None:
        self._scopes.append(self._aliases.copy())
        self.generic_visit(node)
        self._scopes.pop()

    def _bind(self, targets: list[ast.expr] | tuple[ast.expr, ...], value: ast.expr) -> None:
        resolved = self._resolved_name(value)
        if resolved is None:
            return
        for target in targets:
            match target:
                case ast.Name(id=name):
                    self._aliases[name] = resolved
                case ast.Attribute(value=ast.Name(id="self"), attr=attribute):
                    self._aliases[f"self.{attribute}"] = resolved
                case _:
                    continue

    def _resolved_name(self, node: ast.expr) -> str | None:
        match node:
            case ast.Name(id=name):
                return self._aliases.get(name, _SHORT_TARGETS.get(name))
            case ast.Attribute(value=ast.Name(id="self"), attr=attribute):
                return self._aliases.get(f"self.{attribute}", self._aliases.get(attribute))
            case ast.Attribute(value=value, attr=attribute):
                base = self._resolved_name(value)
                return f"{base}.{attribute}" if base is not None else None
            case ast.Call(func=ast.Name(id="getattr"), args=(base, ast.Constant(value=attribute))):
                resolved = self._resolved_name(base)
                return f"{resolved}.{attribute}" if isinstance(attribute, str) and resolved is not None else None
            case _:
                return None
