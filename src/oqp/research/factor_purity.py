"""Content-level factor-boundary inspection and implementation fingerprints."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import inspect
import json
from pathlib import Path
import textwrap
from types import ModuleType
from typing import Any, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
PURITY_REVIEW_PATH = (
    REPO_ROOT / "departments" / "research" / "factors" / "purity_review.yaml"
)
PURE_FACTOR_PORTFOLIO_LAYERS = {"alpha_score", "predictive_signal"}
FORBIDDEN_FACTOR_OUTPUT_COLUMNS = {
    "active_state",
    "final_target_weight",
    "held_signal",
    "held_target_weight",
    "portfolio_weight",
    "position",
    "positions",
    "target_position",
    "target_weight",
}
FORBIDDEN_FACTOR_PARAMETER_NAMES = {
    "bottom_n",
    "cooldown_ticks",
    "exit_on_zero_failure",
    "exit_rank",
    "gross_leverage",
    "hold_days",
    "hold_ticks",
    "liquidity_quantile",
    "max_abs_close_return",
    "max_contracts",
    "max_gross_leverage",
    "max_hold_days",
    "max_holding_days",
    "max_names",
    "max_weight_per_asset",
    "max_weight_per_contract",
    "max_weight_per_name",
    "max_zero_volume_pct",
    "min_avg_open_interest",
    "min_avg_traded_value",
    "min_avg_volume",
    "position_size",
    "portfolio_vol_target",
    "risk_budget",
    "stop_atr_multiplier",
    "stop_z",
    "target_gross",
    "target_gross_leverage",
    "top_n",
    "weight_per_contract",
}
FORBIDDEN_FACTOR_HELPER_MARKERS = {
    "accepted_state",
    "allocate_capped",
    "build_bucket_state",
    "build_held",
    "build_signed_state",
    "build_ttl_state",
    "cap_daily_gross",
    "equal_weight_active",
    "equal_weight_signed",
    "hold_signal",
    "quality_weight_signed_state",
    "scale_signed_signal",
    "scale_to_gross",
}


@dataclass(frozen=True, slots=True)
class FactorPurityInspection:
    """Static content-level result for one active factor source."""

    factor_id: str
    source: str
    implementation_fingerprint: str
    portfolio_layer: str
    alpha_signal_col: str
    execution_weight_col: str
    allocation_parameters: tuple[str, ...]
    forbidden_output_columns: tuple[str, ...]
    lifecycle_helpers: tuple[str, ...]
    negative_shift_lines: tuple[int, ...]
    issues: tuple[str, ...]

    @property
    def pure(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_id": self.factor_id,
            "source": self.source,
            "implementation_fingerprint": self.implementation_fingerprint,
            "portfolio_layer": self.portfolio_layer,
            "alpha_signal_col": self.alpha_signal_col,
            "execution_weight_col": self.execution_weight_col,
            "allocation_parameters": ";".join(self.allocation_parameters),
            "forbidden_output_columns": ";".join(
                self.forbidden_output_columns
            ),
            "lifecycle_helpers": ";".join(self.lifecycle_helpers),
            "negative_shift_lines": ";".join(
                str(value) for value in self.negative_shift_lines
            ),
            "purity_issues": "; ".join(self.issues),
            "content_pure": self.pure,
        }


def build_factor_purity_review_index(
    active_factor_ids: tuple[str, ...] | list[str],
    *,
    review_path: Path = PURITY_REVIEW_PATH,
) -> dict[str, dict[str, Any]]:
    """Expand the audited review ledger and require complete active-ID coverage."""

    payload = yaml.safe_load(review_path.read_text(encoding="utf-8")) or {}
    index: dict[str, dict[str, Any]] = {}
    for group_name, details in (payload.get("review_groups") or {}).items():
        if not isinstance(details, Mapping):
            continue
        ids = [str(value) for value in details.get("factor_ids", [])]
        if details.get("collection") == "gtja_alpha191":
            ids.extend(
                factor_id
                for factor_id in active_factor_ids
                if _numeric_factor_id(factor_id) in range(120, 311)
            )
            expected_count = int(details.get("expected_count", len(ids)))
            if len(set(ids)) != expected_count:
                raise ValueError(
                    f"{group_name} expected {expected_count} factors, "
                    f"found {len(set(ids))}"
                )
        for factor_id in ids:
            if factor_id in index:
                raise ValueError(f"duplicate purity review for {factor_id}")
            index[factor_id] = {
                "review_group": str(group_name),
                "review_conclusion": str(details.get("conclusion") or ""),
                "extracted_component_ids": tuple(
                    str(value)
                    for value in details.get("extracted_component_ids", [])
                ),
                "lookahead_fix": "",
            }

    for factor_id, details in (payload.get("extractions") or {}).items():
        factor_id = str(factor_id)
        if factor_id in index:
            raise ValueError(
                f"{factor_id} appears in both pure and extraction review groups"
            )
        if not isinstance(details, Mapping):
            details = {}
        index[factor_id] = {
            "review_group": "factor_boundary_extraction",
            "review_conclusion": "pure_after_component_extraction",
            "extracted_component_ids": tuple(
                str(value) for value in details.get("components", [])
            ),
            "lookahead_fix": str(details.get("lookahead_fix") or ""),
        }

    active = set(map(str, active_factor_ids))
    reviewed = set(index)
    missing = sorted(active.difference(reviewed))
    extra = sorted(reviewed.difference(active))
    if missing or extra:
        messages = []
        if missing:
            messages.append("missing active reviews: " + ", ".join(missing))
        if extra:
            messages.append("reviews for inactive IDs: " + ", ".join(extra))
        raise ValueError("; ".join(messages))
    return index


def inspect_factor_source_purity(
    path: Path,
    module: ModuleType | Any,
) -> FactorPurityInspection:
    """Inspect hard portfolio-ownership violations in a factor implementation."""

    source_text = _factor_source_text(path, module)
    tree = ast.parse(source_text, filename=str(path))
    factor_id = str(getattr(module, "FACTOR_ID", path.stem)).strip()
    metadata = getattr(module, "FACTOR_METADATA", {}) or {}
    contract = getattr(module, "FACTOR_CONTRACT", {}) or {}
    portfolio_layer = str(metadata.get("portfolio_layer") or "").strip().lower()
    alpha_signal_col = str(contract.get("alpha_signal_col") or "").strip()
    execution_weight_col = str(
        contract.get("execution_weight_col") or ""
    ).strip()
    parameter_names = _declared_mapping_keys(tree, "FACTOR_PARAMETERS")
    parameter_names.update(_factor_entrypoint_parameter_names(tree))
    declared_parameters = getattr(module, "FACTOR_PARAMETERS", {}) or {}
    if isinstance(declared_parameters, Mapping):
        parameter_names.update(str(value) for value in declared_parameters)
    allocation_parameters = tuple(
        sorted(parameter_names.intersection(FORBIDDEN_FACTOR_PARAMETER_NAMES))
    )
    output_columns = _assigned_column_names(tree)
    forbidden_outputs = tuple(
        sorted(
            column
            for column in output_columns
            if column in FORBIDDEN_FACTOR_OUTPUT_COLUMNS
            or column.endswith("_active_state")
            or column.endswith("_target_weight")
        )
    )
    top_level_functions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    lifecycle_helpers = tuple(
        sorted(
            name
            for name in top_level_functions
            if any(marker in name.lower() for marker in FORBIDDEN_FACTOR_HELPER_MARKERS)
        )
    )
    negative_shift_lines = tuple(sorted(_negative_shift_lines(tree)))
    temporal_holding = _temporal_policy_owns_holding(tree) or bool(
        {
            "holding_mode",
            "holding_periods",
            "holding_unit",
            "zero_signal_action",
        }.intersection(
            (getattr(module, "TEMPORAL_POLICY", {}) or {}).keys()
            if isinstance(getattr(module, "TEMPORAL_POLICY", {}), Mapping)
            else {"holding_mode"}
        )
    )
    issues: list[str] = []
    if portfolio_layer not in PURE_FACTOR_PORTFOLIO_LAYERS:
        issues.append(f"non-predictive portfolio_layer: {portfolio_layer!r}")
    if (
        alpha_signal_col
        and execution_weight_col
        and execution_weight_col != alpha_signal_col
    ):
        issues.append(
            "execution_weight_col differs from alpha_signal_col: "
            f"{execution_weight_col!r} != {alpha_signal_col!r}"
        )
    if allocation_parameters:
        issues.append(
            "allocation/lifecycle parameters remain in factor: "
            + ", ".join(allocation_parameters)
        )
    if forbidden_outputs:
        issues.append(
            "position/holding output columns remain in factor: "
            + ", ".join(forbidden_outputs)
        )
    if lifecycle_helpers:
        issues.append(
            "position/holding helper functions remain in factor: "
            + ", ".join(lifecycle_helpers)
        )
    if temporal_holding:
        issues.append("TEMPORAL_POLICY owns executable holding behavior")
    if negative_shift_lines:
        issues.append(
            "negative shift may consume future rows at line(s): "
            + ", ".join(str(value) for value in negative_shift_lines)
        )
    return FactorPurityInspection(
        factor_id=factor_id,
        source=_portable_source(path),
        implementation_fingerprint=(
            factor_implementation_fingerprint(path)
            if path.is_file()
            else _in_memory_implementation_fingerprint(module, source_text)
        ),
        portfolio_layer=portfolio_layer,
        alpha_signal_col=alpha_signal_col,
        execution_weight_col=execution_weight_col,
        allocation_parameters=allocation_parameters,
        forbidden_output_columns=forbidden_outputs,
        lifecycle_helpers=lifecycle_helpers,
        negative_shift_lines=negative_shift_lines,
        issues=tuple(dict.fromkeys(issues)),
    )


def _factor_entrypoint_parameter_names(tree: ast.AST) -> set[str]:
    """Collect parameters from factor APIs, including preparation hooks.

    The original audit checked only ``FACTOR_PARAMETERS`` mappings. Hybrid
    implementations could therefore hide allocation or sample-wide universe
    rules in keyword-only ``compute`` or ``prepare_data`` arguments.
    """

    entrypoints = {"build_factor", "compute", "compute_factor", "prepare_data"}
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name not in entrypoints:
            continue
        for argument in (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        ):
            if argument.arg not in {"data", "frame", "self"}:
                names.add(argument.arg)
    return names


def factor_implementation_fingerprint(path: Path) -> str:
    """Hash one factor and its recursively imported repository Python helpers."""

    digest = hashlib.sha256()
    for dependency in _local_python_dependencies(path):
        try:
            relative = dependency.resolve().relative_to(REPO_ROOT.resolve())
        except ValueError:
            relative = dependency.resolve()
        digest.update(str(relative).encode("utf-8"))
        digest.update(b"\0")
        digest.update(dependency.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _factor_source_text(path: Path, module: ModuleType | Any) -> str:
    if path.is_file():
        return path.read_text(encoding="utf-8")
    functions: list[str] = []
    for name in ("compute", "compute_factor", "build_factor"):
        function = getattr(module, name, None)
        if not callable(function):
            continue
        try:
            functions.append(textwrap.dedent(inspect.getsource(function)))
        except (OSError, TypeError):
            continue
    return "\n\n".join(functions)


def _in_memory_implementation_fingerprint(
    module: ModuleType | Any,
    source_text: str,
) -> str:
    payload = {
        "factor_id": str(getattr(module, "FACTOR_ID", "")),
        "metadata": getattr(module, "FACTOR_METADATA", {}) or {},
        "contract": getattr(module, "FACTOR_CONTRACT", {}) or {},
        "parameters": getattr(module, "FACTOR_PARAMETERS", {}) or {},
        "source": source_text,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _local_python_dependencies(path: Path) -> tuple[Path, ...]:
    pending = [path.resolve()]
    visited: set[Path] = set()
    while pending:
        current = pending.pop()
        if current in visited or not current.is_file():
            continue
        visited.add(current)
        try:
            tree = ast.parse(current.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for dependency in _resolved_import_paths(tree):
            resolved = dependency.resolve()
            if resolved not in visited:
                pending.append(resolved)
    return tuple(sorted(visited, key=lambda value: str(value)))


def _resolved_import_paths(tree: ast.Module) -> set[Path]:
    paths: set[Path] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                paths.update(_module_candidates(alias.name))
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            paths.update(_module_candidates(node.module))
            for alias in node.names:
                paths.update(_module_candidates(f"{node.module}.{alias.name}"))
    return {path for path in paths if path.is_file()}


def _module_candidates(module_name: str) -> set[Path]:
    module_path = Path(*module_name.split("."))
    candidates = {
        REPO_ROOT / f"{module_path}.py",
        REPO_ROOT / module_path / "__init__.py",
        REPO_ROOT / "src" / f"{module_path}.py",
        REPO_ROOT / "src" / module_path / "__init__.py",
    }
    return candidates


def _declared_mapping_keys(tree: ast.Module, name: str) -> set[str]:
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(
            isinstance(target, ast.Name) and target.id == name
            for target in targets
        ):
            continue
        if not isinstance(node.value, ast.Dict):
            return set()
        return {
            str(key.value)
            for key in node.value.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
    return set()


def _assigned_column_names(tree: ast.Module) -> set[str]:
    columns: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store):
            value = _string_literal(node.slice)
            if value:
                columns.add(value)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "alias" and node.args:
                value = _string_literal(node.args[0])
                if value:
                    columns.add(value)
    return columns


def _negative_shift_lines(tree: ast.Module) -> set[int]:
    lines: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in {"shift", "lead"} or not node.args:
            continue
        first = node.args[0]
        if (
            isinstance(first, ast.UnaryOp)
            and isinstance(first.op, ast.USub)
            and isinstance(first.operand, ast.Constant)
            and isinstance(first.operand.value, (int, float))
            and first.operand.value > 0
        ):
            lines.add(int(getattr(node, "lineno", 0)))
    return lines


def _temporal_policy_owns_holding(tree: ast.Module) -> bool:
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(
            isinstance(target, ast.Name) and target.id == "TEMPORAL_POLICY"
            for target in targets
        ):
            continue
        if not isinstance(node.value, ast.Dict):
            return True
        for key in node.value.keys:
            value = _string_literal(key)
            if value in {
                "holding_mode",
                "holding_periods",
                "holding_unit",
                "zero_signal_action",
            }:
                return True
    return False


def _string_literal(node: ast.AST) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return ""


def _portable_source(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _numeric_factor_id(factor_id: str) -> int:
    try:
        return int(str(factor_id).split("_", 2)[1])
    except (IndexError, TypeError, ValueError):
        return -1


__all__ = [
    "FORBIDDEN_FACTOR_HELPER_MARKERS",
    "FORBIDDEN_FACTOR_OUTPUT_COLUMNS",
    "FORBIDDEN_FACTOR_PARAMETER_NAMES",
    "FactorPurityInspection",
    "PURITY_REVIEW_PATH",
    "build_factor_purity_review_index",
    "factor_implementation_fingerprint",
    "inspect_factor_source_purity",
]
