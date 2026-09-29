"""Build the property arrays of a model from a declarative specification.

Pipeline:
    1. Evaluate every field spec, in dependency order, on the global grid.
    2. Apply edits (equals / multiply / add / copy within optional boxes).
    3. Fill defaults for anything not given (PERMY = PERMX, NTG = 1, ...).
    4. Validate the values on active cells.

Property values stay in input units (permeability in mD); conversion to SI
happens in the preprocessor.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from resim.grid import Grid, global_to_ijk
from resim.props.sources import SOURCES, BuildContext, dependencies

REQUIRED = ("PORO", "PERMX")
INTEGER_KEYWORDS = {"SATNUM", "PVTNUM", "EQLNUM", "FIPNUM"}
DEFAULTS = {
    "PERMY": "PERMX",
    "PERMZ": "PERMX",
    "NTG": 1.0,
    "MULTX": 1.0,
    "MULTY": 1.0,
    "MULTZ": 1.0,
    "MULTPV": 1.0,
    "SATNUM": 1,
    "PVTNUM": 1,
    "EQLNUM": 1,
    "FIPNUM": 1,
}


class PropertySet:
    """Named per-cell arrays on the global grid (Eclipse order)."""

    def __init__(self, grid: Grid, fields: dict[str, np.ndarray]):
        self.grid = grid
        self.fields = fields

    def __getitem__(self, name: str) -> np.ndarray:
        return self.fields[name]

    def __contains__(self, name: str) -> bool:
        return name in self.fields

    @property
    def names(self) -> list[str]:
        return list(self.fields)

    def active(self, name: str) -> np.ndarray:
        """Values of ``name`` on active cells only."""
        return self.fields[name][self.grid.cell_global]

    def summary(self) -> str:
        lines = [f"{'property':<10}{'min':>12}{'mean':>12}{'max':>12}"]
        for name in self.names:
            v = self.active(name)
            lines.append(f"{name:<10}{v.min():>12.4g}{v.mean():>12.4g}{v.max():>12.4g}")
        return "\n".join(lines)


def normalize_spec(value) -> dict:
    """Accept shorthand forms: a scalar, ``{copy: X}`` or ``{expr: ...}``."""
    if isinstance(value, (int, float)):
        return {"source": "constant", "value": value}
    if not isinstance(value, dict):
        raise ValueError(f"property spec must be a number or a mapping, got {value!r}")
    spec = dict(value)
    if "source" not in spec:
        for shorthand in ("copy", "expr"):
            if shorthand in spec:
                spec["source"] = shorthand
                if shorthand == "copy":
                    spec["from"] = spec.pop("copy")
                break
        else:
            raise ValueError(f"property spec {value!r} has no 'source'")
    if spec["source"] not in SOURCES:
        raise ValueError(f"unknown property source {spec['source']!r}; available: {sorted(SOURCES)}")
    return spec


def _order(specs: dict[str, dict]) -> list[str]:
    """Topologically sort field names by their dependencies."""
    order, state = [], {}

    def visit(name, chain):
        if state.get(name) == "done":
            return
        if state.get(name) == "visiting":
            raise ValueError(f"circular property dependency: {' -> '.join(chain + [name])}")
        if name not in specs:
            raise ValueError(f"property {chain[-1]!r} depends on undefined property {name!r}")
        state[name] = "visiting"
        for dep in sorted(dependencies(specs[name])):
            visit(dep, chain + [name])
        state[name] = "done"
        order.append(name)

    for name in specs:
        visit(name, [])
    return order


def _box_mask(box, ctx: BuildContext) -> np.ndarray:
    if box is None:
        return np.ones(ctx.n, dtype=bool)
    if len(box) != 6:
        raise ValueError(f"box must be [i1, i2, j1, j2, k1, k2] (1-based, inclusive), got {box}")
    ijk = ctx.ijk1()
    mask = np.ones(ctx.n, dtype=bool)
    for axis in range(3):
        lo, hi = box[2 * axis], box[2 * axis + 1]
        if not 1 <= lo <= hi <= ctx.dims[axis]:
            raise ValueError(f"box {box} is outside the grid {ctx.dims}")
        mask &= (ijk[:, axis] >= lo) & (ijk[:, axis] <= hi)
    return mask


def _default(name: str, fields: dict) -> np.ndarray | None:
    if name not in DEFAULTS:
        return None
    default = DEFAULTS[name]
    if isinstance(default, str):
        return fields[default].copy() if default in fields else None
    n = next(iter(fields.values())).size
    return np.full(n, default, dtype=float)


def apply_edits(fields: dict[str, np.ndarray], edits: list[dict], ctx: BuildContext) -> None:
    for edit in edits:
        op, name = edit.get("op"), edit.get("field")
        if op not in ("equals", "multiply", "add", "copy") or not name:
            raise ValueError(f"invalid edit {edit!r}: need op in equals/multiply/add/copy and a field")
        mask = _box_mask(edit.get("box"), ctx)
        if name not in fields:
            fields[name] = _default(name, fields)
            if fields[name] is None:
                if op not in ("equals", "copy"):
                    raise ValueError(f"edit {op} on undefined property {name!r}")
                fields[name] = np.zeros(ctx.n)
        target = fields[name]
        if op == "copy":
            if edit.get("from") not in fields:
                raise ValueError(f"edit copy: source property {edit.get('from')!r} is undefined")
            target[mask] = fields[edit["from"]][mask]
        elif "value" not in edit:
            raise ValueError(f"edit {op} on {name} requires 'value'")
        elif op == "equals":
            target[mask] = edit["value"]
        elif op == "multiply":
            target[mask] *= edit["value"]
        else:
            target[mask] += edit["value"]


def validate(props: PropertySet) -> None:
    """Check physical bounds on active cells; report offending (i, j, k)."""
    grid = props.grid
    checks = {
        "PORO": (0.0, 1.0),
        "NTG": (0.0, 1.0),
        "PERMX": (0.0, np.inf),
        "PERMY": (0.0, np.inf),
        "PERMZ": (0.0, np.inf),
        "MULTX": (0.0, np.inf),
        "MULTY": (0.0, np.inf),
        "MULTZ": (0.0, np.inf),
        "MULTPV": (0.0, np.inf),
    }
    checks.update({k: (1, np.inf) for k in INTEGER_KEYWORDS})
    errors = []
    for name in props.names:
        v = props.active(name)
        lo, hi = checks.get(name, (-np.inf, np.inf))
        bad = ~np.isfinite(v) | (v < lo) | (v > hi)
        if bad.any():
            where = global_to_ijk(grid.cell_global[bad][:5], grid.dims) + 1
            cells = ", ".join(f"({i},{j},{k})={val:.4g}" for (i, j, k), val in zip(where, v[bad][:5]))
            errors.append(f"{name}: {int(bad.sum())} active cells outside [{lo}, {hi}], e.g. {cells}")
    if errors:
        raise ValueError("invalid properties:\n  " + "\n  ".join(errors))


def build_properties(
    grid: Grid,
    spec: dict,
    edits: list[dict] | None = None,
    base_dir: str | Path = ".",
) -> PropertySet:
    specs = {name.upper(): normalize_spec(value) for name, value in (spec or {}).items()}
    missing = [r for r in REQUIRED if r not in specs]
    if missing:
        raise ValueError(f"required properties missing: {', '.join(missing)}")

    ctx = BuildContext(grid=grid, base_dir=Path(base_dir))
    for name in _order(specs):
        values = np.asarray(SOURCES[specs[name]["source"]](specs[name], ctx))
        if values.shape != (ctx.n,):
            raise ValueError(f"{name}: source produced shape {values.shape}, expected ({ctx.n},)")
        ctx.fields[name] = values.astype(float)

    apply_edits(ctx.fields, [{**e, "field": e.get("field", "").upper()} for e in edits or []], ctx)

    for name in DEFAULTS:
        if name not in ctx.fields:
            ctx.fields[name] = _default(name, ctx.fields)

    for name in INTEGER_KEYWORDS:
        v = ctx.fields[name]
        if not np.all(v[grid.cell_global] == np.round(v[grid.cell_global])):
            raise ValueError(f"{name} must hold integer values")
        ctx.fields[name] = np.round(v).astype(np.int64)

    props = PropertySet(grid, ctx.fields)
    validate(props)
    return props
