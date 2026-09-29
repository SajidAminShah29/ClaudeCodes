"""Property sources: each turns a YAML spec into a global cell array.

A source is registered with ``@source(name)`` and has the signature
``fn(spec, ctx) -> np.ndarray`` (length nx*ny*nz, Eclipse order). Sources may
declare which other properties they need via ``dependencies(spec)`` so the
builder can evaluate them in the right order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from resim.grid import Grid, global_to_ijk
from resim.props import expr as expression
from resim.props import geostat, spe10
from resim.props.grdecl import read_values

SOURCES: dict[str, Callable] = {}
_DEPENDENCIES: dict[str, Callable] = {}


def source(name: str, depends: Callable | None = None):
    def register(fn):
        SOURCES[name] = fn
        _DEPENDENCIES[name] = depends or (lambda spec: set())
        return fn

    return register


def dependencies(spec: dict) -> set[str]:
    return set(_DEPENDENCIES[spec["source"]](spec))


@dataclass
class BuildContext:
    grid: Grid
    base_dir: Path = Path(".")
    fields: dict[str, np.ndarray] = field(default_factory=dict)
    _cache: dict = field(default_factory=dict)

    @property
    def dims(self) -> tuple[int, int, int]:
        return self.grid.dims

    @property
    def n(self) -> int:
        return self.grid.n_global

    def ijk1(self) -> np.ndarray:
        """1-based (I, J, K) for every global cell, shape (n, 3)."""
        if "ijk" not in self._cache:
            self._cache["ijk"] = global_to_ijk(np.arange(self.n), self.dims) + 1
        return self._cache["ijk"]

    def spacing(self) -> tuple[float, float, float]:
        """Mean cell size of the active grid, used for geostatistics."""
        return tuple(self.grid.cell_dxyz.mean(axis=0))

    def path(self, p) -> Path:
        p = Path(p)
        return p if p.is_absolute() else self.base_dir / p

    def cached(self, key, loader):
        if key not in self._cache:
            self._cache[key] = loader()
        return self._cache[key]

    def variables(self) -> dict[str, np.ndarray]:
        ijk = self.ijk1()
        return {**self.fields, "I": ijk[:, 0], "J": ijk[:, 1], "K": ijk[:, 2]}


def _require(spec: dict, *keys):
    missing = [k for k in keys if k not in spec]
    if missing:
        raise ValueError(f"source {spec['source']!r} requires {', '.join(missing)}")


def _length(values: np.ndarray, n: int, what: str) -> np.ndarray:
    values = np.asarray(values).ravel()
    if values.size != n:
        raise ValueError(f"{what}: got {values.size} values, expected {n}")
    return values


@source("constant")
def _constant(spec, ctx):
    _require(spec, "value")
    return np.full(ctx.n, spec["value"], dtype=float)


@source("layers")
def _layers(spec, ctx):
    _require(spec, "values")
    values = np.asarray(spec["values"], dtype=float)
    if values.size != ctx.dims[2]:
        raise ValueError(f"layers: got {values.size} values, expected nz={ctx.dims[2]}")
    return values[ctx.ijk1()[:, 2] - 1]


@source("array")
def _array(spec, ctx):
    _require(spec, "values")
    return _length(np.asarray(spec["values"], dtype=float), ctx.n, "array")


@source("file")
def _file(spec, ctx):
    _require(spec, "path")
    return _length(read_values(ctx.path(spec["path"]), spec.get("keyword")), ctx.n, f"file {spec['path']}")


@source("copy", depends=lambda spec: {spec["from"]})
def _copy(spec, ctx):
    return ctx.fields[spec["from"]].copy()


@source("expr", depends=lambda spec: expression.expression_names(spec["expr"]) - {"I", "J", "K"})
def _expr(spec, ctx):
    result = expression.evaluate(spec["expr"], ctx.variables())
    return np.broadcast_to(np.asarray(result, dtype=float), (ctx.n,)).copy()


def _gaussian(spec, ctx) -> np.ndarray:
    vario = spec.get("variogram", {})
    ranges = vario.get("range", 1.0)
    y = geostat.gaussian_random_field(
        ctx.dims,
        ctx.spacing(),
        kind=vario.get("type", "spherical"),
        ranges=ranges,
        azimuth=vario.get("azimuth", 0.0),
        seed=spec.get("seed"),
    )
    return y.ravel(order="F")


@source("gaussian_field")
def _gaussian_field(spec, ctx):
    _require(spec, "mean", "std")
    values = geostat.transform(_gaussian(spec, ctx), spec["mean"], spec["std"], spec.get("transform", "normal"))
    if "clip" in spec:
        values = np.clip(values, *spec["clip"])
    return values


@source("truncated_gaussian")
def _truncated_gaussian(spec, ctx):
    _require(spec, "proportions")
    return geostat.truncate(_gaussian(spec, ctx), spec["proportions"], spec.get("codes"))


def _poro_perm_deps(spec):
    return {spec["from"]} | ({spec["by"]} if "by" in spec else set())


@source("poro_perm", depends=_poro_perm_deps)
def _poro_perm(spec, ctx):
    """log10(k) = a * phi + b, optionally with (a, b) per region / facies."""
    phi = ctx.fields[spec["from"]]
    if "by" not in spec:
        _require(spec, "a", "b")
        return 10.0 ** (spec["a"] * phi + spec["b"])
    _require(spec, "params")
    region = ctx.fields[spec["by"]]
    out = np.full(ctx.n, np.nan)
    for code, p in spec["params"].items():
        mask = region == int(code)
        out[mask] = 10.0 ** (p["a"] * phi[mask] + p["b"])
    missing = np.unique(region[np.isnan(out)])
    if missing.size:
        raise ValueError(f"poro_perm: no (a, b) given for {spec['by']} values {missing.tolist()}")
    return out


@source("spe10")
def _spe10(spec, ctx):
    _require(spec, "path", "component")
    path = ctx.path(spec["path"])
    data = ctx.cached(("spe10", str(path)), lambda: spe10.read_spe10_file(path))
    return spe10.extract(data, spec["component"], ctx.dims, spec.get("origin", (1, 1, 1)))
