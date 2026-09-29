"""Load a simulation case from YAML.

    grid:        geometry (see build_grid)
    properties:  per-cell property definitions (see resim.props)
    edits:       optional list of box edits
    preprocess:  optional {minpv: m³}
    simulation:  optional single-phase run (see run_simulation)

Units are metric: m, mD, bar, cP, day, m³/day, 1/bar, kg/m³.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from resim.grid import Grid, build_cartesian
from resim.preproc import Domain, build_domain
from resim.props import PropertySet, build_properties
from resim.props.builder import normalize_spec
from resim.props.expr import evaluate
from resim.props.grdecl import read_values
from resim.props.sources import SOURCES, BuildContext
from resim.solvers import Fluid, simulate_transient, solve_steady
from resim.units import BAR, CENTIPOISE, DAY


@dataclass
class Case:
    config: dict
    grid: Grid
    props: PropertySet
    domain: Domain


def _tops(spec, dims, base_dir: Path):
    """TOPS as a number, list, ``{expr: ...}`` over map indices I, J, or ``{file: path}``."""
    if not isinstance(spec, dict):
        return spec
    nx, ny = int(dims[0]), int(dims[1])
    if "expr" in spec:
        g = np.arange(nx * ny)
        variables = {"I": g % nx + 1, "J": g // nx + 1}
        return np.broadcast_to(np.asarray(evaluate(spec["expr"], variables), dtype=float), (nx * ny,))
    if "file" in spec:
        return read_values(base_dir / spec["file"], spec.get("keyword"))
    raise ValueError("grid.tops must be a number, a list, {expr: ...} or {file: ...}")


def build_grid(cfg: dict, base_dir: Path) -> Grid:
    cfg = dict(cfg)
    kind = cfg.pop("type", "cartesian")
    if kind != "cartesian":
        raise ValueError(f"grid type {kind!r} is not supported yet (cartesian)")
    for key in ("dims", "dx", "dy", "dz"):
        if key not in cfg:
            raise ValueError(f"grid: missing {key!r}")
    geometry = dict(
        dims=cfg["dims"], dx=cfg["dx"], dy=cfg["dy"], dz=cfg["dz"], tops=_tops(cfg.get("tops", 0.0), cfg["dims"], base_dir)
    )
    actnum_spec = cfg.get("actnum")
    if actnum_spec is None or actnum_spec == "all":
        return build_cartesian(**geometry)
    # ACTNUM may use any property source (file, expr, layers, ...), evaluated on the full grid.
    full = build_cartesian(**geometry)
    spec = normalize_spec(actnum_spec)
    actnum = SOURCES[spec["source"]](spec, BuildContext(grid=full, base_dir=base_dir))
    return build_cartesian(**geometry, actnum=actnum)


def load_case(path) -> Case:
    path = Path(path)
    config = yaml.safe_load(path.read_text())
    return build_case(config, base_dir=path.parent)


def build_case(config: dict, base_dir: str | Path = ".") -> Case:
    base_dir = Path(base_dir)
    for section in ("grid", "properties"):
        if section not in config:
            raise ValueError(f"case is missing the {section!r} section")
    grid = build_grid(config["grid"], base_dir)
    props = build_properties(grid, config["properties"], config.get("edits"), base_dir)
    domain = build_domain(grid, props, minpv=config.get("preprocess", {}).get("minpv", 0.0))
    return Case(config=config, grid=grid, props=props, domain=domain)


def run_simulation(case: Case):
    """Run the ``simulation`` section of a case (single-phase for now).

    simulation:
      type: steady | transient
      fluid: {viscosity: cP, density: kg/m³, compressibility: 1/bar}
      rock_compressibility: 1/bar
      boundary: {xmin: bar, xmax: bar, ...}
      sources: [{cell: [i, j, k], rate: m³/day}]    # positive = injection
      initial_pressure: bar                          # transient only
      timestep: day
      steps: int
    """
    sim = case.config.get("simulation")
    if not sim:
        raise ValueError("case has no 'simulation' section")
    f = sim.get("fluid", {})
    fluid = Fluid(
        viscosity=f.get("viscosity", 1.0) * CENTIPOISE,
        density=f.get("density", 1000.0),
        compressibility=f.get("compressibility", 0.0) / BAR,
    )
    bc = {tag: p * BAR for tag, p in (sim.get("boundary") or {}).items()}
    q = np.zeros(case.domain.n_cells)
    for s in sim.get("sources") or []:
        q[case.domain.cell_index(*s["cell"])] += s["rate"] / DAY

    kind = sim.get("type", "steady")
    if kind == "steady":
        return solve_steady(case.domain, fluid, bc=bc, sources=q)
    if kind == "transient":
        return simulate_transient(
            case.domain,
            fluid,
            p0=sim["initial_pressure"] * BAR,
            dt=sim["timestep"] * DAY,
            n_steps=int(sim["steps"]),
            bc=bc,
            sources=q,
            rock_compressibility=sim.get("rock_compressibility", 0.0) / BAR,
        )
    raise ValueError(f"unknown simulation type {kind!r} (steady, transient)")
