# resim — a reservoir simulator

A reservoir simulator written in Python for clarity and correctness first.
This first milestone covers the **grid → properties → domain** pipeline and a
single-phase flow solver used to validate it.

```
 case.yaml
     │
     ▼
 GridBuilder ──► Grid ──► PropertyBuilder ──► PropertySet ──► Preprocessor ──► Domain ──► solver
 (Cartesian)    cells,    PORO, PERM, NTG,    global arrays    pore volumes,     graph of cells
                faces     regions, MULT*      (input units)    transmissibility  and connections (SI)
```

The solver only ever sees the `Domain`: pore volumes, depths, and a list of
connections with transmissibilities. New grid types (corner-point, later)
only need to produce a `Grid`; nothing downstream changes.

## Quick start

```bash
pip install -e .[dev]
pytest
resim info examples/layered_steady.yaml
resim run  examples/channelized_transient.yaml --vtk model.vtk   # open in ParaView
```

## Layout

```
resim/
  grid/      grid.py (Grid data structure), cartesian.py (DX/DY/DZ/TOPS builder)
  props/     builder.py (pipeline), sources.py (constant, layers, file, expr, geostat, ...),
             expr.py (safe expressions), geostat.py (Gaussian fields), grdecl.py, spe10.py
  preproc/   domain.py (pore volume, Domain), transmissibility.py (TPFA)
  solvers/   single_phase.py (steady + transient)
  io/        case.py (YAML), vtk.py (ParaView output)
  cli.py
```

## Case file format

Units: **m, mD, bar, cP, day, m³/day, 1/bar, kg/m³**. The core converts to SI once.

### Grid

```yaml
grid:
  type: cartesian
  dims: [nx, ny, nz]
  dx: 20.0                  # scalar, per-column list (nx), or full array (nx*ny*nz)
  dy: 20.0                  # scalar, per-row list (ny), or full array
  dz: [2, 2, 3]             # scalar, per-layer list (nz), or full array
  tops: 2500.0              # scalar | nx*ny map | {expr: "2000 + 0.05*(I-10)**2"} | {file: tops.grdecl}
  actnum: {expr: "(I + J) > 2"}   # optional; any property source, nonzero = active
```

Cell ordering follows Eclipse: `i` fastest, then `j`, then `k`. Depth is
positive downwards. Structure and dip come from `tops`.

### Properties

Each property is defined by a **source**. Properties may reference each
other in any order; they are evaluated in dependency order.

| Source | Example |
|---|---|
| constant | `PORO: 0.2` |
| layers | `PERMX: {source: layers, values: [100, 5, 300]}` |
| array | `PORO: {source: array, values: [...]}` (nx*ny*nz values) |
| file | `PORO: {source: file, path: poro.grdecl, keyword: PORO}` (GRDECL, plain text or `.npy`) |
| copy | `PERMY: {copy: PERMX}` |
| expr | `PERMZ: {expr: "0.1 * PERMX"}`, `SATNUM: {expr: "where(K <= 3, 1, 2)"}` |
| gaussian_field | `{source: gaussian_field, mean: 0.2, std: 0.04, transform: normal, clip: [0.01, 0.35], variogram: {type: spherical, range: [500, 200, 5], azimuth: 30}, seed: 1}` |
| truncated_gaussian | `{source: truncated_gaussian, proportions: [0.7, 0.3], variogram: {...}, seed: 2}` → codes 1..n |
| poro_perm | `{source: poro_perm, from: PORO, a: 12, b: -1}` (log10 k = a·φ + b), or per facies with `by: FACIES, params: {1: {a:, b:}, 2: {...}}` |
| spe10 | `{source: spe10, path: spe_perm.dat, component: permx, origin: [1, 1, 35]}` |

Expressions may use any property, the 1-based indices `I`, `J`, `K`, the
operators `+ - * / ** < <= > >= == != & |` and the functions `exp, log,
log10, sqrt, abs, minimum, maximum, where, clip`.

Variogram `range` is the practical range in metres: `[major, minor, vertical]`,
with `azimuth` giving the major direction in degrees clockwise from +y.
Geostatistical fields are unconditional (not yet tied to well data).

**Required:** `PORO`, `PERMX`. **Defaults:** `PERMY = PERMZ = PERMX`,
`NTG = 1`, `MULTX/Y/Z = MULTPV = 1`, `SATNUM = PVTNUM = EQLNUM = FIPNUM = 1`.
Custom names (e.g. `FACIES`) are allowed and can be used by other properties.

### Edits

Applied in order after all properties are built, and **before** defaults.
So a `PERMX` edit carries over into a defaulted `PERMY`.

```yaml
edits:
  - {op: multiply, field: PERMX, value: 0.01, box: [1, 50, 1, 1, 5, 5]}   # [i1,i2,j1,j2,k1,k2], 1-based
  - {op: equals,   field: MULTZ, value: 0.0,  box: [1, 50, 1, 1, 4, 4]}
  - {op: copy,     field: PERMZ, from: PERMX}
  - {op: add,      field: PORO,  value: 0.01}
```

### Preprocessing

```yaml
preprocess:
  minpv: 1.0     # m³; cells at or below this pore volume are removed (zero-PV cells always are)
```

- Pore volume = V · PORO · NTG · MULTPV
- TPFA: `t_i = A·|c·K n|/|c|²`, `T = MULT / (1/t_a + 1/t_b)`
- NTG scales horizontal permeability only; `MULTX` of cell (i,j,k) scales its +X face (Eclipse convention).

### Simulation (single-phase)

```yaml
simulation:
  type: steady | transient
  fluid: {viscosity: 1.0, density: 1000.0, compressibility: 1.0e-4}
  rock_compressibility: 5.0e-5
  boundary: {xmin: 280.0, xmax: 250.0}       # fixed pressure on model sides, bar
  sources: [{cell: [i, j, k], rate: 100.0}]  # m³/day, + injection / − production
  initial_pressure: 250.0                    # transient only
  timestep: 5.0
  steps: 20
```

Sources are cell-rate terms. Proper wells (Peaceman well index, BHP control)
come in a later phase.

## Validation

`pytest` checks each stage against exact answers:

- **Grid:** volumes, face counts, Eclipse ordering, centroids, outward boundary normals, ACTNUM.
- **Properties:** every source, dependency ordering and cycle detection, box edits, defaults,
  validation that reports the offending (i, j, k), GRDECL parsing, expression sandboxing.
- **Geostatistics:** mean, variance and correlation at several lags match the variogram;
  anisotropy/azimuth; facies proportions; seeding makes results reproducible.
- **Transmissibility:** `kA/Δx`, series harmonic averaging, layered vertical flow, NTG, multipliers, MINPV.
- **Flow:** Darcy linear flow, heterogeneous series flow, hydrostatic equilibrium (no flow),
  source/boundary balance, closed-system mass conservation to 1e-10, transient → steady state.

## Roadmap

1. ~~Cartesian grid, property pipeline, TPFA, single-phase validation~~ (this milestone)
2. Two-phase oil–water, Buckley–Leverett validation
3. Fully implicit Newton with automatic differentiation, gravity and capillarity
4. Peaceman wells and hydrostatic equilibration
5. Black-oil (SPE1)
6. Corner-point grids (GRDECL `COORD`/`ZCORN`), faults and NNCs
7. CPR-AMG linear solver and performance (SPE10 Model 2)
