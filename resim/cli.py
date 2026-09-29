"""Command line interface.

    resim info case.yaml [--vtk out.vtk]    build grid, properties and domain; print a summary
    resim run  case.yaml [--vtk out.vtk]    also run the case's simulation section
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from resim.io.case import load_case, run_simulation
from resim.io.vtk import write_vtk
from resim.units import BAR, DAY


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="resim", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("command", choices=["info", "run"])
    parser.add_argument("case", help="YAML case file")
    parser.add_argument("--vtk", help="write grid and properties (and final pressure) to a VTK file")
    args = parser.parse_args(argv)

    try:
        case = load_case(args.case)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(case.grid.summary())
    print(case.props.summary())
    print(case.domain.summary())

    data = {name: case.props[name] for name in case.props.names}
    if args.command == "run":
        result = run_simulation(case)
        if hasattr(result, "times"):
            p = result.pressure[-1]
            print(f"transient: {len(result.times) - 1} steps to t = {result.times[-1] / DAY:g} days")
        else:
            p = result.pressure
            print(f"steady: net boundary outflow {result.boundary_flux.sum() * DAY:.6g} m³/day")
        print(f"pressure: min {p.min() / BAR:.4f} bar, mean {p.mean() / BAR:.4f} bar, max {p.max() / BAR:.4f} bar")
        full = np.full(case.grid.n_cells, np.nan)
        full[case.grid.global_to_active[case.domain.cell_global]] = p / BAR
        data["PRESSURE"] = full

    if args.vtk:
        write_vtk(args.vtk, case.grid, data)
        print(f"wrote {args.vtk}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
