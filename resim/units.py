"""Unit conversion factors.

The simulator core works exclusively in SI units (m, m², Pa, Pa·s, s, kg/m³).
YAML input uses practical metric units, converted once at the input boundary:

    length          m
    permeability    mD
    pressure        bar
    viscosity       cP
    time            day
    rate            m³/day
    compressibility 1/bar
    density         kg/m³
"""

DARCY = 9.869233e-13  # m²
MILLIDARCY = DARCY * 1e-3  # m²
BAR = 1e5  # Pa
CENTIPOISE = 1e-3  # Pa·s
DAY = 86400.0  # s
GRAVITY = 9.80665  # m/s²
