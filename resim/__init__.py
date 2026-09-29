"""resim: a reservoir simulator."""

from resim.grid import Grid, build_cartesian
from resim.props import PropertySet, build_properties
from resim.preproc import Domain, build_domain

__all__ = [
    "Grid",
    "build_cartesian",
    "PropertySet",
    "build_properties",
    "Domain",
    "build_domain",
]
__version__ = "0.1.0"
