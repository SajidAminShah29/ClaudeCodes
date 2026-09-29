"""Reader for Eclipse GRDECL-style keyword files.

Handles ``--`` comments, ``N*value`` repeat notation, ``/`` record
terminators and data-less flag keywords. Files with bare numbers and no
keyword are also accepted (returned under the key ``None``).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

_KEYWORD = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,7}$")


def _tokens(text: str):
    for line in text.splitlines():
        line = line.split("--", 1)[0]
        for tok in line.replace("/", " / ").split():
            yield tok


def _expand(tok: str) -> list[str]:
    if "*" in tok:
        count, _, value = tok.partition("*")
        if not value:
            raise ValueError(f"defaulted value {tok!r} is not supported")
        return [value] * int(count)
    return [tok]


def _convert(values: list[str]):
    try:
        return np.array([float(v.replace("D", "E").replace("d", "e")) for v in values])
    except ValueError:
        return [v.strip("'\"") for v in values]


def parse_grdecl(text: str) -> dict:
    """Parse keyword records. Returns ``{keyword: array or list}``."""
    records: dict = {}
    keyword, values, open_record = None, [], False
    for tok in _tokens(text):
        if tok == "/":
            records[keyword] = _convert(values)
            keyword, values, open_record = None, [], False
        elif _KEYWORD.match(tok) and not values:
            if open_record:
                records[keyword] = _convert([])  # previous keyword was a flag
            keyword, open_record = tok.upper(), True
        else:
            values.extend(_expand(tok))
            open_record = True
    if values:
        if keyword is not None:
            raise ValueError(f"keyword {keyword} is not terminated by '/'")
        records[None] = _convert(values)
    elif open_record and keyword is not None:
        records[keyword] = _convert([])
    return records


def read_grdecl(path) -> dict:
    return parse_grdecl(Path(path).read_text())


def read_values(path, keyword: str | None = None) -> np.ndarray:
    """Read one numeric array from a text / GRDECL / .npy file."""
    path = Path(path)
    if path.suffix == ".npy":
        if keyword is not None:
            raise ValueError(f"{path}: 'keyword' is not supported for .npy files")
        return np.load(path).ravel()
    records = read_grdecl(path)
    if keyword is not None:
        if keyword.upper() not in records:
            raise KeyError(f"{path}: keyword {keyword.upper()} not found (have {sorted(k for k in records if k)})")
        data = records[keyword.upper()]
    else:
        data_records = {k: v for k, v in records.items() if len(v)}
        if len(data_records) != 1:
            raise ValueError(f"{path}: contains {len(data_records)} data records; specify 'keyword'")
        data = next(iter(data_records.values()))
    if not isinstance(data, np.ndarray):
        raise ValueError(f"{path}: record is not numeric")
    return data
