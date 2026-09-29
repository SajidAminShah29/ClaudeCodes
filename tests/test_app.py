"""Headless tests of the Streamlit app (skipped if Streamlit is not installed)."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).parent.parent / "app" / "main.py")


def test_app_loads_every_example():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    box = at.selectbox[0]
    for option in box.options:
        at = box.select(option).run()
        assert not at.exception and not at.error
        assert at.metric[1].value != ""  # active cells shown
        box = at.selectbox[0]
