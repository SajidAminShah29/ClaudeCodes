"""resim web app — Step 1: load an example case and show what the engine built.

Run with:   streamlit run app/main.py

How Streamlit works: this whole script runs top to bottom every time the user
interacts with the page (e.g. picks a different case). Each ``st.*`` call
draws one element on the page.
"""

from pathlib import Path

import streamlit as st

from resim.io import load_case

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

st.set_page_config(page_title="resim", layout="wide")
st.title("resim — reservoir simulator")

# 1. Let the user choose a case file.
case_files = sorted(EXAMPLES.glob("*.yaml"))
choice = st.selectbox("Example case", case_files, format_func=lambda p: p.stem)


# 2. Ask the engine to build it. Cached: only rebuilt when the chosen file changes.
@st.cache_resource
def build(path: Path):
    return load_case(path)


try:
    case = build(choice)
except ValueError as exc:  # the engine's validation messages are already user-friendly
    st.error(f"Could not build this case:\n\n{exc}")
    st.stop()

# 3. Show what the engine produced.
grid, props, domain = case.grid, case.props, case.domain
nx, ny, nz = grid.dims

col1, col2, col3, col4 = st.columns(4)
col1.metric("Grid", f"{nx} × {ny} × {nz}")
col2.metric("Active cells", f"{grid.n_cells:,}")
col3.metric("Pore volume", f"{domain.pore_volume.sum():,.0f} m³")
col4.metric("Connections", f"{domain.n_conn:,}")

st.subheader("Properties (active cells)")
st.dataframe(
    [
        {"property": name, "min": v.min(), "mean": v.mean(), "max": v.max()}
        for name in props.names
        for v in [props.active(name)]
    ],
    hide_index=True,
)

with st.expander("Engine summaries"):
    st.text(grid.summary())
    st.text(domain.summary())
