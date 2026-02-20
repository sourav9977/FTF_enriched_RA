"""FTF Enriched RA — Streamlit Frontend.

Launch with:
    streamlit run app.py
"""

import os
import sys
import tempfile
import logging
from pathlib import Path
from io import StringIO

import streamlit as st
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULTS
from src.run_pipeline import run_pipeline
from src.pipeline_logger import setup_logging


# ── Page Config ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="FTF Enriched RA",
    page_icon="F",
    layout="wide",
)


# ══════════════════════════════════════════════════════════════════════
#  Persistence helpers — widgets lose state when not rendered.
#  We copy every widget value into a durable "store_*" key on change.
# ══════════════════════════════════════════════════════════════════════

# ── File persistence ──────────────────────────────────────────────────

_FILE_SLOTS = {
    "ra":      {"widget_key": "widget_ra",      "store_key": "file_ra",      "label": "RA"},
    "trends":  {"widget_key": "widget_trends",  "store_key": "file_trends",  "label": "Trend Report"},
    "sales":   {"widget_key": "widget_sales",   "store_key": "file_sales",   "label": "Sales Data"},
    "catalog": {"widget_key": "widget_catalog", "store_key": "file_catalog", "label": "Product Catalog"},
    "aop":     {"widget_key": "widget_aop",     "store_key": "file_aop",     "label": "AOP Targets"},
}


def _on_file_change(slot_id: str):
    info = _FILE_SLOTS[slot_id]
    uploaded = st.session_state.get(info["widget_key"])
    if uploaded is not None:
        st.session_state[info["store_key"]] = uploaded
    else:
        st.session_state.pop(info["store_key"], None)


def _get_file(slot_id: str):
    return st.session_state.get(_FILE_SLOTS[slot_id]["store_key"])


# ── Config persistence ────────────────────────────────────────────────

_CFG_PARAMS = [
    {"widget_key": "widget_overlap_threshold",    "store_key": "cfg_overlap_threshold",    "default_key": "overlap_threshold",    "label": "Overlap Threshold"},
    {"widget_key": "widget_replacement_threshold", "store_key": "cfg_replacement_threshold", "default_key": "replacement_threshold", "label": "Replacement Threshold"},
    {"widget_key": "widget_ftf_added_cap_pct",    "store_key": "cfg_ftf_added_cap_pct",    "default_key": "ftf_added_cap_pct",    "label": "FTF Added Cap %"},
    {"widget_key": "widget_moq_per_option",       "store_key": "cfg_moq_per_option",       "default_key": "moq_per_option",       "label": "MOQ per Option"},
    {"widget_key": "widget_max_adjustment_pct",   "store_key": "cfg_max_adjustment_pct",   "default_key": "max_adjustment_pct",   "label": "Max Adjustment %"},
    {"widget_key": "widget_shelf_life_days",      "store_key": "cfg_shelf_life_days",      "default_key": "shelf_life_days",      "label": "Shelf Life (days)"},
    {"widget_key": "widget_str_target",           "store_key": "cfg_str_target",           "default_key": "str_target",           "label": "STR Target"},
    {"widget_key": "widget_k_neighbors",          "store_key": "cfg_k_neighbors",          "default_key": "k_neighbors",          "label": "K Neighbors"},
    {"widget_key": "widget_min_nearest_neighbors", "store_key": "cfg_min_nearest_neighbors", "default_key": "min_nearest_neighbors", "label": "Min Nearest Neighbors"},
]


def _on_cfg_change(store_key: str, widget_key: str):
    st.session_state[store_key] = st.session_state[widget_key]


def _get_cfg(store_key: str, default_key: str):
    return st.session_state.get(store_key, DEFAULTS[default_key])


# ── Other helpers ─────────────────────────────────────────────────────

def _save_upload(uploaded_file, tmp_dir: str) -> str:
    path = os.path.join(tmp_dir, uploaded_file.name)
    uploaded_file.seek(0)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


def _count_rows(uploaded_file) -> int:
    try:
        uploaded_file.seek(0)
        name = uploaded_file.name.lower()
        if name.endswith(".csv"):
            n = sum(1 for _ in uploaded_file) - 1
            uploaded_file.seek(0)
            return max(n, 0)
        else:
            df = pd.read_excel(uploaded_file)
            uploaded_file.seek(0)
            return len(df)
    except Exception:
        uploaded_file.seek(0)
        return -1


# ── Sidebar Navigation ───────────────────────────────────────────────

st.sidebar.title("FTF Enriched RA")
page = st.sidebar.radio(
    "Navigation",
    ["Upload Inputs", "Set Configurations", "Generate Enriched RA"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.caption("Upload Status")
for slot_id, info in _FILE_SLOTS.items():
    f = _get_file(slot_id)
    if f is not None:
        st.sidebar.markdown(f"- **{info['label']}**: {f.name}")
    else:
        st.sidebar.markdown(f"- {info['label']}: --")


# ══════════════════════════════════════════════════════════════════════
#  PAGE 1: Upload Inputs
# ══════════════════════════════════════════════════════════════════════

if page == "Upload Inputs":
    st.header("Upload Input Files")
    st.caption("Upload the required and optional data files for the pipeline.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Required Files")
        st.file_uploader("RA (Range Architecture)", type=["csv", "xlsx", "xls"],
                         key="widget_ra", on_change=_on_file_change, args=("ra",))
        st.file_uploader("Trend Report", type=["csv", "xlsx", "xls"],
                         key="widget_trends", on_change=_on_file_change, args=("trends",))
        st.file_uploader("Sales Data", type=["csv", "xlsx", "xls"],
                         key="widget_sales", on_change=_on_file_change, args=("sales",))

    with col2:
        st.subheader("Optional Files")
        st.file_uploader("Product Catalog", type=["csv", "xlsx", "xls"],
                         key="widget_catalog", on_change=_on_file_change, args=("catalog",))
        st.file_uploader("AOP Targets", type=["csv", "xlsx", "xls"],
                         key="widget_aop", on_change=_on_file_change, args=("aop",))

    st.markdown("---")
    st.subheader("File Summary")

    summary_data = []
    required_labels = {"RA", "Trend Report", "Sales Data"}
    for slot_id, info in _FILE_SLOTS.items():
        f = _get_file(slot_id)
        if f is not None:
            rows = _count_rows(f)
            row_str = f"{rows:,}" if rows >= 0 else "unknown"
            summary_data.append({"File": info["label"], "Status": "Uploaded", "Filename": f.name, "Rows": row_str})
        else:
            summary_data.append({"File": info["label"], "Status": "Required" if info["label"] in required_labels else "Optional", "Filename": "--", "Rows": "--"})

    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════
#  PAGE 2: Set Configurations
# ══════════════════════════════════════════════════════════════════════

elif page == "Set Configurations":
    st.header("Pipeline Configuration")
    st.caption("Adjust thresholds, MOQ, and other parameters. Values are saved automatically and persist across pages.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Matching Thresholds")

        st.slider(
            "Overlap Threshold", min_value=0.0, max_value=1.0, step=0.05,
            value=_get_cfg("cfg_overlap_threshold", "overlap_threshold"),
            key="widget_overlap_threshold",
            on_change=_on_cfg_change, args=("cfg_overlap_threshold", "widget_overlap_threshold"),
            help="Min overlap score for FTF APPROVED classification (Tier 1).",
        )
        st.slider(
            "Replacement Threshold", min_value=0.0, max_value=1.0, step=0.05,
            value=_get_cfg("cfg_replacement_threshold", "replacement_threshold"),
            key="widget_replacement_threshold",
            on_change=_on_cfg_change, args=("cfg_replacement_threshold", "widget_replacement_threshold"),
            help="Below this threshold, confidence-based replacement applies (Tier 3).",
        )
        st.slider(
            "FTF Added Cap %", min_value=0.0, max_value=0.50, step=0.05,
            value=_get_cfg("cfg_ftf_added_cap_pct", "ftf_added_cap_pct"),
            key="widget_ftf_added_cap_pct",
            on_change=_on_cfg_change, args=("cfg_ftf_added_cap_pct", "widget_ftf_added_cap_pct"),
            help="Max percentage of final RA that can be FTF ADDED items.",
        )

    with col2:
        st.subheader("Quantity & MOQ")

        st.number_input(
            "MOQ per Option", min_value=0, max_value=50000, step=100,
            value=_get_cfg("cfg_moq_per_option", "moq_per_option"),
            key="widget_moq_per_option",
            on_change=_on_cfg_change, args=("cfg_moq_per_option", "widget_moq_per_option"),
            help="Minimum order quantity. Items below this are boosted or dropped.",
        )
        st.slider(
            "Max Adjustment %", min_value=0.0, max_value=0.30, step=0.01,
            value=_get_cfg("cfg_max_adjustment_pct", "max_adjustment_pct"),
            key="widget_max_adjustment_pct",
            on_change=_on_cfg_change, args=("cfg_max_adjustment_pct", "widget_max_adjustment_pct"),
            help="Max quantity adjustment for APPROVED items based on performance score.",
        )
        st.number_input(
            "Shelf Life (days)", min_value=7, max_value=365, step=5,
            value=_get_cfg("cfg_shelf_life_days", "shelf_life_days"),
            key="widget_shelf_life_days",
            on_change=_on_cfg_change, args=("cfg_shelf_life_days", "widget_shelf_life_days"),
            help="Product shelf life for quantity capping.",
        )
        st.slider(
            "STR Target", min_value=0.50, max_value=1.0, step=0.05,
            value=_get_cfg("cfg_str_target", "str_target"),
            key="widget_str_target",
            on_change=_on_cfg_change, args=("cfg_str_target", "widget_str_target"),
            help="Sell-through rate target for shelf-life cap calculation.",
        )

    with col3:
        st.subheader("Nearest Neighbor")

        st.number_input(
            "K Neighbors", min_value=1, max_value=20,
            value=_get_cfg("cfg_k_neighbors", "k_neighbors"),
            key="widget_k_neighbors",
            on_change=_on_cfg_change, args=("cfg_k_neighbors", "widget_k_neighbors"),
            help="Number of nearest neighbors for FTF ADDED quantity estimation.",
        )
        st.number_input(
            "Min Nearest Neighbors", min_value=1, max_value=10,
            value=_get_cfg("cfg_min_nearest_neighbors", "min_nearest_neighbors"),
            key="widget_min_nearest_neighbors",
            on_change=_on_cfg_change, args=("cfg_min_nearest_neighbors", "widget_min_nearest_neighbors"),
            help="Min neighbors required before falling back to family-level percentile.",
        )

    # Active config summary
    st.markdown("---")
    st.subheader("Active Configuration")

    cfg_summary = []
    for p in _CFG_PARAMS:
        current = _get_cfg(p["store_key"], p["default_key"])
        default = DEFAULTS[p["default_key"]]
        cfg_summary.append({
            "Parameter": p["label"],
            "Value": current,
            "Default": default,
            "Modified": "Yes" if current != default else "",
        })

    st.dataframe(pd.DataFrame(cfg_summary), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════
#  PAGE 3: Generate Enriched RA
# ══════════════════════════════════════════════════════════════════════

elif page == "Generate Enriched RA":
    st.header("Generate Enriched RA")

    ra_file = _get_file("ra")
    trend_file = _get_file("trends")
    sales_file = _get_file("sales")
    catalog_file = _get_file("catalog")
    aop_file = _get_file("aop")

    required_ready = all([ra_file, trend_file, sales_file])

    # Build config overrides from durable store keys
    config_overrides = {}
    for p in _CFG_PARAMS:
        val = _get_cfg(p["store_key"], p["default_key"])
        if val != DEFAULTS[p["default_key"]]:
            config_overrides[p["default_key"]] = val

    if not required_ready:
        missing = []
        if not ra_file:
            missing.append("RA")
        if not trend_file:
            missing.append("Trend Report")
        if not sales_file:
            missing.append("Sales Data")
        st.warning(f"Missing required files: **{', '.join(missing)}**. Go to **Upload Inputs** first.")

    # Show the config that will be used for this run
    st.subheader("Run Configuration")
    run_cfg_data = []
    for p in _CFG_PARAMS:
        val = _get_cfg(p["store_key"], p["default_key"])
        default = DEFAULTS[p["default_key"]]
        run_cfg_data.append({
            "Parameter": p["label"],
            "Value": val,
            "Modified": "Yes" if val != default else "",
        })
    st.dataframe(pd.DataFrame(run_cfg_data), use_container_width=True, hide_index=True, height=370)

    run_clicked = st.button(
        "Start Run",
        disabled=not required_ready,
        type="primary",
        use_container_width=True,
    )

    if run_clicked:
        tmp_dir = tempfile.mkdtemp(prefix="ftf_run_")
        output_path = os.path.join(tmp_dir, "enriched_ra.xlsx")

        ra_path = _save_upload(ra_file, tmp_dir)
        trends_path = _save_upload(trend_file, tmp_dir)
        sales_path = _save_upload(sales_file, tmp_dir)
        catalog_path = _save_upload(catalog_file, tmp_dir) if catalog_file else None
        aop_path = _save_upload(aop_file, tmp_dir) if aop_file else None

        # Reset logging for a fresh run
        import src.pipeline_logger as pl
        pl._INITIALIZED = False
        root_logger = logging.getLogger("src")
        for h in root_logger.handlers[:]:
            root_logger.removeHandler(h)

        log_stream = StringIO()
        log_capture = logging.StreamHandler(log_stream)
        log_capture.setLevel(logging.INFO)
        log_capture.setFormatter(logging.Formatter("%(message)s"))

        progress_bar = st.progress(0, text="Initializing pipeline...")

        step_labels = [
            (5,  "Step 0: Loading configuration..."),
            (10, "Step 1: Ingesting & validating data..."),
            (20, "Step 2: Normalizing attributes..."),
            (35, "Step 3: Running factor analysis..."),
            (50, "Step 4: Scoring overlaps & classifying items..."),
            (65, "Steps 5-7: AOP breakdown & quantity estimation..."),
            (80, "Step 8: Rebalancing & applying constraints..."),
            (90, "Step 9: Generating output..."),
        ]

        with st.status("Running FTF Pipeline...", expanded=True) as run_status:
            try:
                setup_logging()
                root_logger = logging.getLogger("src")
                root_logger.addHandler(log_capture)

                for pct, label in step_labels:
                    progress_bar.progress(pct, text=label)
                    st.write(label)

                result_df = run_pipeline(
                    ra_path=ra_path,
                    trends_path=trends_path,
                    sales_path=sales_path,
                    catalog_path=catalog_path,
                    aop_path=aop_path,
                    output_path=output_path,
                    config_overrides=config_overrides if config_overrides else None,
                )

                root_logger.removeHandler(log_capture)
                progress_bar.progress(100, text="Pipeline completed!")
                run_status.update(label="Pipeline completed successfully!", state="complete", expanded=False)

            except Exception as e:
                root_logger.removeHandler(log_capture)
                progress_bar.progress(100, text="Pipeline failed.")
                run_status.update(label=f"Pipeline failed: {e}", state="error")
                st.error(f"Pipeline error: {e}")
                st.exception(e)
                st.stop()

        log_stream.seek(0)
        log_text = log_stream.read()

        st.session_state["result_df"] = result_df
        st.session_state["output_path"] = output_path
        st.session_state["log_text"] = log_text
        st.session_state["run_complete"] = True

    # ── Display Results (persisted via session state) ─────────────────

    if st.session_state.get("run_complete"):
        result_df = st.session_state["result_df"]
        output_path = st.session_state["output_path"]
        log_text = st.session_state.get("log_text", "")

        st.markdown("---")
        st.subheader("Results Summary")

        total = len(result_df)
        if total > 0 and "ftf_status" in result_df.columns:
            approved = int((result_df["ftf_status"] == "APPROVED").sum())
            original = int((result_df["ftf_status"] == "ORIGINAL").sum())
            added = int((result_df["ftf_status"] == "ADDED").sum())

            orig_qty = pd.to_numeric(result_df.get("original_quantity", 0), errors="coerce").sum()
            adj_qty = pd.to_numeric(result_df.get("adjusted_quantity", 0), errors="coerce").sum()
            delta_pct = ((adj_qty - orig_qty) / orig_qty * 100) if orig_qty > 0 else 0

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Items", f"{total:,}")
            m2.metric("APPROVED", f"{approved:,}", f"{approved/total:.1%}")
            m3.metric("ORIGINAL", f"{original:,}", f"{original/total:.1%}")
            m4.metric("ADDED", f"{added:,}", f"{added/total:.1%}")

            q1, q2, q3 = st.columns(3)
            q1.metric("Original Qty", f"{orig_qty:,.0f}")
            q2.metric("Adjusted Qty", f"{adj_qty:,.0f}")
            q3.metric("Qty Delta", f"{delta_pct:+.1f}%")

        st.markdown("---")
        st.subheader("Enriched RA Details")

        if "ftf_status" in result_df.columns:
            status_options = ["All"] + sorted(result_df["ftf_status"].dropna().unique().tolist())
            filter_status = st.selectbox("Filter by FTF Status", status_options)
            display_df = result_df if filter_status == "All" else result_df[result_df["ftf_status"] == filter_status]
            st.dataframe(display_df, use_container_width=True, height=500)
            st.caption(f"Showing {len(display_df):,} of {total:,} rows")
        else:
            st.dataframe(result_df, use_container_width=True, height=500)

        st.markdown("---")
        st.subheader("Downloads")

        dl1, dl2 = st.columns(2)

        if Path(output_path).exists():
            with open(output_path, "rb") as f:
                dl1.download_button(
                    label="Download Enriched RA (Excel)",
                    data=f.read(),
                    file_name="enriched_ra.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    type="primary",
                )

        log_dir = PROJECT_ROOT / "logs"
        if log_dir.exists():
            log_files = sorted(log_dir.glob("ftf_pipeline_*.log"), reverse=True)
            if log_files:
                with open(log_files[0], "r") as f:
                    dl2.download_button(
                        label="Download Calculation Log",
                        data=f.read(),
                        file_name=log_files[0].name,
                        mime="text/plain",
                        use_container_width=True,
                    )

        with st.expander("View Pipeline Log", expanded=False):
            st.text(log_text if log_text.strip() else "No log output captured.")
