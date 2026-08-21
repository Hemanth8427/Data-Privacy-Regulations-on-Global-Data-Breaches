"""
Streamlit dashboard: "Assessing the Effectiveness of Data Privacy Regulations on
Global Data Breaches".

The interactive decision-support deliverable described in Objective 5 and sections
1.5, 2.7 and 3.9 of the dissertation. All analysis is delegated to ``analysis.py``,
a faithful port of ``Ref_Hemanth.ipynb``, so the tables and figures shown here
reproduce Chapter 4.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import analysis

MODELS_DIR = Path(__file__).parent / "models"
METRICS_PATH = MODELS_DIR / "metrics.json"

CAUSATION_NOTE = (
    "**Association, not causation.** Regulatory periods are *temporal markers* "
    "based on the year a breach was recorded — they do not establish that an "
    "organisation was legally subject to GDPR or CCPA. Differences observed either "
    "side of a boundary may reflect changes in reporting behaviour, detection "
    "capability or the mix of organisations in the dataset, not regulatory effect."
)

PERIOD_FILTERS = {
    "All periods": None,
    "Pre-GDPR (before 2018)": ("gdpr_period", "Pre-GDPR"),
    "Post-GDPR (2018 onwards)": ("gdpr_period", "Post-GDPR"),
    "Pre-CCPA (before 2020)": ("ccpa_period", "Pre-CCPA"),
    "Post-CCPA (2020 onwards)": ("ccpa_period", "Post-CCPA"),
}

st.set_page_config(
    page_title="Data Breach & Privacy Regulation Dashboard",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Cached data and model loading
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Cleaning the dataset…")
def load_data(file_bytes: bytes) -> dict:
    """Clean the uploaded CSV. Cached on the file's raw bytes."""
    import io

    return analysis.prepare_data(io.BytesIO(file_bytes))


def _load_saved_pipelines(expected_threshold: float, n_rows: int):
    """Load pipelines from ``models/`` if they match the current dataset.

    Section 3.9 requires the dashboard to reuse the *same saved preprocessing
    pipeline and model* as the offline analysis. Artifacts are only accepted when
    the manifest's row count and threshold match the uploaded data, so a mismatched
    pair can never silently produce wrong predictions.
    """
    if not METRICS_PATH.exists():
        return None, "no-artifacts"

    try:
        manifest = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None, "unreadable-manifest"

    if int(manifest.get("analysis_rows", -1)) != int(n_rows):
        return None, "row-mismatch"

    if abs(float(manifest.get("threshold", -1)) - float(expected_threshold)) > 1.0:
        return None, "threshold-mismatch"

    pipelines: dict = {}
    for name in manifest.get("models", []):
        path = MODELS_DIR / f"{name.lower().replace(' ', '_')}.joblib"
        if not path.exists():
            return None, "missing-model-file"
        try:
            pipelines[name] = joblib.load(path)
        except Exception:
            return None, "unloadable-model"

    if not pipelines:
        return None, "no-models-listed"

    return pipelines, "loaded"


def _save_pipelines(pipelines: dict, split: dict, df: pd.DataFrame) -> None:
    """Persist freshly fitted pipelines so later runs reuse them."""
    try:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        for name, pipeline in pipelines.items():
            joblib.dump(pipeline, MODELS_DIR / f"{name.lower().replace(' ', '_')}.joblib")

        METRICS_PATH.write_text(
            json.dumps(
                {
                    "source_file": "uploaded via dashboard",
                    "random_state": analysis.RANDOM_STATE,
                    "stratified": False,
                    "threshold": split["threshold"],
                    "analysis_rows": int(len(df)),
                    "train_rows": int(len(split["X_train"])),
                    "test_rows": int(len(split["X_test"])),
                    "models": list(pipelines.keys()),
                    "primary_model": analysis.PRIMARY_MODEL,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception:
        # Persisting is a convenience; a read-only deployment must still work.
        pass


@st.cache_resource(show_spinner="Preparing the machine-learning models…")
def get_models(signature: str, _df: pd.DataFrame) -> dict:
    """Return the scored model bundle, preferring saved artifacts.

    ``signature`` is the cache key; ``_df`` is excluded from hashing by its
    leading underscore.
    """
    split = analysis.split_and_threshold(_df)

    pipelines, status = _load_saved_pipelines(split["threshold"], len(_df))

    if pipelines is None:
        pipelines = analysis.fit_pipelines(split["X_train"], split["y_train"])
        _save_pipelines(pipelines, split, _df)
        status = "trained"

    bundle = analysis.assemble_bundle(pipelines, split, _df)
    bundle["artifact_status"] = status
    return bundle


# ---------------------------------------------------------------------------
# Small UI helpers
# ---------------------------------------------------------------------------


def show_figure(fig) -> None:
    """Render a matplotlib figure and release it (Streamlit reruns leak figures)."""
    if fig is None:
        return
    st.pyplot(fig)
    plt.close(fig)


def download(df: pd.DataFrame, label: str, filename: str) -> None:
    """Attach a CSV download button to a table."""
    st.download_button(
        label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=f"dl_{filename}",
    )


def table(df: pd.DataFrame, filename: str | None = None, **kwargs) -> None:
    """Display a dataframe full-width, optionally with a download button."""
    st.dataframe(df, use_container_width=True, hide_index=True, **kwargs)
    if filename:
        download(df, "⬇ Download this table (CSV)", filename)


def metric_row(items: list[tuple[str, str]]) -> None:
    """Render a row of KPI metrics."""
    columns = st.columns(len(items))
    for column, (label, value) in zip(columns, items):
        column.metric(label, value)


def significance_badge(significant: bool, p_value: float) -> None:
    """Consistent p-value verdict styling."""
    if significant:
        st.success(f"Statistically significant (p = {p_value:.6f} < 0.05)")
    else:
        st.info(f"Not statistically significant (p = {p_value:.6f} ≥ 0.05)")


def section_note(text: str) -> None:
    st.caption(text)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("🔐 Breach Analytics")
st.sidebar.caption(
    "MSc Big Data Analytics dissertation dashboard — Sheffield Hallam University"
)

uploaded = st.sidebar.file_uploader(
    "Breach dataset (CSV)",
    type=["csv"],
    help=(
        "Upload the raw 'World's Biggest Data Breaches and Hacks' CSV. The file is "
        "cleaned using the same rules as the dissertation notebook."
    ),
)

if uploaded is None:
    # ----------------------- Empty state -----------------------
    st.title("Assessing the Effectiveness of Data Privacy Regulations on Global Data Breaches")
    st.markdown(
        "#### An interactive dashboard for exploring breach frequency, severity and "
        "regulatory periods"
    )

    st.info("**Upload the breach dataset CSV in the sidebar to begin.**", icon="⬆️")

    left, right = st.columns([3, 2])

    with left:
        st.subheader("What this dashboard does")
        st.markdown(
            """
            Built as the software deliverable for the dissertation *"Developing a
            Machine Learning Model to Assess the Effectiveness of Data Privacy
            Regulations on Global Data Breaches"*, it lets you:

            - Track **annual breach frequency and severity** from 2004 to 2022
            - Compare **sectors** and **breach methods** by both count and impact
            - Contrast breach outcomes **before and after GDPR (2018) and CCPA (2020)**
            - Review the **statistical tests** — Mann–Whitney U, chi-square,
              Kruskal–Wallis and OLS regression
            - Compare **five machine-learning classifiers** of high-severity breaches
            - Score a **hypothetical breach event** against the saved model
            """
        )

    with right:
        st.subheader("Expected CSV format")
        st.markdown(
            """
            The raw source file, unmodified. The cleaner expects:

            | Column | Purpose |
            |---|---|
            | `organisation` | Breached entity |
            | `records lost` | Severity measure |
            | `year` | Year the story broke |
            | `sector` | Organisational sector |
            | `method` | Breach mechanism |
            | `data sensitivity` | Ordinal 1–5 score |
            """
        )

    st.stop()


# ----------------------- Data loaded -----------------------

try:
    data = load_data(uploaded.getvalue())
except Exception as exc:
    st.error(f"Could not read that CSV: {type(exc).__name__}: {exc}")
    st.stop()

analysis_df: pd.DataFrame = data["analysis"]

if analysis_df.empty:
    st.error(
        "No usable rows after cleaning. Check that the file contains "
        "`records lost`, `year`, `sector` and `method` columns."
    )
    st.stop()

required_columns = {"records_lost", "year", "sector", "method"}
missing = required_columns - set(analysis_df.columns)
if missing:
    st.error(f"The dataset is missing required column(s): {', '.join(sorted(missing))}")
    st.stop()

year_min, year_max = int(analysis_df["year"].min()), int(analysis_df["year"].max())
all_sectors = sorted(analysis_df["sector"].dropna().unique().tolist())
all_methods = sorted(analysis_df["method"].dropna().unique().tolist())

st.sidebar.success(f"Loaded **{len(analysis_df)}** breach events ({year_min}–{year_max})")

st.sidebar.header("Filters")
section_note("")

if st.sidebar.button("↺ Reset filters", use_container_width=True):
    for key in ("flt_years", "flt_sectors", "flt_methods", "flt_period"):
        st.session_state.pop(key, None)
    st.rerun()

year_range = st.sidebar.slider(
    "Year range",
    min_value=year_min,
    max_value=year_max,
    value=(year_min, year_max),
    key="flt_years",
)

selected_sectors = st.sidebar.multiselect(
    "Sector",
    options=all_sectors,
    default=all_sectors,
    key="flt_sectors",
    help="Leave all selected to include every sector.",
)

selected_methods = st.sidebar.multiselect(
    "Breach method",
    options=all_methods,
    default=all_methods,
    key="flt_methods",
)

period_choice = st.sidebar.radio(
    "Regulatory period",
    options=list(PERIOD_FILTERS.keys()),
    index=0,
    key="flt_period",
)

st.sidebar.divider()


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the sidebar selection to the analytical dataset."""
    mask = df["year"].between(year_range[0], year_range[1])

    if selected_sectors:
        mask &= df["sector"].isin(selected_sectors)
    if selected_methods:
        mask &= df["method"].isin(selected_methods)

    period = PERIOD_FILTERS[period_choice]
    if period is not None:
        column, value = period
        mask &= df[column] == value

    return df[mask].copy()


filtered = apply_filters(analysis_df)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("Data Privacy Regulation & Global Data Breaches")
st.caption(
    "Interactive analysis of breach frequency, severity, sectoral exposure, breach "
    "methods and regulatory periods — dissertation Chapter 4."
)

if filtered.empty:
    st.warning(
        "No breach events match the current filters. Widen the selection in the "
        "sidebar to see results."
    )
    st.stop()

if len(filtered) < len(analysis_df):
    st.info(
        f"Filters active: showing **{len(filtered)}** of **{len(analysis_df)}** "
        "breach events in the descriptive tabs.",
        icon="🔎",
    )

tabs = st.tabs(
    [
        "1 · Overview",
        "2 · Temporal",
        "3 · Sector",
        "4 · Breach Method",
        "5 · Regulatory Periods",
        "6 · Statistical Tests",
        "7 · ML Models",
        "8 · Predict Severity",
    ]
)


# ---------------------------------------------------------------------------
# Tab 1 - Overview
# ---------------------------------------------------------------------------

with tabs[0]:
    st.header("Overview of the analytical sample")
    section_note("Corresponds to dissertation sections 4.2–4.3 (Tables 1–2, Figures 4–5).")

    metric_row(
        [
            ("Breach events", f"{len(filtered):,}"),
            ("Total records lost", f"{filtered['records_lost'].sum():,.0f}"),
            ("Mean records lost", f"{filtered['records_lost'].mean():,.0f}"),
            ("Median records lost", f"{filtered['records_lost'].median():,.0f}"),
            (
                "High severity (top quartile)",
                f"{int(filtered['high_severity_exploratory'].sum()):,}",
            ),
        ]
    )

    st.caption(
        "The mean sits far above the median because a small number of very large "
        "incidents dominate the total — median and percentile measures describe the "
        "typical breach more faithfully (section 4.3)."
    )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Data cleaning and final sample")
        st.caption("Table 1 — computed on the full uploaded file, not the filtered view.")
        table(data["cleaning_summary"], "table1_cleaning_summary.csv")

    with right:
        st.subheader("Descriptive statistics for records lost")
        st.caption("Table 2 — reflects the current filter selection.")
        descriptive = analysis.severity_descriptive_table(filtered)
        st.dataframe(
            descriptive,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Records lost": st.column_config.NumberColumn(format="%.0f")
            },
        )
        download(descriptive, "⬇ Download this table (CSV)", "table2_severity_descriptive.csv")

    st.divider()

    st.subheader("Distribution of breach severity")
    left, right = st.columns(2)
    with left:
        show_figure(analysis.fig_severity_histogram(filtered))
        st.caption("Figure 4 — log-transformed severity.")
    with right:
        show_figure(analysis.fig_severity_boxplot(filtered))
        st.caption("Figure 5 — boxplot of log-transformed severity.")

    st.caption(
        "Severity is analysed as log(records lost + 1) because the raw distribution "
        "is heavily right-skewed (section 3.6)."
    )

    with st.expander("Data quality of the cleaned file"):
        st.write(
            f"Cleaned shape: **{data['cleaned'].shape[0]} rows × "
            f"{data['cleaned'].shape[1]} columns**"
        )
        st.write(
            "Empty columns removed: "
            + (", ".join(f"`{c}`" for c in data["meta"]["empty_columns"]) or "none")
        )
        st.write(f"Duplicate rows removed: **{data['meta']['duplicates_removed']}**")
        table(analysis.data_quality_table(data["cleaned"]))

    with st.expander("Glossary and definitions"):
        st.markdown(
            f"""
            - **Breach event** — the unit of analysis: one recorded organisational
              breach, not one affected individual.
            - **Severity** — the number of records lost in an incident. Modelled as
              `log(records lost + 1)` in the statistical tests.
            - **High-severity breach** — an incident at or above the 75th percentile
              of records lost. For modelling, that threshold is computed from the
              **training split only**, so it never leaks into the test set
              (sections 3.6, 4.2).
            - **GDPR period** — recorded before {analysis.GDPR_YEAR} vs.
              {analysis.GDPR_YEAR} onwards.
            - **CCPA period** — recorded before {analysis.CCPA_YEAR} vs.
              {analysis.CCPA_YEAR} onwards.
            """
        )
        st.warning(CAUSATION_NOTE)

    with st.expander("Limitations of the study (section 5.4)"):
        st.markdown(
            """
            1. Built on a **secondary dataset of publicly recorded breaches** —
               unreported or undocumented incidents are absent.
            2. Regulatory-period labels **do not verify** which organisations were
               legally subject to GDPR or CCPA.
            3. The **observational design cannot establish causation**.
            4. Severity is **heavily skewed with extreme outliers**, which complicates
               modelling.
            5. The sample is **small for a cybersecurity prediction task** and the
               high-severity class is **imbalanced**.
            """
        )


# ---------------------------------------------------------------------------
# Tab 2 - Temporal
# ---------------------------------------------------------------------------

with tabs[1]:
    st.header("Temporal pattern of data breaches")
    section_note("Corresponds to section 4.4 (Tables 3–4, Figures 1–3).")

    annual_counts = analysis.annual_breach_counts(filtered)
    annual_severity = analysis.annual_severity_table(filtered)

    show_figure(analysis.fig_annual_frequency(annual_counts))
    st.caption(
        "Figure 1 — annual breach frequency. Dashed lines mark the GDPR (2018) and "
        "CCPA (2020) reference years."
    )

    st.info(
        "Rising counts in later years should **not** be read as deteriorating "
        "security. Detection, reporting practice and dataset composition all changed "
        "over the period (section 5.2).",
        icon="ℹ️",
    )

    st.divider()

    st.subheader("Total records lost by year")
    log_scale = st.toggle(
        "Logarithmic scale",
        value=False,
        help="A log scale makes the very wide spread across years readable (Figure 3).",
    )
    show_figure(analysis.fig_total_records_by_year(annual_severity, log_scale))
    st.caption("Figure 3 — logarithmic scale." if log_scale else "Figure 2 — linear scale.")

    st.caption(
        "Cumulative annual exposure is driven by incident size, not just incident "
        "count: some low-frequency years carry very high aggregate exposure."
    )

    st.divider()

    left, right = st.columns([1, 2])
    with left:
        st.subheader("Annual frequency")
        st.caption("Table 3")
        table(annual_counts, "table3_annual_frequency.csv")

    with right:
        st.subheader("Annual severity summary")
        st.caption("Table 4")
        st.dataframe(
            annual_severity,
            use_container_width=True,
            hide_index=True,
            column_config={
                column: st.column_config.NumberColumn(format="%.0f")
                for column in annual_severity.columns
                if column != "year"
            },
        )
        download(annual_severity, "⬇ Download this table (CSV)", "table4_annual_severity.csv")


# ---------------------------------------------------------------------------
# Tab 3 - Sector
# ---------------------------------------------------------------------------

with tabs[2]:
    st.header("Sectoral distribution and severity")
    section_note("Corresponds to section 4.5 (Tables 5–6, Figures 6–7).")

    sector_freq = analysis.sector_frequency_table(filtered)
    sector_sev = analysis.sector_severity_table(filtered)

    left, right = st.columns(2)

    with left:
        show_figure(analysis.fig_sector_frequency(sector_freq))
        st.caption("Figure 6 — number of breaches by sector.")

    with right:
        show_figure(
            analysis.fig_sector_severity_box(filtered, analysis.MIN_SECTOR_OBSERVATIONS)
        )
        st.caption(
            f"Figure 7 — severity across sectors with at least "
            f"{analysis.MIN_SECTOR_OBSERVATIONS} observations."
        )

    st.info(
        "Frequency and severity rank differently. The web sector leads on both count "
        "and aggregate exposure, whereas technology shows fewer incidents but high "
        "mean impact, and health shows relatively high severity at lower frequency "
        "(section 4.5).",
        icon="ℹ️",
    )

    st.divider()

    st.subheader("Frequency of breach events by sector")
    st.caption("Table 5")
    st.dataframe(
        sector_freq,
        use_container_width=True,
        hide_index=True,
        column_config={"percentage": st.column_config.NumberColumn(format="%.3f %%")},
    )
    download(sector_freq, "⬇ Download this table (CSV)", "table5_sector_frequency.csv")

    st.subheader("Breach severity by sector")
    st.caption("Table 6")
    st.dataframe(
        sector_sev,
        use_container_width=True,
        hide_index=True,
        column_config={
            column: st.column_config.NumberColumn(format="%.0f")
            for column in sector_sev.columns
            if column not in ("sector", "percentage")
        }
        | {"percentage": st.column_config.NumberColumn(format="%.3f %%")},
    )
    download(sector_sev, "⬇ Download this table (CSV)", "table6_sector_severity.csv")


# ---------------------------------------------------------------------------
# Tab 4 - Breach method
# ---------------------------------------------------------------------------

with tabs[3]:
    st.header("Breach method and severity")
    section_note("Corresponds to section 4.6 (Tables 7–8, Figures 8–9).")

    method_freq = analysis.method_frequency_table(filtered)
    method_sev = analysis.method_severity_table(filtered)

    left, right = st.columns(2)

    with left:
        show_figure(analysis.fig_method_frequency(method_freq))
        st.caption("Figure 8 — number of breaches by method.")

    with right:
        show_figure(analysis.fig_method_severity_box(filtered))
        st.caption("Figure 9 — severity distribution by method.")

    st.info(
        "The most common mechanism is not the most damaging one. Hacking dominates by "
        "count, while poor-security incidents carry the highest mean severity — so "
        "controls must address internal practice and configuration weaknesses as well "
        "as external attack (sections 4.6, 5.3.3).",
        icon="ℹ️",
    )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Frequency by method")
        st.caption("Table 7")
        st.dataframe(
            method_freq,
            use_container_width=True,
            hide_index=True,
            column_config={
                "percentage": st.column_config.NumberColumn(format="%.3f %%")
            },
        )
        download(method_freq, "⬇ Download this table (CSV)", "table7_method_frequency.csv")

    with right:
        st.subheader("Severity by method")
        st.caption("Table 8")
        st.dataframe(
            method_sev,
            use_container_width=True,
            hide_index=True,
            column_config={
                column: st.column_config.NumberColumn(format="%.0f")
                for column in method_sev.columns
                if column not in ("method", "percentage")
            }
            | {"percentage": st.column_config.NumberColumn(format="%.3f %%")},
        )
        download(method_sev, "⬇ Download this table (CSV)", "table8_method_severity.csv")


# ---------------------------------------------------------------------------
# Tab 5 - Regulatory periods
# ---------------------------------------------------------------------------

with tabs[4]:
    st.header("Regulatory period comparisons")
    section_note("Corresponds to sections 4.7–4.8 (Tables 9, 11, Figures 10–13).")

    st.warning(CAUSATION_NOTE)

    if period_choice != "All periods":
        st.info(
            f"The sidebar is filtered to **{period_choice}**, so one side of each "
            "comparison below will be empty. Select *All periods* for the full "
            "contrast.",
            icon="⚠️",
        )

    for label, period_col, filename in [
        ("GDPR", "gdpr_period", "table9_gdpr_summary.csv"),
        ("CCPA", "ccpa_period", "table11_ccpa_summary.csv"),
    ]:
        st.subheader(f"{label} period ({analysis.GDPR_YEAR if label == 'GDPR' else analysis.CCPA_YEAR} boundary)")

        summary = analysis.period_summary_table(filtered, period_col)

        left, right = st.columns(2)
        with left:
            show_figure(analysis.fig_period_counts(summary, period_col, label))
            st.caption(
                f"Figure {10 if label == 'GDPR' else 12} — recorded events before and "
                f"after {label}."
            )
        with right:
            show_figure(analysis.fig_period_severity_box(filtered, period_col, label))
            st.caption(
                f"Figure {11 if label == 'GDPR' else 13} — severity distribution before "
                f"and after {label}."
            )

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True,
            column_config={
                column: st.column_config.NumberColumn(format="%.0f")
                for column in summary.columns
                if column not in (period_col, "percentage_of_breaches")
            }
            | {
                "percentage_of_breaches": st.column_config.NumberColumn(
                    format="%.3f %%"
                )
            },
        )
        download(summary, "⬇ Download this table (CSV)", filename)
        st.divider()

    st.caption(
        "Formal significance testing for these comparisons is in the "
        "**Statistical Tests** tab."
    )


# ---------------------------------------------------------------------------
# Tab 6 - Statistical tests
# ---------------------------------------------------------------------------

with tabs[5]:
    st.header("Inferential statistics")
    section_note(
        "Corresponds to sections 4.9–4.11 (Tables 10, 12–16, Figure 14)."
    )

    scope = st.radio(
        "Test scope",
        ["Full dataset (reproduces the dissertation)", "Current filter selection"],
        index=0,
        horizontal=True,
        help=(
            "The published results are computed on the complete analytical sample. "
            "Switch to the filtered selection to explore subgroups — results will "
            "then differ from Chapter 4."
        ),
    )

    test_df = analysis_df if scope.startswith("Full") else filtered

    if scope.startswith("Full"):
        st.caption(f"Using all **{len(test_df)}** breach events.")
    else:
        st.warning(
            f"Using the **{len(test_df)}** filtered events — these results will not "
            "match the dissertation tables."
        )

    # --- Mann-Whitney U ---
    st.subheader("Mann–Whitney U — breach severity across regulatory periods")
    st.caption(
        "Tables 10 and 12. Non-parametric, because log severity is not normally "
        "distributed (section 3.7)."
    )

    left, right = st.columns(2)

    for column, (label, period_col, pre, post) in zip(
        (left, right),
        [
            ("GDPR", "gdpr_period", "Pre-GDPR", "Post-GDPR"),
            ("CCPA", "ccpa_period", "Pre-CCPA", "Post-CCPA"),
        ],
    ):
        with column:
            st.markdown(f"**{label}**")
            result = analysis.mann_whitney_test(test_df, period_col, pre, post)

            if "error" in result:
                st.info(result["error"])
            else:
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Measure": [
                                "U statistic",
                                "p-value",
                                "Effect size r",
                                "Effect size",
                                f"n ({pre})",
                                f"n ({post})",
                            ],
                            "Value": [
                                f"{result['u_statistic']:,.1f}",
                                f"{result['p_value']:.6f}",
                                f"{result['effect_size_r']:.3f}",
                                result["effect_size_label"],
                                f"{result['n_pre']}",
                                f"{result['n_post']}",
                            ],
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
                significance_badge(result["significant"], result["p_value"])

    st.caption(
        "A statistically significant difference with a small effect size indicates a "
        "distributional shift, not a large practical change — and not evidence that "
        "the regulation caused it (section 4.7)."
    )

    st.divider()

    # --- Chi-square ---
    st.subheader("Chi-square — regulatory period and breach composition")
    st.caption("Table 13. Cramér's V reports association strength.")

    chi_rows = []
    chi_tables = {}

    for relationship, variable1, variable2 in [
        ("GDPR period × method", "gdpr_period", "method"),
        ("GDPR period × sector", "gdpr_period", "sector"),
        ("CCPA period × method", "ccpa_period", "method"),
        ("CCPA period × sector", "ccpa_period", "sector"),
    ]:
        result = analysis.chi_square_test(test_df, variable1, variable2)

        if "error" in result:
            chi_rows.append(
                {
                    "Relationship": relationship,
                    "Chi-square": None,
                    "df": None,
                    "p-value": None,
                    "Cramér's V": None,
                    "Conclusion": "Not testable in this selection",
                }
            )
        else:
            chi_rows.append(
                {
                    "Relationship": relationship,
                    "Chi-square": round(result["chi2"], 3),
                    "df": result["dof"],
                    "p-value": f"{result['p_value']:.6f}",
                    "Cramér's V": round(result["cramers_v"], 3),
                    "Conclusion": (
                        "Significant association"
                        if result["significant"]
                        else "No significant association"
                    ),
                }
            )
            chi_tables[relationship] = result["table"]

    table(pd.DataFrame(chi_rows), "table13_chi_square.csv")

    if chi_tables:
        with st.expander("Contingency tables"):
            choice = st.selectbox("Relationship", list(chi_tables.keys()))
            st.dataframe(chi_tables[choice], use_container_width=True)

    st.caption(
        "Because the mix of sectors and breach methods itself changes across the "
        "regulatory boundaries, these comparisons are not a clean before-and-after "
        "experiment (section 4.9)."
    )

    st.divider()

    # --- Kruskal-Wallis ---
    st.subheader("Kruskal–Wallis — severity across methods and sectors")
    st.caption("Table 14")

    kruskal_rows = []
    for grouping, column, min_obs in [
        ("Breach method", "method", None),
        ("Sector", "sector", analysis.MIN_SECTOR_OBSERVATIONS),
    ]:
        result = analysis.kruskal_test(test_df, column, min_obs)

        if "error" in result:
            kruskal_rows.append(
                {
                    "Grouping variable": grouping,
                    "H statistic": None,
                    "p-value": None,
                    "Groups": None,
                    "Conclusion": result["error"],
                }
            )
        else:
            kruskal_rows.append(
                {
                    "Grouping variable": grouping,
                    "H statistic": round(result["h_statistic"], 3),
                    "p-value": f"{result['p_value']:.6f}",
                    "Groups": result["n_groups"],
                    "Conclusion": result["conclusion"],
                }
            )

    table(pd.DataFrame(kruskal_rows), "table14_kruskal_wallis.csv")
    st.caption(
        f"The sector test is restricted to sectors with at least "
        f"{analysis.MIN_SECTOR_OBSERVATIONS} observations, matching the analysis."
    )

    st.divider()

    # --- OLS ---
    st.subheader("OLS regression — log-transformed breach severity")
    st.caption("Tables 15–16 and Figure 14.")
    st.code(analysis.OLS_FORMULA, language="text")

    model = analysis.fit_ols(test_df)

    if model is None:
        st.info(
            "The regression could not be estimated for this selection — it needs "
            "enough observations and variation across sectors and methods."
        )
    else:
        left, right = st.columns([1, 2])

        with left:
            st.markdown("**Model statistics**")
            table(analysis.ols_summary_stats(model), "table15_ols_statistics.csv")

        with right:
            st.markdown("**Coefficients**")
            coefficients = analysis.ols_coefficient_table(model)

            if st.checkbox("Show significant predictors only (p < 0.05)", value=True):
                display_df = coefficients[coefficients["Significant"]]
            else:
                display_df = coefficients

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Coefficient": st.column_config.NumberColumn(format="%.3f"),
                    "Std_Error": st.column_config.NumberColumn(format="%.3f"),
                    "t_value": st.column_config.NumberColumn(format="%.3f"),
                    "p_value": st.column_config.NumberColumn(format="%.4f"),
                    "CI_Lower": st.column_config.NumberColumn(format="%.3f"),
                    "CI_Upper": st.column_config.NumberColumn(format="%.3f"),
                },
            )
            download(coefficients, "⬇ Download all coefficients (CSV)", "table16_ols_coefficients.csv")

        show_figure(analysis.fig_residuals(model))
        st.caption("Figure 14 — residuals versus fitted values.")

        st.error(
            "**Interpret with caution (section 4.11).** The residuals depart from "
            "normality and the condition number is very large, indicating possible "
            "multicollinearity and numerical instability. Several sector categories "
            "are very small, giving imprecise estimates. This model evidences "
            "**association only** — it does not estimate regulatory effectiveness.",
            icon="⚠️",
        )


# ---------------------------------------------------------------------------
# Tab 7 - ML models
# ---------------------------------------------------------------------------

with tabs[6]:
    st.header("Machine-learning classification of high-severity breaches")
    section_note("Corresponds to sections 4.12–4.13 (Tables 17–19, Figures 15–23).")

    st.info(
        "Models are always trained on the **full analytical dataset**, never the "
        "sidebar filters — the split and threshold must stay fixed to remain "
        "comparable with the dissertation.",
        icon="🔒",
    )

    signature = f"{uploaded.name}|{len(analysis_df)}|{analysis_df['records_lost'].sum():.0f}"

    try:
        ml = get_models(signature, analysis_df)
    except Exception as exc:
        st.error(f"Model preparation failed: {type(exc).__name__}: {exc}")
        st.stop()

    status_messages = {
        "loaded": "Using **saved model artifacts** from `models/` "
        "(same pipeline as the offline analysis).",
        "trained": "No matching artifacts found — models were **trained now** and "
        "saved to `models/` for reuse.",
    }
    st.caption(status_messages.get(ml["artifact_status"], ""))

    if not ml["has_xgboost"]:
        st.warning(
            "`xgboost` is not installed, so four models are compared instead of five."
        )

    metric_row(
        [
            ("High-severity threshold", f"{ml['threshold']:,.0f} records"),
            ("Training observations", f"{len(ml['X_train']):,}"),
            ("Testing observations", f"{len(ml['X_test']):,}"),
            ("Models compared", f"{len(ml['pipelines'])}"),
        ]
    )

    st.caption(
        "The threshold is the 75th percentile of the **training** severities only, so "
        "it does not leak into the test set (sections 3.6, 4.2)."
    )

    st.subheader("Class distribution")
    st.caption("Table 17")
    table(ml["class_distribution"], "table17_class_distribution.csv")

    st.divider()

    st.subheader("Comparative model performance")
    st.caption("Table 18")

    metrics = ml["metrics"]
    st.dataframe(
        metrics.style.format(
            {
                column: "{:.3f}"
                for column in metrics.columns
                if column != "Model"
            }
        ).highlight_max(
            subset=[c for c in metrics.columns if c != "Model"], color="#d4f5dd"
        ),
        use_container_width=True,
        hide_index=True,
    )
    download(metrics, "⬇ Download this table (CSV)", "table18_model_performance.csv")

    st.info(
        f"**Why {analysis.PRIMARY_MODEL} is used for interpretation.** With a minority "
        "high-severity class, accuracy is misleading — Logistic Regression scores well "
        "on accuracy and ROC-AUC while recalling very few high-severity incidents. "
        f"{analysis.PRIMARY_MODEL} gives the best balance of recall, F1 and PR-AUC, "
        "which matches the objective of *finding* high-severity events (section 4.12).",
        icon="🎯",
    )

    st.divider()

    st.subheader("Confusion matrix")
    model_names = list(ml["pipelines"].keys())
    default_index = (
        model_names.index(analysis.PRIMARY_MODEL)
        if analysis.PRIMARY_MODEL in model_names
        else 0
    )
    chosen = st.selectbox("Model", model_names, index=default_index)

    left, right = st.columns([1, 1])
    with left:
        show_figure(
            analysis.fig_confusion_matrix(
                ml["y_test"], ml["predictions"][chosen], chosen
            )
        )
        st.caption("Figures 15–19")
    with right:
        st.markdown("**Classification report**")
        st.code(
            analysis.classification_text_report(
                ml["y_test"], ml["predictions"][chosen]
            ),
            language="text",
        )

    st.divider()

    st.subheader("Discrimination curves")
    left, right = st.columns(2)
    with left:
        show_figure(analysis.fig_roc_curves(ml["y_test"], ml["probabilities"]))
        st.caption("Figure 20 — ROC curves.")
    with right:
        show_figure(analysis.fig_pr_curves(ml["y_test"], ml["probabilities"]))
        st.caption(
            "Figure 21 — precision-recall curves, the more informative view for an "
            "imbalanced minority class."
        )

    if "rf_importance" in ml:
        st.divider()
        st.subheader(f"{analysis.PRIMARY_MODEL} interpretation")

        left, right = st.columns([2, 1])

        with left:
            show_figure(analysis.fig_rf_importance(ml["rf_importance"], top_n=15))
            st.caption("Figure 22 — top 15 predictors.")

        with right:
            st.markdown("**Feature importance**")
            st.caption("Table 19")
            st.dataframe(
                ml["rf_importance"],
                use_container_width=True,
                hide_index=True,
                height=420,
                column_config={
                    "Importance": st.column_config.NumberColumn(format="%.3f")
                },
            )
            download(
                ml["rf_importance"],
                "⬇ Download this table (CSV)",
                "table19_rf_importance.csv",
            )

        st.caption(
            "Year dominates, followed by sector indicators, with regulatory-period "
            "flags contributing less — so regulatory period alone does not explain the "
            "model's decisions (section 4.13)."
        )

        st.markdown("**SHAP summary**")
        with st.spinner("Computing SHAP values…"):
            shap_fig, shap_error = analysis.fig_shap_summary(
                ml["pipelines"][analysis.PRIMARY_MODEL],
                ml["X_test"],
                ml["feature_names"],
            )

        if shap_fig is not None:
            show_figure(shap_fig)
            st.caption(
                "Figure 23 — SHAP values for the high-severity class. High feature "
                "importance means the model *uses* that feature, not that changing it "
                "would change breach outcomes."
            )
        else:
            st.info(f"SHAP summary unavailable — {shap_error}")

    st.error(
        "**Not an operational detector.** The selected model does not identify every "
        "high-severity incident. Treat it as illustrative decision support for "
        "prioritising investigation, subject to ongoing validation against new data "
        "(sections 4.12, 5.3.6).",
        icon="⚠️",
    )


# ---------------------------------------------------------------------------
# Tab 8 - Predict severity
# ---------------------------------------------------------------------------

with tabs[7]:
    st.header("Predict breach severity")
    section_note(
        "Fulfils Objective 5 and the prediction function described in section 1.5."
    )

    st.markdown(
        "Describe a hypothetical breach event and score it against a saved model. "
        "The regulatory-period indicators are derived from the year automatically, "
        "exactly as during training."
    )

    signature = f"{uploaded.name}|{len(analysis_df)}|{analysis_df['records_lost'].sum():.0f}"

    try:
        ml = get_models(signature, analysis_df)
    except Exception as exc:
        st.error(f"Model preparation failed: {type(exc).__name__}: {exc}")
        st.stop()

    model_names = list(ml["pipelines"].keys())
    default_index = (
        model_names.index(analysis.PRIMARY_MODEL)
        if analysis.PRIMARY_MODEL in model_names
        else 0
    )

    with st.form("prediction_form"):
        columns = st.columns(4)

        chosen_model = columns[0].selectbox(
            "Model", model_names, index=default_index
        )
        input_year = columns[1].number_input(
            "Year",
            min_value=int(year_min),
            max_value=int(year_max) + 10,
            value=int(year_max),
            step=1,
            help="Years beyond the observed range extrapolate — treat with care.",
        )
        input_sector = columns[2].selectbox("Sector", ml["sectors"])
        input_method = columns[3].selectbox("Breach method", ml["methods"])

        submitted = st.form_submit_button("Predict severity", type="primary")

    if submitted:
        result = analysis.predict_single(
            ml["pipelines"][chosen_model], input_year, input_sector, input_method
        )

        model_metrics = ml["metrics"].set_index("Model").loc[chosen_model]
        probability = result["probability_high"]

        st.divider()

        left, right = st.columns([1, 1])

        with left:
            st.subheader("Prediction")
            if result["predicted_class"] == 1:
                st.error(
                    f"### ⚠️ {result['predicted_label']}\n"
                    f"At or above the {ml['threshold']:,.0f}-record threshold."
                )
            else:
                st.success(
                    f"### ✅ {result['predicted_label']}\n"
                    f"Below the {ml['threshold']:,.0f}-record threshold."
                )

            st.metric("Probability of high severity", f"{probability:.1%}")
            st.progress(min(max(probability, 0.0), 1.0))

        with right:
            st.subheader("How much to trust this")
            st.caption(
                f"Test-set performance of **{chosen_model}** for the high-severity "
                "class:"
            )
            metric_row(
                [
                    ("Recall", f"{model_metrics['Recall']:.1%}"),
                    ("Precision", f"{model_metrics['Precision']:.1%}"),
                ]
            )
            st.caption(
                f"Recall of {model_metrics['Recall']:.1%} means the model misses "
                f"roughly {1 - model_metrics['Recall']:.0%} of genuine high-severity "
                "incidents. A 'Normal severity' prediction is therefore **not** an "
                "assurance of low risk."
            )

        with st.expander("Feature vector passed to the model"):
            st.dataframe(result["features"], use_container_width=True, hide_index=True)

        st.error(
            "**Decision support only.** This prediction must not replace professional "
            "cybersecurity judgement. Use it to help prioritise investigation and "
            "resource allocation, and validate the model against new data before any "
            "operational use (section 5.3.6).",
            icon="⚠️",
        )
    else:
        st.info("Set the event characteristics above, then select **Predict severity**.")
