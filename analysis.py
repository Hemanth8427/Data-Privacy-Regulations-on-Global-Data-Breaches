"""
Analysis engine for the data-breach / privacy-regulation dissertation dashboard.

This module is a faithful port of ``Ref_Hemanth.ipynb``. Every cleaning rule,
aggregation, statistical test, model hyper-parameter and random seed matches the
notebook, so the figures and tables produced here reproduce Chapter 4 of the
dissertation exactly.

Deliberately contains **no Streamlit imports** so that the same functions serve
``app.py`` (dashboard) and ``train_models.py`` (offline artifact builder), and so
the numbers can be checked independently of the UI.

Notebook provenance is noted against each section as "cells N-M".
"""

from __future__ import annotations

import re
import warnings
from typing import Iterable, Sequence

# Matches the notebook's cell 0, and keeps seaborn/statsmodels chatter out of the UI.
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib

# Non-interactive backend: required for headless Streamlit / CLI use.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu

import statsmodels.formula.api as smf

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:  # XGBoost is optional: the dashboard degrades to four models without it.
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except Exception:  # pragma: no cover - environment dependent
    HAS_XGBOOST = False

try:
    import shap

    HAS_SHAP = True
except Exception:  # pragma: no cover - environment dependent
    HAS_SHAP = False


# ---------------------------------------------------------------------------
# Constants (dissertation sections 3.6 and 3.8)
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20
HIGH_SEVERITY_QUANTILE = 0.75

GDPR_YEAR = 2018  # GDPR in force 25 May 2018
CCPA_YEAR = 2020  # CCPA in force 1 January 2020

#: Minimum group size for the sector boxplot and the sector Kruskal-Wallis test
#: (notebook cells 35 and 56).
MIN_SECTOR_OBSERVATIONS = 5

ANALYSIS_COLUMNS = [
    "organisation",
    "records_lost",
    "year",
    "sector",
    "method",
    "data_sensitivity",
]

FEATURES = ["year", "sector", "method", "gdpr_post", "ccpa_post"]
NUMERIC_FEATURES = ["year", "gdpr_post", "ccpa_post"]
CATEGORICAL_FEATURES = ["sector", "method"]

#: Documented category mapping (dissertation section 3.5, notebook cell 14).
SECTOR_MAPPING = {"financial": "finance"}

CLASS_LABELS = ["Normal Severity", "High Severity"]

MODEL_ORDER = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "Gradient Boosting",
    "XGBoost",
]

#: Model used for detailed interpretation in section 4.13.
PRIMARY_MODEL = "Random Forest"


def apply_plot_style() -> None:
    """Reproduce the notebook's plotting configuration (cell 1)."""
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["figure.dpi"] = 120


apply_plot_style()


# ---------------------------------------------------------------------------
# Data loading and cleaning (notebook cells 2-23)
# ---------------------------------------------------------------------------


def load_raw(source) -> pd.DataFrame:
    """Read the raw breach CSV.

    ``source`` may be a path or any file-like object (e.g. a Streamlit upload).
    """
    return pd.read_csv(source)


def clean_records_lost(value):
    """Convert a free-text ``records lost`` entry to a float (notebook cell 12).

    Handles thousands separators, approximation markers and ``k``/``m``/``b``
    magnitude suffixes. Unparseable values become ``NaN``.
    """
    if pd.isna(value):
        return np.nan

    value = str(value).strip().lower()

    # Remove thousands separators and whitespace
    value = value.replace(",", "")
    value = value.replace(" ", "")

    # Remove approximation markers
    value = value.replace("~", "")
    value = value.replace("approximately", "")
    value = value.replace("approx", "")

    match = re.match(r"^([0-9]*\.?[0-9]+)(m|million|b|billion|k|thousand)?$", value)

    if match:
        number = float(match.group(1))
        unit = match.group(2)

        if unit in ["m", "million"]:
            number *= 1_000_000
        elif unit in ["b", "billion"]:
            number *= 1_000_000_000
        elif unit in ["k", "thousand"]:
            number *= 1_000

        return number

    # Otherwise attempt direct numeric conversion
    try:
        return float(value)
    except Exception:
        return np.nan


def clean_dataframe(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply the notebook's cleaning chain (cells 9-17).

    Returns the cleaned frame plus a metadata dict recording what each stage did,
    which feeds the cleaning-summary table (Table 1).
    """
    meta: dict = {"original_rows": len(df_raw)}

    # Cell 9 - drop the embedded description row shipped inside the source CSV
    df = df_raw.iloc[1:].copy()
    meta["rows_after_description_removal"] = len(df)

    # Cell 10 - normalise column names
    df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)

    # Cell 11 - drop completely empty columns
    empty_columns = [col for col in df.columns if df[col].isna().all()]
    df = df.drop(columns=empty_columns)
    meta["empty_columns"] = empty_columns

    # Cell 12 - numeric severity measure
    if "records_lost" in df.columns:
        df["records_lost"] = df["records_lost"].apply(clean_records_lost)

    # Cell 13 - year as nullable integer
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    # Cell 14 - standardise sector labels
    if "sector" in df.columns:
        df["sector"] = df["sector"].astype("string").str.strip().str.lower()
        df["sector"] = df["sector"].replace(SECTOR_MAPPING)

    # Cell 15 - standardise breach-method labels
    if "method" in df.columns:
        df["method"] = df["method"].astype("string").str.strip().str.lower()

    # Cell 16 - ordinal data-sensitivity score
    if "data_sensitivity" in df.columns:
        df["data_sensitivity"] = pd.to_numeric(df["data_sensitivity"], errors="coerce")

    # Cell 17 - remove duplicate observations
    duplicates = int(df.duplicated().sum())
    if duplicates > 0:
        df = df.drop_duplicates().copy()
    meta["duplicates_removed"] = duplicates
    meta["rows_after_deduplication"] = len(df)

    return df, meta


def build_analysis_df(df: pd.DataFrame) -> pd.DataFrame:
    """Select the analytical columns and drop rows missing essentials (cell 19)."""
    available = [col for col in ANALYSIS_COLUMNS if col in df.columns]
    analysis_df = df[available].copy()

    required = [
        col
        for col in ["records_lost", "year", "sector", "method"]
        if col in analysis_df.columns
    ]
    analysis_df = analysis_df.dropna(subset=required).copy()

    return analysis_df


def add_derived_columns(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """Add regulatory-period, log-severity and exploratory severity flags.

    Notebook cells 20-22. The regulatory indicators are **temporal markers**, not
    determinations of legal coverage (dissertation sections 1.4 and 3.6).
    """
    df = analysis_df.copy()

    # Cell 20 - regulatory-period classification
    df["gdpr_period"] = np.where(df["year"] < GDPR_YEAR, "Pre-GDPR", "Post-GDPR")
    df["ccpa_period"] = np.where(df["year"] < CCPA_YEAR, "Pre-CCPA", "Post-CCPA")
    df["gdpr_post"] = (df["year"] >= GDPR_YEAR).astype(int)
    df["ccpa_post"] = (df["year"] >= CCPA_YEAR).astype(int)

    # Cell 21 - log transform to damp the extreme right tail
    df["log_records_lost"] = np.log1p(df["records_lost"])

    # Cell 22 - exploratory (descriptive only) high-severity flag at the overall p75.
    # NOTE: the modelling threshold is derived from the *training* split instead;
    # see split_and_threshold().
    overall_q75 = df["records_lost"].quantile(HIGH_SEVERITY_QUANTILE)
    df["high_severity_exploratory"] = (df["records_lost"] >= overall_q75).astype(int)

    return df


def cleaning_summary_table(meta: dict, final_rows: int) -> pd.DataFrame:
    """Build Table 1 - data-cleaning stages and final analytical sample (cell 23)."""
    original = meta["original_rows"]
    after_description = meta["rows_after_description_removal"]
    after_dedup = meta.get("rows_after_deduplication", after_description)

    return pd.DataFrame(
        {
            "Stage": [
                "Original CSV",
                "Remove embedded description row",
                "Remove completely empty columns",
                "Remove duplicate rows",
                "Final analytical dataset",
            ],
            "Records": [
                original,
                after_description,
                after_description,
                after_dedup,
                final_rows,
            ],
        }
    )


def prepare_data(source) -> dict:
    """Run the full pipeline: raw CSV in, analysis-ready bundle out."""
    df_raw = load_raw(source)
    cleaned, meta = clean_dataframe(df_raw)
    analysis_df = add_derived_columns(build_analysis_df(cleaned))

    return {
        "raw": df_raw,
        "cleaned": cleaned,
        "analysis": analysis_df,
        "meta": meta,
        "cleaning_summary": cleaning_summary_table(meta, len(analysis_df)),
    }


def data_quality_table(df: pd.DataFrame) -> pd.DataFrame:
    """Missing-value profile of the cleaned frame (notebook cell 18)."""
    return (
        pd.DataFrame(
            {
                "Variable": df.columns,
                "Missing_Count": df.isna().sum().values,
                "Missing_Percentage": df.isna().mean().values * 100,
            }
        )
        .sort_values("Missing_Percentage", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Descriptive statistics (notebook cells 24-42, 46-48)
# ---------------------------------------------------------------------------


def severity_descriptive_table(df: pd.DataFrame) -> pd.DataFrame:
    """Table 2 - descriptive statistics for records lost (cell 24)."""
    series = df["records_lost"]

    return pd.DataFrame(
        {
            "Statistic": [
                "Number of observations",
                "Mean",
                "Standard deviation",
                "Minimum",
                "25th percentile",
                "Median",
                "75th percentile",
                "90th percentile",
                "95th percentile",
                "99th percentile",
                "Maximum",
            ],
            "Records lost": [
                series.count(),
                series.mean(),
                series.std(),
                series.min(),
                series.quantile(0.25),
                series.median(),
                series.quantile(0.75),
                series.quantile(0.90),
                series.quantile(0.95),
                series.quantile(0.99),
                series.max(),
            ],
        }
    )


def annual_breach_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Table 3 - annual frequency of recorded breach events (cell 25)."""
    return df.groupby("year").size().reset_index(name="breach_count")


def _severity_aggregation(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Shared severity aggregation used for year, sector, method and period."""
    return (
        df.groupby(group_col)
        .agg(
            breach_count=("records_lost", "count"),
            total_records_lost=("records_lost", "sum"),
            mean_records_lost=("records_lost", "mean"),
            median_records_lost=("records_lost", "median"),
            q1_records_lost=("records_lost", lambda x: x.quantile(0.25)),
            q3_records_lost=("records_lost", lambda x: x.quantile(0.75)),
        )
        .reset_index()
    )


def annual_severity_table(df: pd.DataFrame) -> pd.DataFrame:
    """Table 4 - annual breach severity summary (cell 27)."""
    return _severity_aggregation(df, "year")


def _frequency_table(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Frequency + percentage table for a categorical column (cells 32, 36)."""
    table = df[column].value_counts().reset_index()
    table.columns = [column, "breach_count"]
    table["percentage"] = table["breach_count"] / len(df) * 100
    return table


def sector_frequency_table(df: pd.DataFrame) -> pd.DataFrame:
    """Table 5 - frequency of breach events by sector (cell 32)."""
    return _frequency_table(df, "sector")


def sector_severity_table(df: pd.DataFrame) -> pd.DataFrame:
    """Table 6 - breach severity by sector (cell 34)."""
    table = _severity_aggregation(df, "sector")
    table["percentage"] = table["breach_count"] / len(df) * 100
    return table.sort_values("breach_count", ascending=False).reset_index(drop=True)


def method_frequency_table(df: pd.DataFrame) -> pd.DataFrame:
    """Table 7 - frequency of breach events by method (cell 36)."""
    return _frequency_table(df, "method")


def method_severity_table(df: pd.DataFrame) -> pd.DataFrame:
    """Table 8 - breach severity by method (cell 38)."""
    table = _severity_aggregation(df, "method")
    table["percentage"] = table["breach_count"] / len(df) * 100
    return table


def period_summary_table(df: pd.DataFrame, period_col: str) -> pd.DataFrame:
    """Tables 9 and 11 - descriptive comparison across a regulatory period."""
    table = _severity_aggregation(df, period_col)
    table["percentage_of_breaches"] = table["breach_count"] / len(df) * 100
    return table


def major_sectors(df: pd.DataFrame, min_observations: int = MIN_SECTOR_OBSERVATIONS):
    """Sectors with enough observations to compare (cells 35, 56)."""
    counts = df["sector"].value_counts()
    return counts[counts >= min_observations].index


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_annual_frequency(annual_breaches: pd.DataFrame):
    """Figure 1 - annual number of recorded data breaches (cell 26)."""
    fig, ax = plt.subplots(figsize=(12, 6))

    sns.lineplot(data=annual_breaches, x="year", y="breach_count", marker="o", ax=ax)

    ax.axvline(x=GDPR_YEAR, linestyle="--", color="tab:red", label=f"GDPR: {GDPR_YEAR}")
    ax.axvline(
        x=CCPA_YEAR, linestyle="--", color="tab:green", label=f"CCPA: {CCPA_YEAR}"
    )

    ax.set_title("Annual Number of Recorded Data Breaches")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Breach Events")
    # Years are integers: stop matplotlib labelling ticks as 2007.5 etc.
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.legend()
    fig.tight_layout()
    return fig


def fig_total_records_by_year(annual_severity: pd.DataFrame, log_scale: bool = False):
    """Figures 2 and 3 - total records lost by year, linear or log (cells 28-29)."""
    fig, ax = plt.subplots(figsize=(12, 6))

    sns.barplot(data=annual_severity, x="year", y="total_records_lost", ax=ax)

    if log_scale:
        ax.set_yscale("log")
        ax.set_title("Total Records Lost by Year — Logarithmic Scale")
        ax.set_ylabel("Total Records Lost (log scale)")
    else:
        ax.set_title("Total Records Lost by Year")
        ax.set_ylabel("Total Records Lost")

    ax.set_xlabel("Year")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    return fig


def fig_severity_histogram(df: pd.DataFrame):
    """Figure 4 - distribution of log-transformed severity (cell 30)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.histplot(data=df, x="log_records_lost", bins=30, kde=True, ax=ax)

    ax.set_title("Distribution of Log-Transformed Records Lost")
    ax.set_xlabel("Log(Records Lost + 1)")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    return fig


def fig_severity_boxplot(df: pd.DataFrame):
    """Figure 5 - boxplot of log-transformed severity (cell 31)."""
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.boxplot(y=df["log_records_lost"], ax=ax)

    ax.set_title("Distribution of Log-Transformed Breach Severity")
    ax.set_ylabel("Log(Records Lost + 1)")
    fig.tight_layout()
    return fig


def fig_sector_frequency(sector_frequency: pd.DataFrame):
    """Figure 6 - number of data breaches by sector (cell 33)."""
    plot_df = sector_frequency.sort_values("breach_count", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(data=plot_df, x="breach_count", y="sector", ax=ax)

    ax.set_title("Number of Data Breaches by Sector")
    ax.set_xlabel("Number of Breach Events")
    ax.set_ylabel("Sector")
    fig.tight_layout()
    return fig


def fig_sector_severity_box(
    df: pd.DataFrame, min_observations: int = MIN_SECTOR_OBSERVATIONS
):
    """Figure 7 - severity across sectors with sufficient observations (cell 35)."""
    eligible = major_sectors(df, min_observations)
    plot_df = df[df["sector"].isin(eligible)].copy()

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(data=plot_df, x="log_records_lost", y="sector", ax=ax)

    ax.set_title("Distribution of Breach Severity Across Major Sectors")
    ax.set_xlabel("Log(Records Lost + 1)")
    ax.set_ylabel("Sector")
    fig.tight_layout()
    return fig


def fig_method_frequency(method_frequency: pd.DataFrame):
    """Figure 8 - number of breaches by breach method (cell 37)."""
    plot_df = method_frequency.sort_values("breach_count", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=plot_df, x="breach_count", y="method", ax=ax)

    ax.set_title("Number of Breaches by Breach Method")
    ax.set_xlabel("Number of Breach Events")
    ax.set_ylabel("Breach Method")
    fig.tight_layout()
    return fig


def fig_method_severity_box(df: pd.DataFrame):
    """Figure 9 - severity distribution by breach method (cell 39)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.boxplot(data=df, x="method", y="log_records_lost", ax=ax)

    ax.set_title("Distribution of Breach Severity by Breach Method")
    ax.set_xlabel("Breach Method")
    ax.set_ylabel("Log(Records Lost + 1)")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    return fig


def fig_period_counts(summary: pd.DataFrame, period_col: str, label: str):
    """Figures 10 and 12 - recorded breach events before/after a regulation."""
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.barplot(data=summary, x=period_col, y="breach_count", ax=ax)

    ax.set_title(f"Recorded Breach Events Before and After {label}")
    ax.set_xlabel("Regulatory Period")
    ax.set_ylabel("Number of Breach Events")
    fig.tight_layout()
    return fig


def fig_period_severity_box(df: pd.DataFrame, period_col: str, label: str):
    """Figures 11 and 13 - severity distribution before/after a regulation."""
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.boxplot(data=df, x=period_col, y="log_records_lost", ax=ax)

    ax.set_title(f"Distribution of Breach Severity Before and After {label}")
    ax.set_xlabel("Regulatory Period")
    ax.set_ylabel("Log(Records Lost + 1)")
    fig.tight_layout()
    return fig


def fig_residuals(model):
    """Figure 14 - OLS residuals versus fitted values (cell 60)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.scatterplot(x=model.fittedvalues, y=model.resid, ax=ax)
    ax.axhline(0, linestyle="--", color="tab:red")

    ax.set_title("Residuals vs Fitted Values")
    ax.set_xlabel("Fitted Values")
    ax.set_ylabel("Residuals")
    fig.tight_layout()
    return fig


def fig_confusion_matrix(y_true, y_pred, model_name: str):
    """Figures 15-19 - per-model confusion matrix (cell 73)."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cbar=False,
        xticklabels=["Normal", "High"],
        yticklabels=["Normal", "High"],
        ax=ax,
    )

    ax.set_title(f"Confusion Matrix — {model_name}")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    fig.tight_layout()
    return fig


def fig_roc_curves(y_true, probabilities: dict):
    """Figure 20 - ROC curves for all models (cell 74)."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for model_name, probability in probabilities.items():
        fpr, tpr, _ = roc_curve(y_true, probability)
        ax.plot(fpr, tpr, label=f"{model_name} (AUC = {auc(fpr, tpr):.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")

    ax.set_title("ROC Curves for High-Severity Classification")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    fig.tight_layout()
    return fig


def fig_pr_curves(y_true, probabilities: dict):
    """Figure 21 - precision-recall curves for all models (cell 75)."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for model_name, probability in probabilities.items():
        precision, recall, _ = precision_recall_curve(y_true, probability)
        ap = average_precision_score(y_true, probability)
        ax.plot(recall, precision, label=f"{model_name} (AP = {ap:.3f})")

    ax.set_title("Precision-Recall Curves for High-Severity Classification")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    fig.tight_layout()
    return fig


def fig_rf_importance(importance: pd.DataFrame, top_n: int = 15):
    """Figure 22 - top Random Forest predictors (cell 77)."""
    plot_df = importance.head(top_n).sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(data=plot_df, x="Importance", y="Feature", ax=ax)

    ax.set_title("Top Predictors of High-Severity Data Breaches")
    ax.set_xlabel("Random Forest Feature Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    return fig


def fig_shap_summary(
    pipeline, X_test: pd.DataFrame, feature_names: Sequence[str]
) -> tuple:
    """Figure 23 - SHAP summary for the Random Forest model (cells 78-79).

    Returns ``(figure, error_message)``. On success the error is ``None``; on
    failure the figure is ``None`` and the caller can display the reason rather
    than the tab breaking.
    """
    if not HAS_SHAP:
        return None, "The 'shap' package is not installed."

    try:
        transformed = pipeline.named_steps["preprocessor"].transform(X_test)
        explainer = shap.TreeExplainer(pipeline.named_steps["model"])
        shap_values = explainer(transformed)

        # A binary classifier yields values shaped (n_samples, n_features, n_classes).
        # Take the positive class, which is the high-severity outcome being explained.
        values = shap_values[..., 1] if shap_values.values.ndim == 3 else shap_values

        fig = plt.figure(figsize=(11, 7))
        shap.summary_plot(
            values,
            transformed,
            # Must be an array: shap's legacy plot indexes this with an array of
            # sort positions, which fails on a plain list.
            feature_names=np.array(list(feature_names)),
            show=False,
        )
        plt.title("SHAP Summary — Random Forest High-Severity Prediction")
        plt.tight_layout()
        return fig, None
    except Exception as exc:  # pragma: no cover - depends on shap version
        return None, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Inferential statistics (notebook cells 43-56)
# ---------------------------------------------------------------------------


def interpret_effect_size_r(r: float) -> str:
    """Cohen-style bands for the rank-biserial effect size (cell 45)."""
    if r < 0.10:
        return "Negligible"
    elif r < 0.30:
        return "Small"
    elif r < 0.50:
        return "Medium"
    else:
        return "Large"


def mann_whitney_test(
    df: pd.DataFrame, period_col: str, pre_label: str, post_label: str
) -> dict:
    """Mann-Whitney U on log severity across a regulatory boundary (cells 43-45).

    Non-parametric because severity is heavily right-skewed (section 3.7).
    Returns ``{"error": ...}`` when a group is too small to test.
    """
    pre = df.loc[df[period_col] == pre_label, "log_records_lost"]
    post = df.loc[df[period_col] == post_label, "log_records_lost"]

    if len(pre) < 2 or len(post) < 2:
        return {
            "error": (
                "Not enough observations in the current selection: "
                f"{pre_label} n={len(pre)}, {post_label} n={len(post)} "
                "(at least 2 per group required)."
            )
        }

    u_statistic, p_value = mannwhitneyu(pre, post, alternative="two-sided")

    # Effect size r = |z| / sqrt(N), with z recovered from the two-sided p-value.
    n_total = len(pre) + len(post)
    z_score = abs(stats.norm.ppf(p_value / 2))
    r = z_score / np.sqrt(n_total)

    return {
        "u_statistic": float(u_statistic),
        "p_value": float(p_value),
        "effect_size_r": float(r),
        "effect_size_label": interpret_effect_size_r(r),
        "n_pre": int(len(pre)),
        "n_post": int(len(post)),
        "significant": bool(p_value < 0.05),
        "conclusion": (
            "Statistically significant difference"
            if p_value < 0.05
            else "No statistically significant difference"
        ),
    }


def chi_square_test(data: pd.DataFrame, variable1: str, variable2: str) -> dict:
    """Chi-square test of independence with Cramer's V (cell 50).

    Used for Table 13 - regulatory period against breach method and sector.
    """
    contingency_table = pd.crosstab(data[variable1], data[variable2])

    if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
        return {
            "error": (
                "The current selection does not contain at least two categories "
                f"of both '{variable1}' and '{variable2}'."
            )
        }

    chi2, p, dof, _expected = chi2_contingency(contingency_table)

    n = contingency_table.to_numpy().sum()
    min_dim = min(contingency_table.shape) - 1
    cramers_v = float(np.sqrt(chi2 / (n * min_dim))) if min_dim > 0 else np.nan

    return {
        "table": contingency_table,
        "chi2": float(chi2),
        "p_value": float(p),
        "dof": int(dof),
        "cramers_v": cramers_v,
        "significant": bool(p < 0.05),
    }


def kruskal_test(
    df: pd.DataFrame, group_col: str, min_observations: int | None = None
) -> dict:
    """Kruskal-Wallis H test of log severity across groups (cells 55-56).

    ``min_observations`` mirrors the notebook's sector filter (n >= 5); pass
    ``None`` for the unfiltered method test.
    """
    data = df

    if min_observations is not None:
        counts = data[group_col].value_counts()
        eligible = counts[counts >= min_observations].index
        data = data[data[group_col].isin(eligible)].copy()

    groups = [
        group["log_records_lost"].values for _name, group in data.groupby(group_col)
    ]
    groups = [g for g in groups if len(g) > 0]

    if len(groups) < 2:
        return {
            "error": (
                f"At least two '{group_col}' groups are required; the current "
                f"selection has {len(groups)}."
            )
        }

    h_statistic, p_value = kruskal(*groups)

    return {
        "h_statistic": float(h_statistic),
        "p_value": float(p_value),
        "n_groups": len(groups),
        "n_observations": int(len(data)),
        "significant": bool(p_value < 0.05),
        "conclusion": (
            "Severity differs significantly"
            if p_value < 0.05
            else "No statistically significant difference detected"
        ),
    }


# ---------------------------------------------------------------------------
# OLS regression (notebook cells 57-60)
# ---------------------------------------------------------------------------

OLS_FORMULA = (
    "log_records_lost ~ year + C(gdpr_period) + C(ccpa_period) "
    "+ C(sector) + C(method)"
)


def fit_ols(df: pd.DataFrame):
    """Fit the log-severity OLS model (cells 57-58).

    Returns ``None`` if there are too few observations for the design matrix.
    Coefficients are **associations**, not causal effects (section 4.11).
    """
    reg_df = df.copy()

    for col in ["sector", "method", "gdpr_period", "ccpa_period"]:
        reg_df[col] = reg_df[col].astype("category")

    if len(reg_df) < 20:
        return None

    try:
        return smf.ols(formula=OLS_FORMULA, data=reg_df).fit()
    except Exception:
        return None


def ols_summary_stats(model) -> pd.DataFrame:
    """Table 15 - overall OLS model statistics."""
    return pd.DataFrame(
        {
            "Model statistic": [
                "Observations",
                "R-squared",
                "Adjusted R-squared",
                "F-statistic",
                "Prob. (F-statistic)",
                "Durbin-Watson",
                "Condition number",
            ],
            "Result": [
                f"{int(model.nobs)}",
                f"{model.rsquared:.3f}",
                f"{model.rsquared_adj:.3f}",
                f"{model.fvalue:.3f}",
                f"{model.f_pvalue:.3g}",
                f"{sm_durbin_watson(model.resid):.3f}",
                f"{model.condition_number:.3g}",
            ],
        }
    )


def sm_durbin_watson(residuals) -> float:
    """Durbin-Watson statistic (imported lazily to keep the header tidy)."""
    from statsmodels.stats.stattools import durbin_watson

    return float(durbin_watson(residuals))


def ols_coefficient_table(model) -> pd.DataFrame:
    """Table 16 - OLS coefficients with standard errors, p-values and CIs (cell 59)."""
    conf_int = model.conf_int()

    table = pd.DataFrame(
        {
            "Predictor": model.params.index,
            "Coefficient": model.params.values,
            "Std_Error": model.bse.values,
            "t_value": model.tvalues.values,
            "p_value": model.pvalues.values,
            "CI_Lower": conf_int[0].values,
            "CI_Upper": conf_int[1].values,
        }
    ).reset_index(drop=True)

    table["Significant"] = table["p_value"] < 0.05
    return table


# ---------------------------------------------------------------------------
# Machine learning (notebook cells 61-79)
# ---------------------------------------------------------------------------


def build_preprocessor() -> ColumnTransformer:
    """Preprocessing pipeline shared by all five models (cell 64).

    Median-impute + standardise numerics; mode-impute + one-hot the categoricals
    with ``handle_unknown="ignore"`` so unseen categories are tolerated at
    prediction time.
    """
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def build_pipelines() -> dict:
    """The five classifiers with the notebook's exact hyper-parameters (cells 65-69).

    Each gets its own ``build_preprocessor()`` instance so fitting one model never
    mutates another's encoder state.
    """
    pipelines = {
        "Logistic Regression": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
                ),
            ]
        ),
        "Decision Tree": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=5),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        random_state=RANDOM_STATE,
                        class_weight="balanced",
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=3,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

    if HAS_XGBOOST:
        pipelines["XGBoost"] = Pipeline(
            [
                ("preprocessor", build_preprocessor()),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=300,
                        learning_rate=0.05,
                        max_depth=4,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    return {name: pipelines[name] for name in MODEL_ORDER if name in pipelines}


def split_and_threshold(df: pd.DataFrame) -> dict:
    """Train/test split and the high-severity threshold (cells 61-63).

    The split is intentionally **not stratified**, matching the notebook, so the
    dashboard reproduces Tables 17-18 of the dissertation exactly. (Section 3.8 of
    the write-up describes a stratified split; the executed notebook does not
    stratify. The code follows the notebook.)

    The threshold is the 75th percentile of the **training** severities only, so it
    never leaks into the test set (sections 3.6 and 4.2).
    """
    X = df[FEATURES].copy()
    y_severity = df["records_lost"].copy()

    X_train, X_test, severity_train, severity_test = train_test_split(
        X, y_severity, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    threshold = severity_train.quantile(HIGH_SEVERITY_QUANTILE)

    y_train = (severity_train >= threshold).astype(int)
    y_test = (severity_test >= threshold).astype(int)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "threshold": float(threshold),
    }


def class_distribution_table(y_train, y_test) -> pd.DataFrame:
    """Table 17 - high-severity class distribution across the split."""
    return pd.DataFrame(
        {
            "Dataset": ["Training", "Testing"],
            "Normal severity": [int((y_train == 0).sum()), int((y_test == 0).sum())],
            "High severity": [int((y_train == 1).sum()), int((y_test == 1).sum())],
            "Total": [int(len(y_train)), int(len(y_test))],
        }
    )


def evaluate_model(model_name: str, y_true, y_pred, y_probability) -> dict:
    """Metric set from cell 70. Accuracy alone is inadequate here (section 3.8)."""
    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, y_probability),
        "PR_AUC": average_precision_score(y_true, y_probability),
    }


def rf_importance_table(pipeline) -> pd.DataFrame:
    """Table 19 - Random Forest feature importances (cell 76)."""
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = pipeline.named_steps["model"].feature_importances_

    return (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )


def fit_pipelines(X_train: pd.DataFrame, y_train) -> dict:
    """Fit all available classifiers on the training split (cells 65-69)."""
    pipelines = build_pipelines()

    for pipeline in pipelines.values():
        pipeline.fit(X_train, y_train)

    return pipelines


def assemble_bundle(pipelines: dict, split: dict, df: pd.DataFrame) -> dict:
    """Score fitted pipelines and package everything the dashboard needs.

    Kept separate from fitting so the app can evaluate **saved** pipelines loaded
    from ``models/`` without retraining (dissertation section 3.9).
    """
    X_test = split["X_test"]
    y_train, y_test = split["y_train"], split["y_test"]

    predictions: dict = {}
    probabilities: dict = {}
    results: list = []

    for name, pipeline in pipelines.items():
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        predictions[name] = y_pred
        probabilities[name] = y_prob
        results.append(evaluate_model(name, y_test, y_pred, y_prob))

    metrics = (
        pd.DataFrame(results).sort_values("F1", ascending=False).reset_index(drop=True)
    )

    bundle = {
        **split,
        "pipelines": pipelines,
        "predictions": predictions,
        "probabilities": probabilities,
        "metrics": metrics,
        "class_distribution": class_distribution_table(y_train, y_test),
        "sectors": sorted(df["sector"].dropna().unique().tolist()),
        "methods": sorted(df["method"].dropna().unique().tolist()),
        "year_min": int(df["year"].min()),
        "year_max": int(df["year"].max()),
        "has_xgboost": HAS_XGBOOST,
    }

    if PRIMARY_MODEL in pipelines:
        rf_pipeline = pipelines[PRIMARY_MODEL]
        bundle["rf_importance"] = rf_importance_table(rf_pipeline)
        bundle["feature_names"] = list(
            rf_pipeline.named_steps["preprocessor"].get_feature_names_out()
        )

    return bundle


def run_ml(df: pd.DataFrame) -> dict:
    """Split, fit and evaluate all models end to end (cells 61-77)."""
    split = split_and_threshold(df)
    pipelines = fit_pipelines(split["X_train"], split["y_train"])
    return assemble_bundle(pipelines, split, df)


def classification_text_report(y_true, y_pred) -> str:
    """Per-class precision/recall/F1 text block (cell 72)."""
    return classification_report(
        y_true, y_pred, target_names=CLASS_LABELS, zero_division=0
    )


def predict_single(pipeline, year: int, sector: str, method: str) -> dict:
    """Score one hypothetical breach event.

    Supports the dashboard's prediction view (Objective 5 / section 1.5). The
    regulatory indicators are derived from the year exactly as in training, so the
    feature vector matches what the saved pipeline expects.

    This is illustrative decision support, not an operational detector
    (sections 4.12 and 5.3.6).
    """
    row = pd.DataFrame(
        {
            "year": pd.array([int(year)], dtype="Int64"),
            "sector": [sector],
            "method": [method],
            "gdpr_post": [int(year >= GDPR_YEAR)],
            "ccpa_post": [int(year >= CCPA_YEAR)],
        }
    )[FEATURES]

    probability = float(pipeline.predict_proba(row)[:, 1][0])
    predicted = int(pipeline.predict(row)[0])

    return {
        "predicted_class": predicted,
        "predicted_label": CLASS_LABELS[predicted],
        "probability_high": probability,
        "features": row,
    }
