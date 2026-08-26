"""
Data Quality and Feature Store Audit Module.

Provides functions to audit Hopsworks feature store after backfill:
1. load_all_features: Retrieve entire feature group as a sorted pandas DataFrame.
2. generate_gap_report: Detect cadence gaps, timestamp coverage, and columns needing attention (>30% nulls).
3. generate_summary_stats: Calculate distribution stats for numeric features, highlighting target AQI metrics.
4. CLI execution to display formatted tables and persist snapshot to data/processed/aqi_features_snapshot.csv.
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src import feature_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_all_features(
    fs: Any, feature_group_name: str = "aqi_features", version: int = 1
) -> pd.DataFrame:
    """Read the entire feature group into a pandas DataFrame sorted by fetched_at ascending.

    Args:
        fs (Any): Hopsworks Feature Store object.
        feature_group_name (str, optional): Name of the feature group. Defaults to "aqi_features".
        version (int, optional): Feature group version. Defaults to 1.

    Returns:
        pd.DataFrame: DataFrame containing all features sorted chronologically.
    """
    logger.info("Loading feature group '%s' (v%d) from Feature Store...", feature_group_name, version)
    try:
        fg = feature_store.get_or_create_feature_group(fs, name=feature_group_name, version=version)
        df = fg.read()

        if df is None or df.empty:
            logger.warning("Feature group '%s' is empty.", feature_group_name)
            return pd.DataFrame()

        # Sort chronologically by fetched_at
        if "fetched_at" in df.columns:
            df["_sort_dt"] = pd.to_datetime(df["fetched_at"], utc=True)
            df = df.sort_values(by="_sort_dt", ascending=True).drop(columns=["_sort_dt"]).reset_index(drop=True)

        logger.info("Successfully loaded %d records from '%s'.", len(df), feature_group_name)
        return df

    except Exception as err:
        logger.error("Failed to load feature group '%s': %s", feature_group_name, err)
        raise


def generate_gap_report(
    df: pd.DataFrame, expected_cadence_hours: float = 1.0, gap_threshold_hours: float = 1.5
) -> Dict[str, Any]:
    """Audit timestamps and column completeness to detect gaps and null issues.

    Args:
        df (pd.DataFrame): DataFrame containing feature rows.
        expected_cadence_hours (float, optional): Expected hourly interval. Defaults to 1.0.
        gap_threshold_hours (float, optional): Threshold to flag a missing time gap. Defaults to 1.5.

    Returns:
        dict: Audit report with total rows, date range, hourly completeness %,
              detected gap intervals, null counts, and columns needing attention (>30% null).
    """
    report: Dict[str, Any] = {
        "total_rows": len(df) if df is not None else 0,
        "date_range": None,
        "expected_hourly_slots": 0,
        "actual_rows": len(df) if df is not None else 0,
        "completeness_pct": 0.0,
        "gaps": [],
        "null_counts": {},
        "null_percentages": {},
        "needs_attention": [],
    }

    if df is None or df.empty:
        logger.warning("Empty DataFrame provided to generate_gap_report.")
        return report

    # 1. Null counts and percentage per column
    total_len = len(df)
    null_counts = {}
    null_pcts = {}
    needs_attention = []

    for col in df.columns:
        null_cnt = int(df[col].isna().sum())
        null_pct = round((null_cnt / total_len) * 100.0, 2)
        null_counts[col] = null_cnt
        null_pcts[col] = null_pct
        if null_pct > 30.0:
            needs_attention.append(col)

    report["null_counts"] = null_counts
    report["null_percentages"] = null_pcts
    report["needs_attention"] = needs_attention

    # 2. Time Gap and Cadence Analysis
    if "fetched_at" not in df.columns:
        logger.warning("'fetched_at' column missing from DataFrame.")
        return report

    dt_series = pd.to_datetime(df["fetched_at"], utc=True).sort_values().reset_index(drop=True)
    if dt_series.empty:
        return report

    start_dt = dt_series.iloc[0]
    end_dt = dt_series.iloc[-1]
    report["date_range"] = {
        "start": start_dt.isoformat(),
        "end": end_dt.isoformat(),
    }

    total_span_seconds = (end_dt - start_dt).total_seconds()
    total_span_hours = total_span_seconds / 3600.0

    expected_slots = int(total_span_hours / expected_cadence_hours) + 1 if total_span_hours >= 0 else 1
    report["expected_hourly_slots"] = expected_slots

    completeness = round((total_len / max(1, expected_slots)) * 100.0, 2)
    report["completeness_pct"] = min(100.0, completeness)

    # Detect gaps greater than threshold
    gaps: List[Dict[str, Any]] = []
    diffs = dt_series.diff()

    for i in range(1, len(diffs)):
        diff_sec = diffs.iloc[i].total_seconds()
        diff_hrs = diff_sec / 3600.0
        if diff_hrs >= gap_threshold_hours:
            gap_start = dt_series.iloc[i - 1].isoformat()
            gap_end = dt_series.iloc[i].isoformat()
            missing_hours = round(diff_hrs - expected_cadence_hours, 1)
            gaps.append({
                "gap_start": gap_start,
                "gap_end": gap_end,
                "gap_hours": round(diff_hrs, 2),
                "estimated_missing_slots": max(1, int(round(missing_hours))),
            })

    report["gaps"] = gaps
    return report


def generate_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Generate distribution statistics for numeric features, prioritizing target AQI.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Summary statistics table (count, mean, std, min, 25%, 50%, 75%, max).
    """
    if df is None or df.empty:
        return pd.DataFrame()

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return pd.DataFrame()

    desc = numeric_df.describe().T
    desc = desc[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]

    # Round float values for clean presentation
    desc = desc.round(2)

    # Prioritize key target/AQI metrics at top of summary table
    priority_order = ["aqi", "target_aqi_3d", "pm25", "pm10", "temperature", "humidity"]
    ordered_index = [col for col in priority_order if col in desc.index] + [
        col for col in desc.index if col not in priority_order
    ]

    desc = desc.reindex(ordered_index)
    return desc


def _format_table(headers: List[str], rows: List[List[Any]]) -> str:
    """Format table cleanly into ASCII representation."""
    try:
        from tabulate import tabulate
        return tabulate(rows, headers=headers, tablefmt="github")
    except ImportError:
        # Fallback clean ASCII formatter
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))

        header_line = " | ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers))
        sep_line = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
        body_lines = [
            " | ".join(f"{str(cell):<{col_widths[i]}}" for i, cell in enumerate(row))
            for row in rows
        ]
        return f"{header_line}\n{sep_line}\n" + "\n".join(body_lines)


def print_quality_report(report: Dict[str, Any], stats_df: pd.DataFrame) -> None:
    """Print clean and formatted audit report and summary statistics to stdout."""
    print("\n" + "=" * 80)
    print("                      FEATURE STORE DATA QUALITY AUDIT REPORT")
    print("=" * 80)

    print("\n1. DATASET OVERVIEW & CADENCE COVERAGE:")
    print("-" * 50)
    print(f"  Total Rows Loaded        : {report.get('total_rows', 0):,}")
    date_range = report.get("date_range")
    if date_range:
        print(f"  Start Timestamp          : {date_range.get('start')}")
        print(f"  End Timestamp            : {date_range.get('end')}")
    print(f"  Expected Hourly Slots    : {report.get('expected_hourly_slots', 0):,}")
    print(f"  Hourly Completeness      : {report.get('completeness_pct', 0.0):.2f}%")

    # Gaps section
    gaps = report.get("gaps", [])
    print(f"\n2. DETECTED TIME GAPS (Total: {len(gaps)}):")
    print("-" * 50)
    if gaps:
        gap_rows = [
            [g["gap_start"], g["gap_end"], f"{g['gap_hours']} hrs", g["estimated_missing_slots"]]
            for g in gaps[:10]
        ]
        print(_format_table(["Gap Start", "Gap End", "Duration", "Missing Slots"], gap_rows))
        if len(gaps) > 10:
            print(f"  ... and {len(gaps) - 10} more gap period(s).")
    else:
        print("  [OK] No major timestamp gaps detected. Cadence is continuous.")

    # Null completeness section
    print("\n3. COLUMN COMPLETENESS & NULL ANALYSIS:")
    print("-" * 50)
    null_counts = report.get("null_counts", {})
    null_pcts = report.get("null_percentages", {})
    needs_attn = set(report.get("needs_attention", []))

    col_rows = []
    for col, count in null_counts.items():
        pct = null_pcts.get(col, 0.0)
        status = "NEEDS ATTENTION (>30% null)" if col in needs_attn else "OK"
        col_rows.append([col, f"{count:,}", f"{pct:.2f}%", status])

    print(_format_table(["Feature Column", "Null Count", "Null %", "Status"], col_rows))

    # Summary stats section
    print("\n4. NUMERIC DISTRIBUTION SUMMARY (Target AQI highlighted):")
    print("-" * 50)
    if not stats_df.empty:
        stats_rows = []
        for feature, row in stats_df.iterrows():
            tag = "★ TARGET" if feature in ("aqi", "target_aqi_3d") else ""
            stats_rows.append([
                f"{feature} {tag}".strip(),
                f"{int(row['count']):,}",
                f"{row['mean']:.2f}",
                f"{row['std']:.2f}",
                f"{row['min']:.2f}",
                f"{row['25%']:.2f}",
                f"{row['50%']:.2f}",
                f"{row['75%']:.2f}",
                f"{row['max']:.2f}",
            ])
        headers = ["Feature", "Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
        print(_format_table(headers, stats_rows))
    else:
        print("  No numeric features available to summarize.")

    print("\n" + "=" * 80 + "\n")


def ensure_processed_data_snapshot(df: pd.DataFrame, file_path: str = "data/processed/aqi_features_snapshot.csv") -> Path:
    """Save the full feature DataFrame to CSV in data/processed directory."""
    target_path = Path(file_path)
    if not target_path.is_absolute():
        target_path = project_root / target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target_path, index=False)
    logger.info("Saved full feature snapshot to '%s' (%d rows).", target_path, len(df))
    return target_path


def main() -> None:
    """Audit feature store data quality and save processed snapshot for EDA."""
    try:
        import src.config as config
    except Exception as err:
        logger.error("Failed to load project config: %s", err)
        sys.exit(1)

    logger.info("Connecting to Hopsworks Feature Store for Quality Audit...")
    try:
        fs = feature_store.get_feature_store_connection(config)
        df = load_all_features(fs, feature_group_name="aqi_features")
    except Exception as err:
        logger.error("Feature store connection failed: %s", err)
        sys.exit(1)

    if df.empty:
        logger.warning("Feature group contains 0 records. Run backfill or pipeline first.")
        return

    # Generate gap report and summary statistics
    gap_report = generate_gap_report(df)
    stats_df = generate_summary_stats(df)

    # Print formatted audit report
    print_quality_report(gap_report, stats_df)

    # Save to data/processed/aqi_features_snapshot.csv
    snapshot_path = ensure_processed_data_snapshot(df)
    print(f"Snapshot successfully saved to: {snapshot_path}")


if __name__ == "__main__":
    main()
