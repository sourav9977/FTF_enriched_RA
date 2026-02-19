"""Step 9: Output Generation — Final RA with all metadata and summary metrics.

Produces the final RA file with FTF statuses, scores, quantities, and a
summary metrics report for audit.
"""

import logging
from datetime import datetime
from typing import Dict
from dataclasses import dataclass, field

import pandas as pd

from src.pipeline_logger import get_logger

logger = get_logger(__name__)

OUTPUT_VERSION = "1.0.0"

OUTPUT_COLUMNS = [
    "ra_id", "unique_id", "replaced_ra_id",
    "primary_key_type", "primary_key_value",
    "business_unit", "sales_channel", "season",
    "segment", "family", "brand", "brick", "class_", "fashion_grade",
    "month_of_drop", "product_group",
    "mrp", "mrp_bucket",
    "fit", "neck_type", "pattern_type", "sleeve_type", "color", "length",
    "original_quantity", "adjusted_quantity",
    "ftf_status", "ftf_added_source",
    "overlap_score", "ra_confidence_score", "ftf_confidence_score",
    "replacement_reason", "retention_reason",
    "trend_id", "trend_name", "trend_score",
    "trend_stage", "trend_trajectory", "trend_confidence",
    "trend_business_label", "trend_risk_flag",
    "perf_score", "adjustment_factor", "adjustment_reason",
    "shelf_life_cap", "cap_applied",
    "estimation_method",
    "attribute_weights_used",
    "warnings",
    "version", "created_at",
]


@dataclass
class SummaryMetrics:
    """Summary metrics for the final output."""
    total_items: int = 0
    approved_count: int = 0
    approved_pct: float = 0.0
    original_count: int = 0
    original_pct: float = 0.0
    added_count: int = 0
    added_pct: float = 0.0
    added_unmatched: int = 0
    added_replacement: int = 0
    items_replaced: int = 0
    total_original_qty: float = 0.0
    total_adjusted_qty: float = 0.0
    qty_delta_pct: float = 0.0
    items_removed_in_rebalancing: int = 0
    shelf_life_caps: int = 0

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  FTF ENRICHED RA — SUMMARY METRICS",
            "=" * 60,
            f"  Total items:               {self.total_items}",
            f"  FTF APPROVED:              {self.approved_count} ({self.approved_pct:.1%})",
            f"  ORIGINAL:                  {self.original_count} ({self.original_pct:.1%})",
            f"  FTF ADDED:                 {self.added_count} ({self.added_pct:.1%})",
            f"    - From unmatched trends: {self.added_unmatched}",
            f"    - Confidence replacement:{self.added_replacement}",
            f"  Items replaced via conf:   {self.items_replaced}",
            "",
            f"  Total original qty:        {self.total_original_qty:,.0f}",
            f"  Total adjusted qty:        {self.total_adjusted_qty:,.0f}",
            f"  Qty delta:                 {self.qty_delta_pct:+.1%}",
            "",
            f"  Shelf-life caps applied:   {self.shelf_life_caps}",
            f"  Items removed (rebalance): {self.items_removed_in_rebalancing}",
            "=" * 60,
        ]
        return "\n".join(lines)


def compute_summary_metrics(
    final_ra: pd.DataFrame,
    items_removed: int = 0,
    shelf_life_caps: int = 0,
) -> SummaryMetrics:
    """Compute summary metrics from the final RA."""
    m = SummaryMetrics()
    m.total_items = len(final_ra)

    if m.total_items == 0:
        return m

    status = final_ra["ftf_status"]
    m.approved_count = (status == "APPROVED").sum()
    m.original_count = (status == "ORIGINAL").sum()
    m.added_count = (status == "ADDED").sum()

    m.approved_pct = m.approved_count / m.total_items
    m.original_pct = m.original_count / m.total_items
    m.added_pct = m.added_count / m.total_items

    if "ftf_added_source" in final_ra.columns:
        src = final_ra["ftf_added_source"]
        m.added_unmatched = (src == "UNMATCHED_TREND").sum()
        m.added_replacement = (src == "CONFIDENCE_REPLACEMENT").sum()

    if "replaced_ra_id" in final_ra.columns:
        m.items_replaced = final_ra["replaced_ra_id"].notna().sum()

    m.total_original_qty = pd.to_numeric(
        final_ra["original_quantity"], errors="coerce"
    ).sum()
    m.total_adjusted_qty = pd.to_numeric(
        final_ra["adjusted_quantity"], errors="coerce"
    ).sum()

    if m.total_original_qty > 0:
        m.qty_delta_pct = (m.total_adjusted_qty - m.total_original_qty) / m.total_original_qty
    else:
        m.qty_delta_pct = 0.0

    m.items_removed_in_rebalancing = items_removed
    m.shelf_life_caps = shelf_life_caps

    return m


def generate_output(
    final_ra: pd.DataFrame,
    items_removed: int = 0,
    shelf_life_caps: int = 0,
) -> tuple:
    """Generate the final output RA and summary metrics.

    Adds version/timestamp columns and ensures schema conformity.
    Returns (final_df, summary_metrics).
    """
    df = final_ra.copy()

    df["version"] = OUTPUT_VERSION
    df["created_at"] = datetime.now().isoformat()

    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = None

    available_cols = [c for c in OUTPUT_COLUMNS if c in df.columns]
    extra_cols = [c for c in df.columns if c not in OUTPUT_COLUMNS]
    df = df[available_cols + extra_cols]

    metrics = compute_summary_metrics(df, items_removed, shelf_life_caps)

    logger.info(f"Output generated: {len(df)} items, version={OUTPUT_VERSION}")
    logger.info(metrics.summary())
    return df, metrics


def export_to_file(
    final_ra: pd.DataFrame,
    metrics: SummaryMetrics,
    output_path: str,
    format: str = "csv",
) -> str:
    """Export the final RA to a file.

    Supports csv and xlsx formats.
    """
    if format == "xlsx":
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            final_ra.to_excel(writer, sheet_name="Enriched RA", index=False)
            metrics_df = pd.DataFrame([metrics.to_dict()])
            metrics_df.to_excel(writer, sheet_name="Summary", index=False)
    else:
        final_ra.to_csv(output_path, index=False)

    logger.info(f"Exported to {output_path} ({format})")
    return output_path
