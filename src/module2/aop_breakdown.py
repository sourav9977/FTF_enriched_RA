"""Step 5: AOP Breakdown — Decompose AOP targets to Brick level.

Allocates aggregate AOP revenue targets down to Brick and attribute-group
level using historical sales patterns, then adjusts for trend uplift.

Sub-steps:
  5a. Historical brick contribution (brick NSV / total NSV)
  5b. Attribute-level contribution within brick
  5c. Trend adjustment for FTF APPROVED / ADDED items
  5d. Normalize to AOP target (scale quantities to match)
  5e. Validate constraints (MOQ, margin)
"""

import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import FTFConfig
from src.pipeline_logger import get_logger

logger = get_logger(__name__)


@dataclass
class AOPBreakdownResult:
    """Result of AOP breakdown allocation."""
    brick_targets: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    attr_targets: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    scaling_factor: float = 1.0
    total_aop_target: float = 0.0
    total_projected: float = 0.0

    def summary(self) -> str:
        lines = [
            f"AOP Breakdown: {len(self.brick_targets)} brick targets",
            f"  Total AOP target:  {self.total_aop_target:,.0f}",
            f"  Total projected:   {self.total_projected:,.0f}",
            f"  Scaling factor:    {self.scaling_factor:.4f}",
        ]
        if not self.brick_targets.empty:
            lines.append("")
            for _, row in self.brick_targets.iterrows():
                lines.append(
                    f"  {row['brick']:20s} contrib={row['contribution']:.3f}  "
                    f"target_nsv={row['target_nsv']:>10,.0f}  "
                    f"target_qty={row.get('target_qty', 0):>8,.0f}"
                )
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
#  Step 5a: Historical Brick Contribution
# ══════════════════════════════════════════════════════════════════════

def compute_brick_contributions(
    sales_df: pd.DataFrame,
    config: FTFConfig,
) -> pd.DataFrame:
    """Calculate each brick's share of total NSV from historical sales.

    Returns DataFrame with: brick, historical_nsv, contribution
    """
    rev_col = next((c for c in ["revenue", "nsv"] if c in sales_df.columns), None)
    if rev_col is None:
        if "quantity" in sales_df.columns:
            rev_col = "quantity"
        else:
            raise ValueError("Sales data needs revenue or quantity for AOP breakdown")

    if "brick" not in sales_df.columns:
        raise ValueError("Sales data missing 'brick' column")

    brick_nsv = sales_df.groupby("brick")[rev_col].sum().reset_index()
    brick_nsv.columns = ["brick", "historical_nsv"]
    total_nsv = brick_nsv["historical_nsv"].sum()

    brick_nsv["contribution"] = (
        brick_nsv["historical_nsv"] / total_nsv if total_nsv > 0 else 0.0
    )

    # Average ASP per brick
    if "quantity" in sales_df.columns and rev_col != "quantity":
        brick_qty = sales_df.groupby("brick")["quantity"].sum().reset_index()
        brick_qty.columns = ["brick", "historical_qty"]
        brick_nsv = brick_nsv.merge(brick_qty, on="brick", how="left")
        brick_nsv["avg_asp"] = (
            brick_nsv["historical_nsv"] / brick_nsv["historical_qty"]
        ).fillna(0)
    else:
        brick_nsv["historical_qty"] = brick_nsv["historical_nsv"]
        brick_nsv["avg_asp"] = 1.0

    logger.info(f"Brick contributions: {len(brick_nsv)} bricks, total NSV={total_nsv:,.0f}")
    for _, row in brick_nsv.iterrows():
        logger.debug(
            f"  {row['brick']:20s} NSV={row['historical_nsv']:>12,.0f}  "
            f"share={row['contribution']:.3f}  "
            f"ASP={row.get('avg_asp', 0):>8,.1f}"
        )
    return brick_nsv


# ══════════════════════════════════════════════════════════════════════
#  Step 5b: Attribute-Level Contribution
# ══════════════════════════════════════════════════════════════════════

def compute_attribute_contributions(
    sales_df: pd.DataFrame,
    catalog_df: pd.DataFrame,
    group_by: list = None,
) -> pd.DataFrame:
    """Calculate attribute-group contribution within each brick.

    group_by defines the attribute groups (default: pattern × color).
    Returns DataFrame with: brick, group columns, attr_nsv, attr_contribution
    """
    if group_by is None:
        group_by = ["color"]  # start with what's available

    rev_col = next((c for c in ["revenue", "nsv"] if c in sales_df.columns), None)
    if rev_col is None:
        rev_col = "quantity"

    available_cols = [c for c in group_by if c in sales_df.columns]
    if not available_cols or "brick" not in sales_df.columns:
        return pd.DataFrame()

    keys = ["brick"] + available_cols
    attr_nsv = sales_df.groupby(keys)[rev_col].sum().reset_index()
    attr_nsv.columns = keys + ["attr_nsv"]

    brick_totals = attr_nsv.groupby("brick")["attr_nsv"].sum().reset_index()
    brick_totals.columns = ["brick", "brick_total"]

    attr_nsv = attr_nsv.merge(brick_totals, on="brick", how="left")
    attr_nsv["attr_contribution"] = (
        attr_nsv["attr_nsv"] / attr_nsv["brick_total"]
    ).fillna(0)

    return attr_nsv.drop(columns=["brick_total"])


# ══════════════════════════════════════════════════════════════════════
#  Step 5c: Trend Adjustment
# ══════════════════════════════════════════════════════════════════════

def apply_trend_adjustment(
    brick_targets: pd.DataFrame,
    enriched_ra: pd.DataFrame,
    config: FTFConfig,
) -> pd.DataFrame:
    """Adjust brick contributions based on trend signals in the Enriched RA.

    FTF APPROVED items with rising trends get an uplift.
    FTF ADDED items use nearest-neighbor contribution as proxy.
    """
    aop_params = config.aop_breakdown_params
    max_uplift = aop_params.get("trend_uplift_factor_max", 0.10)

    targets = brick_targets.copy()
    targets["trend_uplift"] = 0.0

    for brick in targets["brick"].values:
        brick_items = enriched_ra[
            (enriched_ra["brick"].str.upper() == brick.upper())
        ]
        if brick_items.empty:
            continue

        approved_rising = brick_items[
            (brick_items["ftf_status"] == "APPROVED") &
            (brick_items["trend_stage"].str.upper().isin(["RISE", "RISING"]))
        ]

        if approved_rising.empty:
            continue

        avg_trend_score = pd.to_numeric(
            approved_rising["trend_score"], errors="coerce"
        ).mean()

        if pd.isna(avg_trend_score):
            continue

        uplift = min(max_uplift, 0.1 * (avg_trend_score - 0.5))
        uplift = max(0.0, uplift)

        mask = targets["brick"].str.upper() == brick.upper()
        targets.loc[mask, "trend_uplift"] = uplift
        logger.debug(
            f"  Trend uplift [{brick}]: avg_trend_score={avg_trend_score:.3f}, "
            f"n_rising={len(approved_rising)}, uplift={uplift:+.3f}"
        )

    targets["adjusted_contribution"] = targets["contribution"] * (1 + targets["trend_uplift"])

    # Re-normalize to sum to 1.0
    total = targets["adjusted_contribution"].sum()
    if total > 0:
        targets["adjusted_contribution"] = targets["adjusted_contribution"] / total

    return targets


# ══════════════════════════════════════════════════════════════════════
#  Step 5d: Normalize to AOP Target
# ══════════════════════════════════════════════════════════════════════

def allocate_aop_to_bricks(
    brick_targets: pd.DataFrame,
    aop_df: pd.DataFrame,
    config: FTFConfig,
) -> Tuple[pd.DataFrame, float]:
    """Allocate AOP revenue targets to each brick using adjusted contributions.

    Returns (brick_targets with target_nsv/target_qty, total_aop_target).
    """
    rev_col = next((c for c in ["revenue", "target_nsv", "aop_target"]
                     if c in aop_df.columns), None)
    if rev_col is None:
        raise ValueError(f"AOP data missing revenue column. Available: {list(aop_df.columns)}")

    total_aop = aop_df[rev_col].sum()

    contrib_col = "adjusted_contribution" if "adjusted_contribution" in brick_targets.columns else "contribution"

    targets = brick_targets.copy()
    targets["target_nsv"] = targets[contrib_col] * total_aop

    if "avg_asp" in targets.columns:
        targets["target_qty"] = (targets["target_nsv"] / targets["avg_asp"]).fillna(0).astype(int)
    else:
        targets["target_qty"] = targets["target_nsv"].astype(int)

    logger.debug(f"  AOP allocation: total_aop={total_aop:,.0f}, using '{contrib_col}'")
    for _, row in targets.iterrows():
        logger.debug(
            f"    {row['brick']:20s} target_nsv={row['target_nsv']:>12,.0f}  "
            f"target_qty={row.get('target_qty', 0):>8,.0f}"
        )
    return targets, total_aop


# ══════════════════════════════════════════════════════════════════════
#  Step 5e: Validate Constraints
# ══════════════════════════════════════════════════════════════════════

def validate_constraints(
    brick_targets: pd.DataFrame,
    config: FTFConfig,
) -> pd.DataFrame:
    """Apply MOQ and basic constraints to brick-level targets."""
    targets = brick_targets.copy()

    if "target_qty" in targets.columns:
        targets["pre_moq_qty"] = targets["target_qty"]
        targets["target_qty"] = targets["target_qty"].clip(lower=config.moq_per_option)
        targets["moq_adjusted"] = targets["target_qty"] != targets["pre_moq_qty"]
        targets.drop(columns=["pre_moq_qty"], inplace=True)
    else:
        targets["moq_adjusted"] = False

    return targets


# ══════════════════════════════════════════════════════════════════════
#  Full Pipeline
# ══════════════════════════════════════════════════════════════════════

def run_aop_breakdown(
    aop_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    enriched_ra: pd.DataFrame,
    config: FTFConfig,
    catalog_df: pd.DataFrame = None,
) -> AOPBreakdownResult:
    """Execute the full Step 5 pipeline.

    Returns AOPBreakdownResult with brick-level targets and scaling info.
    """
    # 5a: Brick contributions from historical sales
    brick_contribs = compute_brick_contributions(sales_df, config)

    # 5b: Attribute-level contributions (optional enrichment)
    attr_contribs = pd.DataFrame()
    if catalog_df is not None:
        attr_contribs = compute_attribute_contributions(sales_df, catalog_df)

    # 5c: Trend adjustment
    brick_contribs = apply_trend_adjustment(brick_contribs, enriched_ra, config)

    # 5d: Allocate to AOP target
    brick_contribs, total_aop = allocate_aop_to_bricks(brick_contribs, aop_df, config)

    # 5e: Constraints
    brick_contribs = validate_constraints(brick_contribs, config)

    total_projected = brick_contribs["target_nsv"].sum() if "target_nsv" in brick_contribs.columns else 0
    scaling = total_aop / total_projected if total_projected > 0 else 1.0

    result = AOPBreakdownResult(
        brick_targets=brick_contribs,
        attr_targets=attr_contribs,
        scaling_factor=scaling,
        total_aop_target=total_aop,
        total_projected=total_projected,
    )
    logger.info(result.summary())
    return result
