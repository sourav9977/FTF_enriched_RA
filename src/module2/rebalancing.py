"""Step 8: Constraint Optimization & Rebalancing.

Ensures the combined RA satisfies all business constraints:
  1. MOQ validation (boost high-confidence items, drop low ones)
  2. AOP budget compliance (reduce lowest-priority items first)
  3. Intake margin check
  4. FTF Added cap enforcement
  5. Post-rebalance shelf-life re-validation
"""

import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import FTFConfig

logger = logging.getLogger(__name__)


@dataclass
class RebalancingResult:
    """Outcome of constraint optimization."""
    final_ra: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    items_removed: List[dict] = field(default_factory=list)
    moq_boosts: int = 0
    moq_drops: int = 0
    budget_reductions: int = 0
    ftf_cap_drops: int = 0
    shelf_life_warnings: int = 0

    def summary(self) -> str:
        return (
            f"Rebalancing: {len(self.final_ra)} items retained, "
            f"{len(self.items_removed)} removed\n"
            f"  MOQ boosts: {self.moq_boosts}, MOQ drops: {self.moq_drops}\n"
            f"  Budget reductions: {self.budget_reductions}\n"
            f"  FTF cap drops: {self.ftf_cap_drops}\n"
            f"  Shelf-life warnings: {self.shelf_life_warnings}"
        )


# ══════════════════════════════════════════════════════════════════════
#  Constraint 1: MOQ Validation
# ══════════════════════════════════════════════════════════════════════

def apply_moq_constraint(
    df: pd.DataFrame,
    config: FTFConfig,
) -> Tuple[pd.DataFrame, int, int, List[dict]]:
    """Validate MOQ for each item.

    High-confidence, high-trend items are boosted to MOQ.
    Others below MOQ are removed.

    Returns (df, n_boosts, n_drops, removed_items).
    """
    moq = config.moq_per_option
    result = df.copy()
    removed = []
    boosts = 0
    drops = 0

    drop_indices = []
    for idx, row in result.iterrows():
        qty = pd.to_numeric(row.get("adjusted_quantity"), errors="coerce")
        if pd.isna(qty):
            continue

        if qty >= moq:
            continue

        trend_score = pd.to_numeric(row.get("trend_score"), errors="coerce")
        if pd.isna(trend_score):
            trend_score = 0.0
        confidence = str(row.get("trend_confidence", "")).strip().upper()

        if trend_score >= 0.7 and confidence == "HIGH":
            result.at[idx, "adjusted_quantity"] = moq
            result.at[idx, "adjustment_reason"] = (
                f"{row.get('adjustment_reason', '')}; MOQ boost: {qty:.0f} -> {moq}"
            )
            boosts += 1
        else:
            removed.append(row.to_dict())
            drop_indices.append(idx)
            drops += 1

    result = result.drop(index=drop_indices).reset_index(drop=True)

    logger.info(f"MOQ constraint: {boosts} boosted, {drops} dropped (MOQ={moq})")
    return result, boosts, drops, removed


# ══════════════════════════════════════════════════════════════════════
#  Constraint 2: AOP Budget Compliance
# ══════════════════════════════════════════════════════════════════════

def _compute_rebalance_scores(
    df: pd.DataFrame,
    config: FTFConfig,
) -> pd.Series:
    """Compute rebalancing priority scores.

    Rebalance_Score = 0.40×Past_Sales_Contribution + 0.30×GM_Contribution
                    + 0.20×Trend_Score + 0.10×Product_Mix_Priority

    Items with lower scores are reduced first.
    """
    w = config.rebalancing_weights
    scores = pd.Series(0.5, index=df.index)

    for idx, row in df.iterrows():
        ts = pd.to_numeric(row.get("trend_score"), errors="coerce")
        ts = 0.5 if pd.isna(ts) else min(1.0, max(0.0, ts))

        # Proxy: use overlap_score as sales contribution proxy
        ov = pd.to_numeric(row.get("overlap_score"), errors="coerce")
        ov = 0.5 if pd.isna(ov) else min(1.0, max(0.0, ov))

        # Status-based mix priority: APPROVED > ORIGINAL > ADDED
        status = str(row.get("ftf_status", "")).upper()
        mix = {"APPROVED": 0.8, "ORIGINAL": 0.5, "ADDED": 0.3}.get(status, 0.5)

        score = (
            w.get("past_sales", 0.40) * ov
            + w.get("gm", 0.30) * 0.5
            + w.get("trend", 0.20) * ts
            + w.get("mix", 0.10) * mix
        )
        scores.at[idx] = score

    return scores


def apply_budget_constraint(
    df: pd.DataFrame,
    aop_budget: float,
    config: FTFConfig,
) -> Tuple[pd.DataFrame, int, List[dict]]:
    """Reduce quantities to fit within AOP budget.

    Reduces lowest-priority items first, respecting MOQ.
    Items that fall below MOQ are removed.
    """
    moq = config.moq_per_option

    def _total_nsv(d):
        qty = pd.to_numeric(d["adjusted_quantity"], errors="coerce").fillna(0)
        asp = pd.to_numeric(d.get("mrp", d.get("asp", 0)), errors="coerce").fillna(0)
        asp = asp.where(asp > 0, 1.0)
        return (qty * asp).sum()

    total = _total_nsv(df)
    if total <= aop_budget:
        return df, 0, []

    overage = total - aop_budget
    rebal_scores = _compute_rebalance_scores(df, config)
    df = df.copy()
    df["_rebal_score"] = rebal_scores

    df = df.sort_values("_rebal_score", ascending=True)

    reductions = 0
    removed = []
    drop_indices = []

    for idx, row in df.iterrows():
        if overage <= 0:
            break

        qty = pd.to_numeric(row.get("adjusted_quantity"), errors="coerce")
        if pd.isna(qty) or qty <= 0:
            continue

        asp = pd.to_numeric(row.get("mrp", row.get("asp", 1)), errors="coerce")
        asp = 1.0 if pd.isna(asp) or asp <= 0 else asp

        max_reduction_qty = qty - moq
        if max_reduction_qty <= 0:
            removed.append(row.to_dict())
            drop_indices.append(idx)
            overage -= qty * asp
            continue

        needed_reduction_qty = overage / asp
        actual_reduction = min(max_reduction_qty, needed_reduction_qty)

        df.at[idx, "adjusted_quantity"] = round(qty - actual_reduction)
        overage -= actual_reduction * asp
        reductions += 1

        if df.at[idx, "adjusted_quantity"] < moq:
            removed.append(df.loc[idx].to_dict())
            drop_indices.append(idx)

    df = df.drop(index=drop_indices).reset_index(drop=True)
    if "_rebal_score" in df.columns:
        df = df.drop(columns=["_rebal_score"])

    logger.info(
        f"Budget constraint: {reductions} reductions, "
        f"{len(removed)} removed, overage remaining={max(0, overage):,.0f}"
    )
    return df, reductions, removed


# ══════════════════════════════════════════════════════════════════════
#  Constraint 3: Intake Margin (placeholder)
# ══════════════════════════════════════════════════════════════════════

def check_intake_margin(
    df: pd.DataFrame,
    target_im: float = 0.0,
) -> Tuple[bool, float]:
    """Check weighted intake margin meets target.

    Returns (passes, achieved_im).
    Placeholder: returns True if no margin data available.
    """
    if "margin" not in df.columns and "im" not in df.columns:
        return True, 0.0

    im_col = "margin" if "margin" in df.columns else "im"
    qty = pd.to_numeric(df["adjusted_quantity"], errors="coerce").fillna(0)
    asp = pd.to_numeric(df.get("mrp", 1), errors="coerce").fillna(1)
    im = pd.to_numeric(df[im_col], errors="coerce").fillna(0)

    nsv = qty * asp
    total_nsv = nsv.sum()
    if total_nsv == 0:
        return True, 0.0

    weighted_im = (nsv * im).sum() / total_nsv
    return weighted_im >= target_im, weighted_im


# ══════════════════════════════════════════════════════════════════════
#  Constraint 4: FTF Added Cap
# ══════════════════════════════════════════════════════════════════════

def apply_ftf_added_cap(
    df: pd.DataFrame,
    config: FTFConfig,
) -> Tuple[pd.DataFrame, int, List[dict]]:
    """Ensure FTF Added items don't exceed cap percentage.

    Drops lowest-confidence ADDED items first.
    """
    total = len(df)
    if total == 0:
        return df, 0, []

    added_mask = df["ftf_status"] == "ADDED"
    n_added = added_mask.sum()
    max_added = int(total * config.ftf_added_cap_pct)

    if n_added <= max_added:
        return df, 0, []

    excess = n_added - max_added
    added_items = df[added_mask].copy()

    ftf_conf = pd.to_numeric(added_items.get("ftf_confidence_score", 0), errors="coerce").fillna(0)
    added_items = added_items.assign(_conf=ftf_conf)
    added_items = added_items.sort_values("_conf", ascending=True)

    drop_indices = added_items.index[:excess].tolist()
    removed = [df.loc[i].to_dict() for i in drop_indices]

    df = df.drop(index=drop_indices).reset_index(drop=True)

    logger.info(f"FTF cap: {excess} ADDED items dropped (cap={config.ftf_added_cap_pct:.0%})")
    return df, len(removed), removed


# ══════════════════════════════════════════════════════════════════════
#  Constraint 5: Post-Rebalance Shelf-Life Re-validation
# ══════════════════════════════════════════════════════════════════════

def revalidate_shelf_life(
    df: pd.DataFrame,
    ros_per_brick: Dict[str, float],
    config: FTFConfig,
) -> Tuple[pd.DataFrame, int]:
    """Flag items whose post-rebalance qty exceeds shelf-life cap.

    Does NOT auto-reduce (MOQ takes precedence). Logs warnings.
    """
    warnings = 0
    df = df.copy()
    if "warnings" not in df.columns:
        df["warnings"] = None

    for idx, row in df.iterrows():
        qty = pd.to_numeric(row.get("adjusted_quantity"), errors="coerce")
        if pd.isna(qty) or qty <= 0:
            continue

        brick = str(row.get("brick", "")).strip().upper()
        ros = ros_per_brick.get(brick, 0)
        if ros <= 0:
            continue

        max_qty = (ros * config.shelf_life_days) / config.str_target
        if qty > max_qty:
            warn_msg = (
                f"Post-rebalance qty ({qty:.0f}) exceeds shelf-life cap ({max_qty:.0f})"
            )
            existing = df.at[idx, "warnings"]
            if existing and isinstance(existing, list):
                existing.append(warn_msg)
            else:
                df.at[idx, "warnings"] = [warn_msg]
            warnings += 1

    return df, warnings


# ══════════════════════════════════════════════════════════════════════
#  Full Pipeline
# ══════════════════════════════════════════════════════════════════════

def run_rebalancing(
    enriched_ra: pd.DataFrame,
    aop_budget: float,
    config: FTFConfig,
    ros_per_brick: Dict[str, float] = None,
) -> RebalancingResult:
    """Execute the full Step 8 constraint optimization pipeline."""
    result = RebalancingResult()

    df = enriched_ra.copy()

    # Constraint 1: MOQ
    df, boosts, drops, removed_moq = apply_moq_constraint(df, config)
    result.moq_boosts = boosts
    result.moq_drops = drops
    result.items_removed.extend(removed_moq)

    # Constraint 4: FTF Added cap (before budget, so budget works on final count)
    df, cap_drops, removed_cap = apply_ftf_added_cap(df, config)
    result.ftf_cap_drops = cap_drops
    result.items_removed.extend(removed_cap)

    # Constraint 2: AOP budget
    df, budget_reds, removed_budget = apply_budget_constraint(df, aop_budget, config)
    result.budget_reductions = budget_reds
    result.items_removed.extend(removed_budget)

    # Constraint 3: Intake margin (check only, no auto-fix)
    im_ok, achieved_im = check_intake_margin(df)
    if not im_ok:
        logger.warning(f"Intake margin below target: achieved {achieved_im:.2%}")

    # Constraint 5: Shelf-life re-validation
    if ros_per_brick:
        df, sl_warns = revalidate_shelf_life(df, ros_per_brick, config)
        result.shelf_life_warnings = sl_warns

    result.final_ra = df
    logger.info(result.summary())
    return result
