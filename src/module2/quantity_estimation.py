"""Steps 6 & 7: Quantity Estimation for FTF Approved and FTF Added items.

Step 6 — FTF Approved: Adjust planned quantities using a performance score
         (ROS, STR, Turns, Contribution, Trend Score), capped by shelf-life.

Step 7 — FTF Added:   Estimate quantities for new items via nearest-neighbor
         search on historical sales, adjusted by trend multiplier, capped by
         shelf-life. Falls back to family-level percentile when insufficient
         neighbors.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import FTFConfig
from src.module1.factor_analysis import FactorAnalysisResult, get_default_weights

logger = logging.getLogger(__name__)


@dataclass
class QuantityEstimationResult:
    """Full result of quantity estimation across all Enriched RA items."""
    final_ra: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    approved_adjustments: int = 0
    added_estimates: int = 0
    shelf_life_caps_applied: int = 0
    fallback_count: int = 0

    def summary(self) -> str:
        return (
            f"Quantity Estimation: "
            f"{self.approved_adjustments} APPROVED adjusted, "
            f"{self.added_estimates} ADDED estimated, "
            f"{self.shelf_life_caps_applied} shelf-life caps, "
            f"{self.fallback_count} fallbacks"
        )


# ══════════════════════════════════════════════════════════════════════
#  Step 6: Quantity Estimation — FTF Approved Items
# ══════════════════════════════════════════════════════════════════════

# ── 6a: Performance Score ─────────────────────────────────────────────

def compute_performance_scores(
    enriched_ra: pd.DataFrame,
    sales_df: pd.DataFrame,
    config: FTFConfig,
) -> pd.Series:
    """Compute performance score for each enriched RA item.

    Perf = α×norm(ROS) + β×norm(STR) + γ×norm(Turns) + δ×norm(Contribution) + ε×Trend_Score
    Default weights: α=0.30, β=0.25, γ=0.15, δ=0.15, ε=0.15
    """
    pw = config.performance_weights
    scores = pd.Series(0.5, index=enriched_ra.index, name="perf_score")

    if sales_df is None or sales_df.empty:
        # Use trend score alone as proxy
        ts = pd.to_numeric(enriched_ra.get("trend_score"), errors="coerce").fillna(0.5)
        scores = ts.clip(0, 1)
        return scores

    rev_col = next((c for c in ["revenue", "nsv"] if c in sales_df.columns), None)
    qty_col = "quantity" if "quantity" in sales_df.columns else None

    # Build brick-level sales stats
    if "brick" in sales_df.columns and rev_col:
        sku_col = "sku" if "sku" in sales_df.columns else "sku_code"

        brick_stats = sales_df.groupby("brick").agg(
            total_nsv=(rev_col, "sum"),
            total_qty=(qty_col, "sum") if qty_col else (rev_col, "count"),
            n_skus=(sku_col, "nunique"),
        ).reset_index()

        if qty_col:
            day_col = "day" if "day" in sales_df.columns else None
            if day_col:
                n_days = sales_df[day_col].nunique()
                n_days = max(n_days, 1)
            else:
                n_days = 365

            brick_stats["avg_ros"] = brick_stats["total_qty"] / (brick_stats["n_skus"] * n_days)
        else:
            brick_stats["avg_ros"] = 0.0

        brick_lookup = brick_stats.set_index("brick")
    else:
        brick_lookup = pd.DataFrame()

    for idx, row in enriched_ra.iterrows():
        brick = str(row.get("brick", "")).strip().upper()

        # Trend score component
        ts = pd.to_numeric(row.get("trend_score"), errors="coerce")
        if pd.isna(ts):
            ts = 0.5
        ts = min(1.0, max(0.0, ts))

        if brick in brick_lookup.index:
            stats = brick_lookup.loc[brick]
            total_nsv = stats.get("total_nsv", 0)

            # Contribution proxy
            contrib = 1.0 / max(stats.get("n_skus", 1), 1)
            norm_contrib = min(1.0, contrib * stats.get("n_skus", 1))

            # ROS proxy (normalized)
            ros = stats.get("avg_ros", 0)
            norm_ros = min(1.0, ros / max(brick_lookup["avg_ros"].max(), 0.001))

            # STR proxy
            norm_str = 0.5

            # Turns proxy
            norm_turns = 0.5

            score = (
                pw.get("ros", 0.30) * norm_ros
                + pw.get("str", 0.25) * norm_str
                + pw.get("turns", 0.15) * norm_turns
                + pw.get("contribution", 0.15) * norm_contrib
                + pw.get("trend_score", 0.15) * ts
            )
        else:
            score = 0.5 * (1 - pw.get("trend_score", 0.15)) + pw.get("trend_score", 0.15) * ts

        scores.at[idx] = min(1.0, max(0.0, score))

    return scores


# ── 6b: Calculate Adjustment ──────────────────────────────────────────

def adjust_approved_quantities(
    enriched_ra: pd.DataFrame,
    perf_scores: pd.Series,
    config: FTFConfig,
) -> pd.DataFrame:
    """Adjust planned quantities for FTF APPROVED items.

    Adjustment_Factor = (Perf_Score - 0.5) × Dampening_Factor
    Dampening_Factor = max_adjustment_pct × 2
    Adjusted_Qty = Original_Qty × (1 + Adjustment_Factor)
    """
    df = enriched_ra.copy()
    dampening = config.max_adjustment_pct * 2

    mask = df["ftf_status"] == "APPROVED"
    approved_idx = df[mask].index

    df["adjusted_quantity"] = df["original_quantity"]
    df["adjustment_factor"] = 0.0
    df["adjustment_reason"] = None

    for idx in approved_idx:
        orig_qty = pd.to_numeric(df.at[idx, "original_quantity"], errors="coerce")
        if pd.isna(orig_qty) or orig_qty <= 0:
            continue

        perf = perf_scores.get(idx, 0.5)
        adj_factor = (perf - 0.5) * dampening
        adj_qty = orig_qty * (1 + adj_factor)
        adj_qty = max(0, round(adj_qty))

        df.at[idx, "adjusted_quantity"] = adj_qty
        df.at[idx, "adjustment_factor"] = adj_factor
        df.at[idx, "adjustment_reason"] = f"Perf={perf:.2f}, adj={adj_factor:+.2%}"

    return df


# ── 6c: Shelf-Life & ROS Cap ─────────────────────────────────────────

def apply_shelf_life_cap(
    enriched_ra: pd.DataFrame,
    sales_df: pd.DataFrame,
    config: FTFConfig,
) -> Tuple[pd.DataFrame, int]:
    """Cap adjusted quantities to ensure sellability within shelf life.

    Max_Shelf_Life_Qty = (Projected_ROS × shelf_life_days) / str_target
    """
    df = enriched_ra.copy()
    caps_applied = 0

    if "adjusted_quantity" not in df.columns:
        df["adjusted_quantity"] = df.get("original_quantity", 0)

    # Compute ROS per brick from sales
    ros_per_brick = _compute_brick_ros(sales_df, config)

    df["shelf_life_cap"] = None
    df["cap_applied"] = False

    for idx, row in df.iterrows():
        qty = pd.to_numeric(row.get("adjusted_quantity"), errors="coerce")
        if pd.isna(qty) or qty <= 0:
            continue

        brick = str(row.get("brick", "")).strip().upper()
        projected_ros = ros_per_brick.get(brick, 0)

        if projected_ros <= 0:
            continue

        max_qty = (projected_ros * config.shelf_life_days) / config.str_target
        max_qty = round(max_qty)

        if qty > max_qty:
            df.at[idx, "adjusted_quantity"] = max_qty
            df.at[idx, "shelf_life_cap"] = (
                f"ROS={projected_ros:.1f}/day × {config.shelf_life_days}d / "
                f"{config.str_target:.0%}STR = {max_qty}"
            )
            df.at[idx, "cap_applied"] = True
            caps_applied += 1

    return df, caps_applied


def _compute_brick_ros(sales_df: pd.DataFrame, config: FTFConfig) -> Dict[str, float]:
    """Compute average rate-of-sale per brick (units/day)."""
    if sales_df is None or sales_df.empty:
        return {}

    qty_col = "quantity" if "quantity" in sales_df.columns else None
    if qty_col is None or "brick" not in sales_df.columns:
        return {}

    sku_col = "sku" if "sku" in sales_df.columns else "sku_code"
    day_col = "day" if "day" in sales_df.columns else None

    if day_col and sales_df[day_col].notna().any():
        n_days = max(1, sales_df[day_col].nunique())
    else:
        n_days = 365

    brick_ros = {}
    for brick, grp in sales_df.groupby("brick"):
        total_qty = grp[qty_col].sum()
        n_skus = grp[sku_col].nunique()
        if n_skus > 0:
            brick_ros[str(brick).upper()] = total_qty / (n_skus * n_days)

    return brick_ros


# ══════════════════════════════════════════════════════════════════════
#  Step 7: Quantity Estimation — FTF Added Items
# ══════════════════════════════════════════════════════════════════════

# ── 7a: Nearest Neighbor Search ───────────────────────────────────────

def _nearest_neighbor_estimate(
    item: pd.Series,
    sales_df: pd.DataFrame,
    catalog_df: pd.DataFrame,
    factor_result: FactorAnalysisResult,
    config: FTFConfig,
) -> Tuple[Optional[float], Optional[float], str]:
    """Find K nearest neighbors for a new item and estimate base quantity.

    Returns (base_qty, projected_ros, method).
    """
    brick = str(item.get("brick", "")).strip().upper()
    if not brick:
        return None, None, "no_brick"

    weights = factor_result.get_weights(brick) or get_default_weights()

    sku_col = "sku" if "sku" in sales_df.columns else "sku_code"
    qty_col = "quantity" if "quantity" in sales_df.columns else None
    if qty_col is None:
        return None, None, "no_qty_column"

    brick_sales = sales_df[sales_df["brick"].str.upper() == brick].copy()
    if len(brick_sales) == 0:
        return None, None, "no_brick_sales"

    sku_agg = brick_sales.groupby(sku_col).agg({qty_col: "sum"}).reset_index()
    sku_agg.columns = ["sku_code", "total_qty"]

    # Enrich with catalog attributes if available
    if catalog_df is not None and not catalog_df.empty:
        cat_sku = "sku_code" if "sku_code" in catalog_df.columns else "sku"
        cat_slim = catalog_df.drop_duplicates(subset=[cat_sku])
        sku_agg = sku_agg.merge(
            cat_slim, left_on="sku_code", right_on=cat_sku, how="left"
        )

    # Also check sales-side attributes
    for attr in ["color", "pattern"]:
        if attr not in sku_agg.columns and attr in brick_sales.columns:
            attr_first = brick_sales.groupby(sku_col)[attr].first().reset_index()
            attr_first.columns = ["sku_code", attr]
            sku_agg = sku_agg.merge(attr_first, on="sku_code", how="left")

    # Compute similarity between the item and each historical SKU
    similarities = []
    attr_map = {
        "pattern_type": "pattern",
        "color": "color",
        "fit": "fit",
        "sleeve_type": "sleeve",
        "neck_type": "neck_type",
        "length": "length",
    }

    for _, sku_row in sku_agg.iterrows():
        total_w = 0.0
        matched_w = 0.0
        for item_attr, sku_attr in attr_map.items():
            w_key = sku_attr.replace("_type", "") if sku_attr != "neck_type" else "neck_type"
            w = weights.get(w_key, 0.0)
            if w == 0:
                continue
            total_w += w

            item_val = item.get(item_attr)
            sku_val = sku_row.get(sku_attr)

            if pd.isna(item_val) or str(item_val).strip() == "":
                matched_w += w
                continue
            if pd.notna(sku_val) and str(item_val).strip().upper() == str(sku_val).strip().upper():
                matched_w += w

        sim = matched_w / total_w if total_w > 0 else 0.0
        similarities.append((sim, sku_row.get("total_qty", 0)))

    # Filter neighbors with similarity >= 0.60, take top K
    min_sim = 0.60
    neighbors = [(sim, qty) for sim, qty in similarities if sim >= min_sim]
    neighbors.sort(key=lambda x: -x[0])
    neighbors = neighbors[:config.k_neighbors]

    # Calculate days in sales period
    day_col = "day" if "day" in brick_sales.columns else None
    if day_col and brick_sales[day_col].notna().any():
        n_days = max(1, brick_sales[day_col].nunique())
    else:
        n_days = 365

    n_skus = brick_sales[sku_col].nunique()

    if len(neighbors) >= config.min_nearest_neighbors:
        total_sim = sum(s for s, _ in neighbors)
        if total_sim > 0:
            base_qty = sum(s * q for s, q in neighbors) / total_sim
        else:
            base_qty = np.mean([q for _, q in neighbors])

        avg_qty = np.mean([q for _, q in neighbors])
        projected_ros = avg_qty / n_days if n_days > 0 else 0.0

        return base_qty, projected_ros, "nearest_neighbor"
    else:
        return None, None, "insufficient_neighbors"


# ── 7b: Trend Score Influence ─────────────────────────────────────────

def _apply_trend_multiplier(
    base_qty: float,
    trend_score: float,
    confidence: str,
    config: FTFConfig,
) -> float:
    """Apply trend multiplier to base quantity.

    Trend_Multiplier = base + (scale × Trend_Score × Confidence_Factor)
    Range: [base, base + scale]  (default 0.7 to 1.3)
    """
    params = config.trend_multiplier_params
    base = params.get("base", 0.7)
    scale = params.get("scale", 0.6)
    conf_factors = params.get("confidence_factors", {"High": 1.0, "Medium": 0.8, "Low": 0.6})

    conf_str = str(confidence).strip().capitalize()
    conf_factor = conf_factors.get(conf_str, 0.6)

    multiplier = base + (scale * trend_score * conf_factor)
    multiplier = max(base, min(base + scale, multiplier))

    return round(base_qty * multiplier)


# ── 7d: Family-Level Fallback ─────────────────────────────────────────

def _family_fallback(
    item: pd.Series,
    sales_df: pd.DataFrame,
    config: FTFConfig,
) -> Tuple[float, float]:
    """Fallback quantity estimation using family-level percentiles.

    P = 25th if trend_score < 0.5
    P = 50th if 0.5 <= trend_score < 0.7
    P = 75th if trend_score >= 0.7

    Returns (base_qty, projected_ros).
    """
    pcts = config.fallback_percentiles
    brick = str(item.get("brick", "")).strip().upper()
    trend_score = pd.to_numeric(item.get("trend_score"), errors="coerce")
    if pd.isna(trend_score):
        trend_score = 0.5

    # Determine percentile
    if trend_score < 0.5:
        pct = pcts.get("low", 25)
    elif trend_score < 0.7:
        pct = pcts.get("mid", 50)
    else:
        pct = pcts.get("high", 75)

    qty_col = "quantity" if "quantity" in sales_df.columns else None
    if qty_col is None:
        return 100.0, 1.0

    sku_col = "sku" if "sku" in sales_df.columns else "sku_code"

    # Filter to same brick (or family if brick has no data)
    brick_sales = sales_df[sales_df["brick"].str.upper() == brick] if "brick" in sales_df.columns else sales_df

    if brick_sales.empty:
        return 100.0, 1.0

    sku_qtys = brick_sales.groupby(sku_col)[qty_col].sum()
    base_qty = max(1, np.percentile(sku_qtys, pct))

    day_col = "day" if "day" in brick_sales.columns else None
    n_days = max(1, brick_sales[day_col].nunique()) if day_col and brick_sales[day_col].notna().any() else 365
    n_skus = brick_sales[sku_col].nunique()
    ros = brick_sales[qty_col].sum() / (n_skus * n_days) if n_skus > 0 else 1.0

    return base_qty, ros


# ══════════════════════════════════════════════════════════════════════
#  Full Pipeline
# ══════════════════════════════════════════════════════════════════════

def run_quantity_estimation(
    enriched_ra: pd.DataFrame,
    sales_df: pd.DataFrame,
    config: FTFConfig,
    factor_result: FactorAnalysisResult = None,
    catalog_df: pd.DataFrame = None,
) -> QuantityEstimationResult:
    """Execute Steps 6 + 7: estimate quantities for all Enriched RA items."""

    df = enriched_ra.copy()
    result = QuantityEstimationResult()

    # ── Step 6: FTF Approved ──────────────────────────────────────────
    perf_scores = compute_performance_scores(df, sales_df, config)
    df["perf_score"] = perf_scores

    df = adjust_approved_quantities(df, perf_scores, config)
    result.approved_adjustments = (df["ftf_status"] == "APPROVED").sum()

    # Copy ORIGINAL quantities through
    orig_mask = df["ftf_status"] == "ORIGINAL"
    df.loc[orig_mask, "adjusted_quantity"] = df.loc[orig_mask, "original_quantity"]

    # ── Step 7: FTF Added ─────────────────────────────────────────────
    added_mask = df["ftf_status"] == "ADDED"
    added_idx = df[added_mask].index

    if factor_result is None:
        from src.module1.factor_analysis import FactorAnalysisResult as FAR
        factor_result = FAR()

    for idx in added_idx:
        item = df.loc[idx]

        # 7a: Nearest neighbor
        base_qty, projected_ros, method = _nearest_neighbor_estimate(
            item, sales_df, catalog_df, factor_result, config,
        )

        if base_qty is None:
            # 7d: Fallback
            base_qty, projected_ros = _family_fallback(item, sales_df, config)
            method = "family_fallback"
            result.fallback_count += 1

        # 7b: Trend multiplier
        ts = pd.to_numeric(item.get("trend_score"), errors="coerce")
        ts = 0.5 if pd.isna(ts) else min(1.0, max(0.0, ts))
        conf = str(item.get("trend_confidence", "Medium"))

        adjusted_qty = _apply_trend_multiplier(base_qty, ts, conf, config)

        # 7c: Shelf-life cap
        if projected_ros and projected_ros > 0:
            max_qty = (projected_ros * config.shelf_life_days) / config.str_target
            if adjusted_qty > max_qty:
                adjusted_qty = round(max_qty)
                method += " +shelf_life_cap"
                result.shelf_life_caps_applied += 1

        df.at[idx, "adjusted_quantity"] = adjusted_qty
        df.at[idx, "adjustment_reason"] = f"method={method}, base={base_qty:.0f}, trend_mult applied"
        result.added_estimates += 1

    # ── Shelf-life cap for APPROVED (Step 6c) ─────────────────────────
    df, caps = apply_shelf_life_cap(df, sales_df, config)
    result.shelf_life_caps_applied += caps

    result.final_ra = df
    logger.info(result.summary())
    return result
