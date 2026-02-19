"""Step 4: Enriched RA Generation — Overlap Scoring, Classification, Assembly.

Matches RA items against Trend items within each Brick, computes confidence
scores, classifies items into FTF APPROVED / ORIGINAL / FTF ADDED, and
assembles the Enriched RA that bridges Module 1 and Module 2.

Sub-steps:
  4a. Trend-RA Overlap Scoring (weighted attribute matching)
  4b. Confidence Score Calculation (RA confidence + FTF confidence)
  4c. Three-Tier Classification Logic
  4d. FTF Added Item Generation (from unmatched trends)
  4e. Assemble Enriched RA
"""

import logging
import uuid
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import FTFConfig
from src.module1.ingest import get_mapping
from src.module1.factor_analysis import (
    FactorAnalysisResult, get_default_weights, MATCHABLE_ATTRIBUTES,
)

logger = logging.getLogger(__name__)

# Attributes used in overlap matching (RA name → Trend name from YAML)
_OVERLAP_ATTRS_RA = ["pattern_type", "color", "fit", "sleeve_type", "neck_type", "length"]

# RA attribute name → factor-analysis weight key
_RA_ATTR_TO_WEIGHT_KEY = {
    "pattern_type": "pattern",
    "color": "color",
    "fit": "fit",
    "sleeve_type": "sleeve",
    "neck_type": "neck_type",
    "length": "length",
}

CONFIDENCE_NUMERIC = {"HIGH": 1.0, "MEDIUM": 0.6, "LOW": 0.3}
LIFECYCLE_SCORE = {"RISE": 1.0, "FLAT": 0.6, "DECLINE": 0.2}


# ══════════════════════════════════════════════════════════════════════
#  Step 4a: Trend-RA Overlap Scoring
# ══════════════════════════════════════════════════════════════════════

def _get_ra_to_trend_map() -> Dict[str, str]:
    """Return the RA→Trend attribute name mapping from YAML."""
    return get_mapping().ra_to_trend_map


def compute_overlap_score(
    ra_row: pd.Series,
    trend_row: pd.Series,
    weights: Dict[str, float],
    ra_to_trend: Dict[str, str],
) -> float:
    """Compute weighted overlap score for a single RA×Trend pair.

    match_i = 1 if attributes match OR RA attribute is NULL (wildcard)
    Overlap_Score = Σ(w_i × match_i) / Σ(w_i)
    """
    total_w = 0.0
    matched_w = 0.0

    for ra_attr in _OVERLAP_ATTRS_RA:
        trend_attr = ra_to_trend.get(ra_attr)
        if trend_attr is None:
            continue

        weight_key = _RA_ATTR_TO_WEIGHT_KEY.get(ra_attr, ra_attr)
        w = weights.get(weight_key, 0.0)
        if w == 0.0:
            continue

        total_w += w

        ra_val = ra_row.get(ra_attr)
        trend_val = trend_row.get(trend_attr)

        # NULL RA value = wildcard (auto-match)
        if pd.isna(ra_val) or str(ra_val).strip() == "":
            matched_w += w
            continue

        # Compare normalized values
        if pd.notna(trend_val):
            if str(ra_val).strip().upper() == str(trend_val).strip().upper():
                matched_w += w

    if total_w == 0:
        return 0.0
    return matched_w / total_w


def score_all_overlaps(
    ra_df: pd.DataFrame,
    trends_df: pd.DataFrame,
    factor_result: FactorAnalysisResult,
    ra_to_trend: Dict[str, str] = None,
) -> pd.DataFrame:
    """Score every RA×Trend pair within each Brick.

    Returns a DataFrame with columns:
      ra_idx, trend_idx, brick, overlap_score
    sorted by ra_idx, overlap_score desc.
    """
    if ra_to_trend is None:
        ra_to_trend = _get_ra_to_trend_map()

    results = []

    ra_bricks = ra_df["brick"].str.upper().fillna("")
    trend_bricks = trends_df["brick"].str.upper().fillna("")

    for brick in ra_bricks.unique():
        if brick == "":
            continue

        weights = factor_result.get_weights(brick) or get_default_weights()

        ra_mask = ra_bricks == brick
        trend_mask = trend_bricks == brick

        ra_subset = ra_df[ra_mask]
        trend_subset = trends_df[trend_mask]

        if trend_subset.empty:
            continue

        for ra_idx, ra_row in ra_subset.iterrows():
            for trend_idx, trend_row in trend_subset.iterrows():
                score = compute_overlap_score(ra_row, trend_row, weights, ra_to_trend)
                results.append({
                    "ra_idx": ra_idx,
                    "trend_idx": trend_idx,
                    "brick": brick,
                    "overlap_score": score,
                })

    if not results:
        return pd.DataFrame(columns=["ra_idx", "trend_idx", "brick", "overlap_score"])

    df = pd.DataFrame(results)
    df = df.sort_values(["ra_idx", "overlap_score"], ascending=[True, False])
    return df


def assign_best_trends(overlap_scores: pd.DataFrame) -> pd.DataFrame:
    """For each RA item, pick the trend with the highest overlap score.

    Returns DataFrame with columns: ra_idx, trend_idx, brick, overlap_score
    (one row per RA item that has at least one trend match).
    """
    if overlap_scores.empty:
        return overlap_scores
    return overlap_scores.groupby("ra_idx").first().reset_index()


# ══════════════════════════════════════════════════════════════════════
#  Step 4b: Confidence Score Calculation
# ══════════════════════════════════════════════════════════════════════

def compute_ra_confidence(
    ra_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    config: FTFConfig,
) -> pd.Series:
    """Compute RA Confidence Score for each RA item.

    RA_Confidence = w_sales × norm(past_sales_contribution)
                  + w_ros   × norm(ROS)
                  + w_str   × norm(STR)

    When historical sales data can't be joined (no style/attribute match),
    returns 0.5 as neutral confidence.
    """
    w = config.ra_confidence_weights
    confidence = pd.Series(0.5, index=ra_df.index, name="ra_confidence_score")

    if sales_df is None or sales_df.empty:
        logger.warning("No sales data for RA confidence — using neutral 0.5")
        return confidence

    # Aggregate sales to brick × attribute level for contribution/ROS/STR
    sku_col = "sku" if "sku" in sales_df.columns else "sku_code"
    rev_col = next((c for c in ["revenue", "nsv"] if c in sales_df.columns), None)
    qty_col = "quantity" if "quantity" in sales_df.columns else None

    if rev_col is None and qty_col is None:
        logger.warning("Sales data lacks revenue & quantity — using neutral 0.5")
        return confidence

    # Build brick-level aggregates from sales
    if "brick" not in sales_df.columns:
        logger.warning("Sales data lacks 'brick' — using neutral 0.5")
        return confidence

    brick_agg = {}
    if rev_col:
        brick_sales = sales_df.groupby("brick")[rev_col].sum()
        brick_agg["nsv"] = brick_sales
    if qty_col:
        brick_qty = sales_df.groupby("brick")[qty_col].sum()
        brick_agg["qty"] = brick_qty

    for idx, ra_row in ra_df.iterrows():
        brick = str(ra_row.get("brick", "")).strip().upper()
        if not brick:
            continue

        scores = []

        # Past sales contribution (NSV share of brick)
        if "nsv" in brick_agg:
            total_nsv = brick_agg["nsv"].get(brick, 0.0)
            if total_nsv > 0:
                scores.append(w["w_sales"] * min(1.0, 0.5))
            else:
                scores.append(0.0)
        else:
            scores.append(0.0)

        # ROS proxy — we use average quantity per SKU as proxy
        if "qty" in brick_agg:
            total_qty = brick_agg["qty"].get(brick, 0.0)
            n_skus = sales_df[sales_df["brick"].str.upper() == brick][sku_col].nunique()
            if n_skus > 0:
                avg_ros = total_qty / n_skus
                max_ros = sales_df.groupby("brick").apply(
                    lambda g: g[qty_col].sum() / g[sku_col].nunique()
                ).max()
                norm_ros = avg_ros / max_ros if max_ros > 0 else 0.5
                scores.append(w["w_ros"] * min(1.0, norm_ros))
            else:
                scores.append(0.0)
        else:
            scores.append(0.0)

        # STR proxy — sell-through approximation
        scores.append(w["w_str"] * 0.5)

        confidence.at[idx] = sum(scores)

    # Clip to [0, 1]
    confidence = confidence.clip(0.0, 1.0)
    return confidence


def compute_ftf_confidence(
    trends_df: pd.DataFrame,
    config: FTFConfig,
    target_month_col: str = "m1_month",
) -> pd.Series:
    """Compute FTF Confidence Score for each Trend item.

    FTF_Confidence = w_trend     × trend_score
                   + w_confidence × confidence_numeric
                   + w_lifecycle  × lifecycle_score
    """
    w = config.ftf_confidence_weights
    confidence = pd.Series(0.0, index=trends_df.index, name="ftf_confidence_score")

    for idx, row in trends_df.iterrows():
        # Trend score (current_score, already 0-1)
        trend_score = pd.to_numeric(row.get("current_score", 0), errors="coerce")
        if pd.isna(trend_score):
            trend_score = 0.0
        trend_score = min(1.0, max(0.0, trend_score))

        # Confidence level → numeric
        conf_raw = str(row.get("confidence", "")).strip().upper()
        conf_numeric = CONFIDENCE_NUMERIC.get(conf_raw, 0.3)

        # Lifecycle stage at target month → score
        stage_raw = str(row.get("m1_stage", "")).strip().upper()
        lifecycle = LIFECYCLE_SCORE.get(stage_raw, 0.6)

        score = (
            w["w_trend"] * trend_score
            + w["w_confidence"] * conf_numeric
            + w["w_lifecycle"] * lifecycle
        )
        confidence.at[idx] = min(1.0, max(0.0, score))

    return confidence


# ══════════════════════════════════════════════════════════════════════
#  Step 4c: Three-Tier Classification
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ClassificationResult:
    """Holds classified RA items + generated FTF ADDED items."""
    approved: List[dict] = field(default_factory=list)     # FTF APPROVED
    original: List[dict] = field(default_factory=list)     # ORIGINAL (kept)
    added_replacements: List[dict] = field(default_factory=list)  # FTF ADDED (replacement)
    removed_ra: List[dict] = field(default_factory=list)   # RA items replaced

    def summary(self) -> str:
        return (
            f"Classification: {len(self.approved)} APPROVED, "
            f"{len(self.original)} ORIGINAL, "
            f"{len(self.added_replacements)} ADDED (replacement), "
            f"{len(self.removed_ra)} RA items removed"
        )


def classify_items(
    ra_df: pd.DataFrame,
    trends_df: pd.DataFrame,
    best_matches: pd.DataFrame,
    ra_confidence: pd.Series,
    ftf_confidence: pd.Series,
    config: FTFConfig,
) -> ClassificationResult:
    """Apply the three-tier classification to each RA item.

    TIER 1: overlap ≥ overlap_threshold       → FTF APPROVED
    TIER 2: replacement_thresh ≤ overlap < OT  → ORIGINAL
    TIER 3: overlap < replacement_threshold    → confidence-based decision
    """
    result = ClassificationResult()
    ra_to_trend = _get_ra_to_trend_map()
    trend_to_ra = get_mapping().trend_to_ra_map
    used_trend_indices = set()

    for _, match_row in best_matches.iterrows():
        ra_idx = match_row["ra_idx"]
        trend_idx = match_row["trend_idx"]
        score = match_row["overlap_score"]

        ra_row = ra_df.loc[ra_idx].to_dict()
        trend_row = trends_df.loc[trend_idx].to_dict()

        ra_conf = ra_confidence.get(ra_idx, 0.5)
        ftf_conf = ftf_confidence.get(trend_idx, 0.5)

        if score >= config.overlap_threshold:
            # TIER 1: FTF APPROVED
            item = _build_enriched_item(
                ra_row, trend_row, ra_to_trend, trend_to_ra,
                ftf_status="APPROVED",
                overlap_score=score,
                ra_confidence_score=ra_conf,
                ftf_confidence_score=ftf_conf,
            )
            result.approved.append(item)
            used_trend_indices.add(trend_idx)

        elif score >= config.replacement_threshold:
            # TIER 2: ORIGINAL (trend doesn't qualify)
            item = _build_enriched_item(
                ra_row, trend_row, ra_to_trend, trend_to_ra,
                ftf_status="ORIGINAL",
                overlap_score=score,
                ra_confidence_score=ra_conf,
                ftf_confidence_score=None,
            )
            result.original.append(item)

        else:
            # TIER 3: Confidence-based decision
            if ftf_conf > ra_conf:
                # Replace RA with trend-based FTF ADDED
                replacement = _build_replacement_item(
                    ra_row, trend_row, trend_to_ra,
                    overlap_score=score,
                    ra_confidence_score=ra_conf,
                    ftf_confidence_score=ftf_conf,
                )
                result.added_replacements.append(replacement)
                result.removed_ra.append(ra_row)
                used_trend_indices.add(trend_idx)
            else:
                # RA confidence wins — keep as ORIGINAL
                item = _build_enriched_item(
                    ra_row, trend_row, ra_to_trend, trend_to_ra,
                    ftf_status="ORIGINAL",
                    overlap_score=score,
                    ra_confidence_score=ra_conf,
                    ftf_confidence_score=ftf_conf,
                    retention_reason=f"RA confidence retained: RA={ra_conf:.2f} >= FTF={ftf_conf:.2f}",
                )
                result.original.append(item)

    # RA items with NO trend match at all → ORIGINAL
    matched_ra_indices = set(best_matches["ra_idx"])
    for idx in ra_df.index:
        if idx not in matched_ra_indices:
            ra_row = ra_df.loc[idx].to_dict()
            ra_conf = ra_confidence.get(idx, 0.5)
            item = _build_enriched_item(
                ra_row, None, ra_to_trend, trend_to_ra,
                ftf_status="ORIGINAL",
                overlap_score=None,
                ra_confidence_score=ra_conf,
                ftf_confidence_score=None,
            )
            result.original.append(item)

    result._used_trend_indices = used_trend_indices
    return result


# ══════════════════════════════════════════════════════════════════════
#  Step 4d: FTF Added from Unmatched Trends
# ══════════════════════════════════════════════════════════════════════

def generate_ftf_added_from_unmatched(
    trends_df: pd.DataFrame,
    ra_df: pd.DataFrame,
    overlap_scores: pd.DataFrame,
    ftf_confidence: pd.Series,
    used_trend_indices: set,
    config: FTFConfig,
) -> List[dict]:
    """Generate FTF ADDED items from trends that had no RA match.

    A trend is "unmatched" if no RA item scored above replacement_threshold
    for it AND it wasn't already used as a confidence-based replacement.
    """
    trend_to_ra = get_mapping().trend_to_ra_map

    # Find trends that were matched above replacement_threshold
    matched_trends = set()
    if not overlap_scores.empty:
        above_thresh = overlap_scores[overlap_scores["overlap_score"] >= config.replacement_threshold]
        matched_trends = set(above_thresh["trend_idx"].unique())

    # Candidates: not matched above threshold AND not used in replacements
    candidate_indices = []
    for idx in trends_df.index:
        if idx not in matched_trends and idx not in used_trend_indices:
            candidate_indices.append(idx)

    if not candidate_indices:
        return []

    candidates = trends_df.loc[candidate_indices].copy()
    candidates["ftf_conf"] = ftf_confidence.reindex(candidate_indices).values

    # Rank by FTF confidence
    candidates = candidates.sort_values("ftf_conf", ascending=False)

    # Cap: count(FTF_Added) / count(Total_RA) <= ftf_added_cap_pct
    max_added = int(len(ra_df) * config.ftf_added_cap_pct)

    added_items = []
    for _, trend_row in candidates.iterrows():
        if len(added_items) >= max_added:
            break

        brick = str(trend_row.get("brick", "")).strip().upper()
        if not brick:
            continue

        # Inherit context (segment, family, fashion_grade, etc.) from RA items in same brick
        context = _get_brick_context(ra_df, brick)

        item = _build_added_item_from_trend(
            trend_row.to_dict(), trend_to_ra, context,
            ftf_confidence_score=trend_row.get("ftf_conf", 0.0),
            source="UNMATCHED_TREND",
        )
        added_items.append(item)

    logger.info(
        f"FTF ADDED from unmatched trends: {len(added_items)} "
        f"(cap={max_added}, candidates={len(candidate_indices)})"
    )
    return added_items


# ══════════════════════════════════════════════════════════════════════
#  Step 4e: Assemble Enriched RA
# ══════════════════════════════════════════════════════════════════════

ENRICHED_RA_COLUMNS = [
    "ra_id", "unique_id", "replaced_ra_id",
    "primary_key_type", "primary_key_value",
    "business_unit", "sales_channel", "season",
    "segment", "family", "brand", "brick", "class_", "fashion_grade",
    "month_of_drop", "product_group",
    "mrp", "mrp_bucket",
    "fit", "neck_type", "pattern_type", "sleeve_type",
    "color", "length",
    "original_quantity",
    "ftf_status", "ftf_added_source",
    "overlap_score", "ra_confidence_score", "ftf_confidence_score",
    "replacement_reason", "retention_reason",
    "trend_id", "trend_name", "trend_score",
    "trend_stage", "trend_trajectory", "trend_confidence",
    "trend_business_label", "trend_risk_flag",
    "attribute_weights_used",
]


def assemble_enriched_ra(
    classification: ClassificationResult,
    ftf_added_unmatched: List[dict],
) -> pd.DataFrame:
    """Combine all item types into a single Enriched RA DataFrame."""
    all_items = (
        classification.approved
        + classification.original
        + classification.added_replacements
        + ftf_added_unmatched
    )

    if not all_items:
        return pd.DataFrame(columns=ENRICHED_RA_COLUMNS)

    df = pd.DataFrame(all_items)

    # Ensure all expected columns exist
    for col in ENRICHED_RA_COLUMNS:
        if col not in df.columns:
            df[col] = None

    df = df[ENRICHED_RA_COLUMNS]

    logger.info(
        f"Enriched RA assembled: {len(df)} items "
        f"(APPROVED={len(classification.approved)}, "
        f"ORIGINAL={len(classification.original)}, "
        f"ADDED_REPLACEMENT={len(classification.added_replacements)}, "
        f"ADDED_UNMATCHED={len(ftf_added_unmatched)})"
    )
    return df


# ══════════════════════════════════════════════════════════════════════
#  Full Step 4 Pipeline
# ══════════════════════════════════════════════════════════════════════

@dataclass
class EnrichmentResult:
    """Complete result of the Enriched RA generation pipeline."""
    enriched_ra: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    overlap_scores: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    classification: ClassificationResult = field(default_factory=ClassificationResult)
    ra_confidence: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    ftf_confidence: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    def summary(self) -> str:
        counts = self.enriched_ra["ftf_status"].value_counts().to_dict() if not self.enriched_ra.empty else {}
        lines = [
            f"Enriched RA: {len(self.enriched_ra)} total items",
            f"  APPROVED:  {counts.get('APPROVED', 0)}",
            f"  ORIGINAL:  {counts.get('ORIGINAL', 0)}",
            f"  ADDED:     {counts.get('ADDED', 0)}",
        ]
        if not self.overlap_scores.empty:
            lines.append(f"  Avg overlap score: {self.overlap_scores['overlap_score'].mean():.3f}")
        return "\n".join(lines)


def run_enrichment(
    ra_df: pd.DataFrame,
    trends_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    factor_result: FactorAnalysisResult,
    config: FTFConfig,
) -> EnrichmentResult:
    """Execute the full Step 4 pipeline.

    Returns EnrichmentResult containing the Enriched RA and all
    intermediate artifacts.
    """
    logger.info(f"Starting enrichment: {len(ra_df)} RA items × {len(trends_df)} trends")

    # 4a: Overlap scoring
    overlap_scores = score_all_overlaps(ra_df, trends_df, factor_result)
    best_matches = assign_best_trends(overlap_scores)
    logger.info(
        f"Overlap scoring: {len(overlap_scores)} pairs scored, "
        f"{len(best_matches)} RA items matched to trends"
    )

    # 4b: Confidence scores
    ra_confidence = compute_ra_confidence(ra_df, sales_df, config)
    ftf_confidence = compute_ftf_confidence(trends_df, config)

    # 4c: Classification
    classification = classify_items(
        ra_df, trends_df, best_matches,
        ra_confidence, ftf_confidence, config,
    )
    logger.info(classification.summary())

    # 4d: FTF Added from unmatched trends
    used_trends = getattr(classification, "_used_trend_indices", set())
    ftf_added_unmatched = generate_ftf_added_from_unmatched(
        trends_df, ra_df, overlap_scores,
        ftf_confidence, used_trends, config,
    )

    # 4e: Assemble
    enriched_ra = assemble_enriched_ra(classification, ftf_added_unmatched)

    return EnrichmentResult(
        enriched_ra=enriched_ra,
        overlap_scores=overlap_scores,
        classification=classification,
        ra_confidence=ra_confidence,
        ftf_confidence=ftf_confidence,
    )


# ══════════════════════════════════════════════════════════════════════
#  Helper Functions
# ══════════════════════════════════════════════════════════════════════

def _generate_ra_id() -> str:
    return f"RA-{uuid.uuid4().hex[:8].upper()}"


def _build_enriched_item(
    ra_row: dict,
    trend_row: Optional[dict],
    ra_to_trend: dict,
    trend_to_ra: dict,
    ftf_status: str,
    overlap_score: Optional[float],
    ra_confidence_score: Optional[float],
    ftf_confidence_score: Optional[float],
    retention_reason: str = None,
) -> dict:
    """Build a single row for the Enriched RA from an RA item."""
    item = {
        "ra_id": ra_row.get("ra_id") or _generate_ra_id(),
        "unique_id": ra_row.get("unique_id"),
        "replaced_ra_id": None,
        "primary_key_type": ra_row.get("primary_key_type"),
        "primary_key_value": ra_row.get("primary_key_value"),
        "business_unit": ra_row.get("business_unit"),
        "sales_channel": ra_row.get("sales_channel"),
        "season": ra_row.get("season"),
        "segment": ra_row.get("segment"),
        "family": ra_row.get("family"),
        "brand": ra_row.get("brand"),
        "brick": ra_row.get("brick"),
        "class_": ra_row.get("class_"),
        "fashion_grade": ra_row.get("fashion_grade"),
        "month_of_drop": ra_row.get("month_of_drop"),
        "product_group": ra_row.get("product_group"),
        "mrp": ra_row.get("mrp"),
        "mrp_bucket": ra_row.get("mrp_bucket"),
        "fit": ra_row.get("fit"),
        "neck_type": ra_row.get("neck_type"),
        "pattern_type": ra_row.get("pattern_type"),
        "sleeve_type": ra_row.get("sleeve_type"),
        "color": ra_row.get("color"),
        "length": ra_row.get("length"),
        "original_quantity": ra_row.get("quantities_planned"),
        "ftf_status": ftf_status,
        "ftf_added_source": None,
        "overlap_score": overlap_score,
        "ra_confidence_score": ra_confidence_score,
        "ftf_confidence_score": ftf_confidence_score,
        "replacement_reason": None,
        "retention_reason": retention_reason,
        "attribute_weights_used": None,
    }

    if trend_row and ftf_status == "APPROVED":
        item.update(_extract_trend_metadata(trend_row))

    return item


def _build_replacement_item(
    ra_row: dict,
    trend_row: dict,
    trend_to_ra: dict,
    overlap_score: float,
    ra_confidence_score: float,
    ftf_confidence_score: float,
) -> dict:
    """Build an FTF ADDED item that replaces an RA item."""
    item = {
        "ra_id": _generate_ra_id(),
        "unique_id": None,
        "replaced_ra_id": ra_row.get("unique_id") or ra_row.get("ra_id"),
        "primary_key_type": ra_row.get("primary_key_type"),
        "primary_key_value": ra_row.get("primary_key_value"),
        "business_unit": ra_row.get("business_unit"),
        "sales_channel": ra_row.get("sales_channel"),
        "season": ra_row.get("season"),
        "segment": ra_row.get("segment"),
        "family": ra_row.get("family"),
        "brand": ra_row.get("brand"),
        "brick": str(trend_row.get("brick", ra_row.get("brick", ""))).strip().upper(),
        "class_": ra_row.get("class_"),
        "fashion_grade": ra_row.get("fashion_grade"),
        "month_of_drop": ra_row.get("month_of_drop"),
        "product_group": ra_row.get("product_group"),
        "mrp": None,
        "mrp_bucket": None,
        # Attributes from trend
        "fit": trend_row.get("style"),
        "neck_type": trend_row.get("neck_type"),
        "pattern_type": trend_row.get("print_"),
        "sleeve_type": trend_row.get("sleeve"),
        "color": trend_row.get("color"),
        "length": trend_row.get("length"),
        "original_quantity": None,
        "ftf_status": "ADDED",
        "ftf_added_source": "CONFIDENCE_REPLACEMENT",
        "overlap_score": overlap_score,
        "ra_confidence_score": ra_confidence_score,
        "ftf_confidence_score": ftf_confidence_score,
        "replacement_reason": (
            f"Confidence replacement: FTF={ftf_confidence_score:.2f} > RA={ra_confidence_score:.2f}"
        ),
        "retention_reason": None,
    }
    item.update(_extract_trend_metadata(trend_row))
    return item


def _build_added_item_from_trend(
    trend_row: dict,
    trend_to_ra: dict,
    context: dict,
    ftf_confidence_score: float,
    source: str = "UNMATCHED_TREND",
) -> dict:
    """Build an FTF ADDED item from an unmatched trend."""
    item = {
        "ra_id": _generate_ra_id(),
        "unique_id": None,
        "replaced_ra_id": None,
        "primary_key_type": context.get("primary_key_type"),
        "primary_key_value": context.get("primary_key_value"),
        "business_unit": context.get("business_unit"),
        "sales_channel": context.get("sales_channel"),
        "season": context.get("season"),
        "segment": context.get("segment"),
        "family": context.get("family"),
        "brand": context.get("brand"),
        "brick": str(trend_row.get("brick", "")).strip().upper(),
        "class_": context.get("class_"),
        "fashion_grade": context.get("fashion_grade"),
        "month_of_drop": context.get("month_of_drop"),
        "product_group": context.get("product_group"),
        "mrp": None,
        "mrp_bucket": None,
        # Attributes from trend
        "fit": trend_row.get("style"),
        "neck_type": trend_row.get("neck_type"),
        "pattern_type": trend_row.get("print_"),
        "sleeve_type": trend_row.get("sleeve"),
        "color": trend_row.get("color"),
        "length": trend_row.get("length"),
        "original_quantity": None,
        "ftf_status": "ADDED",
        "ftf_added_source": source,
        "overlap_score": None,
        "ra_confidence_score": None,
        "ftf_confidence_score": ftf_confidence_score,
        "replacement_reason": None,
        "retention_reason": None,
    }
    item.update(_extract_trend_metadata(trend_row))
    return item


def _extract_trend_metadata(trend_row: dict) -> dict:
    """Extract standard trend fields for the enriched RA."""
    return {
        "trend_id": trend_row.get("trend_id") or trend_row.get("trend_name"),
        "trend_name": trend_row.get("trend_name"),
        "trend_score": pd.to_numeric(trend_row.get("current_score"), errors="coerce"),
        "trend_stage": trend_row.get("m1_stage"),
        "trend_trajectory": trend_row.get("trajectory"),
        "trend_confidence": trend_row.get("confidence"),
        "trend_business_label": trend_row.get("business_label"),
        "trend_risk_flag": None,
    }


def _get_brick_context(ra_df: pd.DataFrame, brick: str) -> dict:
    """Get inherited context fields from existing RA items in a Brick."""
    brick_items = ra_df[ra_df["brick"].str.upper() == brick]
    if brick_items.empty:
        return {}

    first = brick_items.iloc[0]
    return {
        "primary_key_type": first.get("primary_key_type"),
        "primary_key_value": first.get("primary_key_value"),
        "business_unit": first.get("business_unit"),
        "sales_channel": first.get("sales_channel"),
        "season": first.get("season"),
        "segment": first.get("segment"),
        "family": first.get("family"),
        "brand": first.get("brand"),
        "class_": first.get("class_"),
        "fashion_grade": first.get("fashion_grade"),
        "month_of_drop": first.get("month_of_drop"),
        "product_group": first.get("product_group"),
    }
