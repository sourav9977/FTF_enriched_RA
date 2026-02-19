"""Step 3: Factor Analysis — Data-driven attribute weighting per Brick.

Determines which product attributes (pattern, color, fit, sleeve, neck,
length) drive sales performance within each Brick, so overlap matching
and nearest-neighbor calculations use data-driven weights, not uniform ones.

Pipeline:
  3a. Build attribute matrix (SKUs × binary-encoded attributes) per Brick
  3b. Correlation analysis (Pearson of each attribute column vs NSV)
  3c. Calculate importance weights (|r_i| / Σ|r_j|, normalized to 1.0)
  3d. Output weight vector per Brick
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.pipeline_logger import get_logger

logger = get_logger(__name__)

MATCHABLE_ATTRIBUTES = ["pattern", "color", "fit", "sleeve", "neck_type", "length"]

# Minimum SKUs per brick to produce reliable weights
MIN_SKUS_PER_BRICK = 10

# Minimum distinct values in an attribute to include it (if only 1 value, no variance)
MIN_DISTINCT_VALUES = 2


@dataclass
class BrickWeights:
    """Weight vector for a single Brick, with audit metadata."""
    brick: str
    weights: Dict[str, float]
    correlations: Dict[str, float]
    sample_size: int
    attribute_coverage: Dict[str, float]     # % non-null per attribute
    calculation_date: str = field(default_factory=lambda: datetime.now().isoformat())
    method: str = "pearson_correlation"

    def to_dict(self) -> dict:
        return {
            "brick": self.brick,
            "weights": self.weights,
            "correlations": self.correlations,
            "sample_size": self.sample_size,
            "attribute_coverage": self.attribute_coverage,
            "calculation_date": self.calculation_date,
            "method": self.method,
        }


@dataclass
class FactorAnalysisResult:
    """Full result of factor analysis across all Bricks."""
    brick_weights: Dict[str, BrickWeights] = field(default_factory=dict)
    skipped_bricks: Dict[str, str] = field(default_factory=dict)  # brick → reason

    def get_weights(self, brick: str) -> Optional[Dict[str, float]]:
        bw = self.brick_weights.get(brick)
        return bw.weights if bw else None

    def summary(self) -> str:
        lines = [
            f"Factor Analysis: {len(self.brick_weights)} bricks computed, "
            f"{len(self.skipped_bricks)} skipped",
            ""
        ]
        for brick, bw in sorted(self.brick_weights.items()):
            w_str = ", ".join(f"{k}={v:.3f}" for k, v in sorted(bw.weights.items(), key=lambda x: -x[1]))
            lines.append(f"  {brick:20s} (n={bw.sample_size:>4d})  {w_str}")
        if self.skipped_bricks:
            lines.append("")
            for brick, reason in sorted(self.skipped_bricks.items()):
                lines.append(f"  {brick:20s} SKIPPED: {reason}")
        return "\n".join(lines)


# ── Step 3a: Build Attribute Matrix ───────────────────────────────────

def _prepare_sku_data(sales_df: pd.DataFrame,
                      catalog_df: pd.DataFrame) -> pd.DataFrame:
    """Join sales and catalog to get SKU-level attributes with NSV.

    Strategy:
    1. Aggregate sales to SKU level (sum of revenue = NSV).
    2. Normalize join keys (convert both to string, strip whitespace).
    3. Inner-join with catalog on SKU.
    4. For attributes present in both, prefer catalog (richer) but
       fall back to sales columns (brick, color) when catalog coverage
       is poorer.
    """
    # Resolve column names
    sku_col = "sku" if "sku" in sales_df.columns else "sku_code"
    rev_col = next((c for c in ["revenue", "nsv"] if c in sales_df.columns), None)

    if sku_col not in sales_df.columns:
        raise ValueError(f"Sales data missing SKU column. Available: {list(sales_df.columns)}")

    # Keep sales-side attributes that might be useful
    sales_attr_cols = [c for c in ["brick", "color", "segment", "family", "class_"]
                       if c in sales_df.columns]

    agg_dict = {}
    if rev_col:
        agg_dict[rev_col] = "sum"
    if "quantity" in sales_df.columns:
        agg_dict["quantity"] = "sum"
    for ac in sales_attr_cols:
        agg_dict[ac] = "first"

    if not any(k in agg_dict for k in [rev_col, "quantity"] if k):
        raise ValueError(f"Sales data missing revenue/quantity columns. Available: {list(sales_df.columns)}")

    sku_sales = sales_df.groupby(sku_col).agg(agg_dict).reset_index()
    nsv_col = rev_col or "quantity"
    sku_sales = sku_sales.rename(columns={sku_col: "sku_code", nsv_col: "nsv"})
    sku_sales["sku_code"] = sku_sales["sku_code"].astype(str).str.strip()

    # Prepare catalog
    cat_sku_col = "sku_code" if "sku_code" in catalog_df.columns else "sku"
    catalog_slim = catalog_df.drop_duplicates(subset=[cat_sku_col]).copy()
    if cat_sku_col != "sku_code":
        catalog_slim = catalog_slim.rename(columns={cat_sku_col: "sku_code"})
    catalog_slim["sku_code"] = catalog_slim["sku_code"].astype(str).str.strip()

    merged = sku_sales.merge(catalog_slim, on="sku_code", how="inner",
                             suffixes=("_sales", "_cat"))

    # Reconcile duplicated columns: prefer catalog, fall back to sales
    for attr in ["brick", "color", "segment", "family"]:
        cat_col = f"{attr}_cat"
        sales_col = f"{attr}_sales"
        if cat_col in merged.columns and sales_col in merged.columns:
            merged[attr] = merged[cat_col].fillna(merged[sales_col])
            merged.drop(columns=[cat_col, sales_col], inplace=True)
        elif cat_col in merged.columns:
            merged.rename(columns={cat_col: attr}, inplace=True)
        elif sales_col in merged.columns:
            merged.rename(columns={sales_col: attr}, inplace=True)

    logger.info(
        f"SKU join: {len(sku_sales)} sales SKUs × {len(catalog_slim)} catalog SKUs "
        f"→ {len(merged)} matched ({len(merged)/max(len(sku_sales),1):.0%} coverage)"
    )

    return merged


def _build_attribute_matrix(sku_data: pd.DataFrame,
                            brick: str,
                            attributes: List[str] = None
                            ) -> Tuple[pd.DataFrame, pd.Series]:
    """Build binary-encoded attribute matrix for a specific Brick.

    Returns (X, y) where X is the binary attribute matrix and y is NSV.
    """
    if attributes is None:
        attributes = MATCHABLE_ATTRIBUTES

    brick_data = sku_data[sku_data["brick"].str.upper() == brick.upper()].copy()

    if len(brick_data) < MIN_SKUS_PER_BRICK:
        return pd.DataFrame(), pd.Series(dtype=float)

    y = brick_data["nsv"].astype(float)

    dummies_list = []
    used_attributes = []

    for attr in attributes:
        if attr not in brick_data.columns:
            logger.debug(f"  [{brick}] attr '{attr}' not in data — skipped")
            continue
        col = brick_data[attr].copy()
        non_null_pct = col.notna().mean()
        if non_null_pct < 0.1:
            logger.debug(f"  [{brick}] attr '{attr}' only {non_null_pct:.0%} non-null — skipped")
            continue

        col = col.fillna("UNKNOWN")
        col = col.astype(str).str.upper().str.strip()
        n_distinct = col.nunique()

        if n_distinct < MIN_DISTINCT_VALUES:
            logger.debug(f"  [{brick}] attr '{attr}' only {n_distinct} distinct value(s) — skipped")
            continue

        dummies = pd.get_dummies(col, prefix=attr, dtype=int)
        dummies_list.append(dummies)
        used_attributes.append(attr)

    if not dummies_list:
        return pd.DataFrame(), pd.Series(dtype=float)

    X = pd.concat(dummies_list, axis=1)
    X.index = y.index

    return X, y


# ── Step 3b: Correlation Analysis ─────────────────────────────────────

def _compute_correlations(X: pd.DataFrame,
                          y: pd.Series) -> Dict[str, float]:
    """Compute Pearson correlation of each binary column with NSV."""
    correlations = {}
    for col in X.columns:
        if X[col].std() == 0:
            correlations[col] = 0.0
            continue
        corr = X[col].corr(y)
        correlations[col] = corr if not np.isnan(corr) else 0.0
    return correlations


# ── Step 3c: Calculate Importance Weights ─────────────────────────────

def _aggregate_to_attribute_weights(correlations: Dict[str, float],
                                    attributes: List[str] = None
                                    ) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Aggregate binary-column correlations up to attribute-level weights.

    For each attribute (e.g., "pattern"), sum the |r| across all its
    binary columns (pattern_SOLID, pattern_STRIPES, etc.), then normalize.

    Returns (weights, attribute_correlations).
    """
    if attributes is None:
        attributes = MATCHABLE_ATTRIBUTES

    attr_abs_corr = {}
    for attr in attributes:
        prefix = f"{attr}_"
        relevant = {k: v for k, v in correlations.items() if k.startswith(prefix)}
        if relevant:
            attr_abs_corr[attr] = sum(abs(v) for v in relevant.values())

    total = sum(attr_abs_corr.values())

    if total == 0:
        n = len(attr_abs_corr) if attr_abs_corr else 1
        weights = {attr: 1.0 / n for attr in attr_abs_corr}
        return weights, attr_abs_corr

    weights = {attr: val / total for attr, val in attr_abs_corr.items()}

    return weights, attr_abs_corr


# ── Step 3d: Full Pipeline ────────────────────────────────────────────

def compute_brick_weights(sku_data: pd.DataFrame,
                          brick: str,
                          attributes: List[str] = None,
                          min_skus: int = MIN_SKUS_PER_BRICK,
                          ) -> Optional[BrickWeights]:
    """Compute the full weight vector for a single Brick.

    Returns BrickWeights or None if insufficient data.
    """
    if attributes is None:
        attributes = MATCHABLE_ATTRIBUTES

    X, y = _build_attribute_matrix(sku_data, brick, attributes)

    if X.empty or len(y) < min_skus:
        return None

    correlations = _compute_correlations(X, y)
    weights, attr_corrs = _aggregate_to_attribute_weights(correlations, attributes)

    # Attribute coverage
    brick_data = sku_data[sku_data["brick"].str.upper() == brick.upper()]
    coverage = {}
    for attr in attributes:
        if attr in brick_data.columns:
            coverage[attr] = round(brick_data[attr].notna().mean(), 3)
        else:
            coverage[attr] = 0.0

    bw = BrickWeights(
        brick=brick,
        weights=weights,
        correlations=attr_corrs,
        sample_size=len(y),
        attribute_coverage=coverage,
    )
    logger.debug(
        f"  [{brick}] n={len(y)}, weights: "
        + ", ".join(f"{k}={v:.3f}" for k, v in sorted(weights.items(), key=lambda x: -x[1]))
    )
    for attr, cov in coverage.items():
        logger.debug(f"    {attr}: coverage={cov:.0%}, |r|={attr_corrs.get(attr, 0):.4f}")
    return bw


def run_factor_analysis(sales_df: pd.DataFrame,
                        catalog_df: pd.DataFrame,
                        attributes: List[str] = None,
                        min_skus: int = MIN_SKUS_PER_BRICK,
                        ) -> FactorAnalysisResult:
    """Run factor analysis across all Bricks in the data.

    Input: Historical Sales Data + Product Catalog (at SKU level).
    Output: FactorAnalysisResult with weight vectors per Brick.
    """
    if attributes is None:
        attributes = MATCHABLE_ATTRIBUTES

    sku_data = _prepare_sku_data(sales_df, catalog_df)

    if "brick" not in sku_data.columns:
        raise ValueError("Joined data missing 'brick' column — check catalog has brick mapped")

    result = FactorAnalysisResult()
    bricks = sku_data["brick"].dropna().str.upper().unique()

    for brick in sorted(bricks):
        brick_mask = sku_data["brick"].str.upper() == brick
        brick_count = brick_mask.sum()
        if brick_count < min_skus:
            result.skipped_bricks[brick] = f"Only {brick_count} SKUs (need {min_skus})"
            continue

        bw = compute_brick_weights(sku_data, brick, attributes)
        if bw is None:
            result.skipped_bricks[brick] = "Insufficient attribute variance"
        else:
            result.brick_weights[brick] = bw

    logger.info(
        f"Factor analysis complete: {len(result.brick_weights)} bricks, "
        f"{len(result.skipped_bricks)} skipped"
    )
    return result


def run_factor_analysis_sales_only(sales_df: pd.DataFrame,
                                    attributes: List[str] = None,
                                    min_skus: int = MIN_SKUS_PER_BRICK,
                                    ) -> FactorAnalysisResult:
    """Fallback: run factor analysis using only sales-side attributes.

    Useful when catalog SKU overlap is too low. Sales data may already
    have brick, color, segment etc. from the original source. Attributes
    not present in sales will receive uniform weights.
    """
    if attributes is None:
        attributes = MATCHABLE_ATTRIBUTES

    sku_col = "sku" if "sku" in sales_df.columns else "sku_code"
    rev_col = next((c for c in ["revenue", "nsv"] if c in sales_df.columns), None)

    agg_dict = {}
    if rev_col:
        agg_dict[rev_col] = "sum"
    elif "quantity" in sales_df.columns:
        agg_dict["quantity"] = "sum"
    else:
        raise ValueError("Sales data needs revenue or quantity for factor analysis")

    for attr in attributes + ["brick"]:
        if attr in sales_df.columns:
            agg_dict[attr] = "first"

    sku_data = sales_df.groupby(sku_col).agg(agg_dict).reset_index()
    nsv_col = rev_col or "quantity"
    sku_data = sku_data.rename(columns={nsv_col: "nsv"})

    if "brick" not in sku_data.columns:
        raise ValueError("Sales data missing 'brick' column for sales-only factor analysis")

    result = FactorAnalysisResult()
    bricks = sku_data["brick"].dropna().str.upper().unique()

    available_attrs = [a for a in attributes if a in sku_data.columns]
    logger.info(f"Sales-only mode: attributes available = {available_attrs}")

    for brick in sorted(bricks):
        brick_mask = sku_data["brick"].str.upper() == brick
        if brick_mask.sum() < min_skus:
            result.skipped_bricks[brick] = f"Only {brick_mask.sum()} SKUs (need {min_skus})"
            continue

        bw = compute_brick_weights(sku_data, brick, available_attrs, min_skus)
        if bw is None:
            result.skipped_bricks[brick] = "Insufficient attribute variance"
        else:
            bw.method = "pearson_correlation (sales-only)"
            result.brick_weights[brick] = bw

    logger.info(
        f"Sales-only factor analysis: {len(result.brick_weights)} bricks, "
        f"{len(result.skipped_bricks)} skipped"
    )
    return result


def get_default_weights(attributes: List[str] = None) -> Dict[str, float]:
    """Return uniform weights as fallback when factor analysis can't run."""
    if attributes is None:
        attributes = MATCHABLE_ATTRIBUTES
    n = len(attributes)
    return {attr: round(1.0 / n, 4) for attr in attributes}
