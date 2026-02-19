"""Step 0: Configuration — frozen config object for a prototype run.

All system parameters are loaded, validated, and frozen before any
processing begins. The config auto-detects the RA input level (Brand
vs Category) during ingestion and locks the primary key for the run.
"""

from dataclasses import dataclass, field, fields
from typing import Dict, Optional


# ── Defaults from Prototype Guide ────────────────────────────────────

DEFAULTS = {
    "overlap_threshold": 1.0,
    "replacement_threshold": 0.60,
    "ftf_added_cap_pct": 0.20,
    "max_adjustment_pct": 0.10,
    "moq_per_option": 2500,
    "shelf_life_days": 60,
    "str_target": 0.75,
    "min_nearest_neighbors": 3,
    "min_sales_days": 30,
    "k_neighbors": 5,
}

PERFORMANCE_WEIGHTS = {
    "ros": 0.30,
    "str": 0.25,
    "turns": 0.15,
    "contribution": 0.15,
    "trend_score": 0.15,
}

TREND_MULTIPLIER_PARAMS = {
    "base": 0.7,
    "scale": 0.6,
    "confidence_factors": {"High": 1.0, "Medium": 0.8, "Low": 0.6},
}

REBALANCING_WEIGHTS = {
    "past_sales": 0.40,
    "gm": 0.30,
    "trend": 0.20,
    "mix": 0.10,
}

FALLBACK_PERCENTILES = {
    "low": 25,       # trend_score < 0.5
    "mid": 50,       # 0.5 <= trend_score < 0.7
    "high": 75,      # trend_score >= 0.7
}

RA_CONFIDENCE_WEIGHTS = {
    "w_sales": 0.35,
    "w_ros": 0.35,
    "w_str": 0.30,
}

FTF_CONFIDENCE_WEIGHTS = {
    "w_trend": 0.40,
    "w_confidence": 0.35,
    "w_lifecycle": 0.25,
}

AOP_BREAKDOWN_PARAMS = {
    "lookback_months": 12,
    "same_month_ly_weight": 0.6,
    "rolling_average_weight": 0.4,
    "trend_uplift_factor_max": 0.10,
}

FASHION_GRADE_MAP = {
    "APL-FASHION": "FASHION",
    "APL-CORE": "CORE",
    "APL-ULTIMATE": "ULTIMATE",
    "APL-FAST": "FAST FASHION",
}


# ── Config Dataclass ─────────────────────────────────────────────────

@dataclass
class FTFConfig:
    """Frozen configuration for a single prototype run."""

    # Primary key — auto-detected from RA, locked for the run
    primary_key_type: Optional[str] = None          # "BRAND" or "CATEGORY"
    primary_key_value_field: Optional[str] = None    # resolved column name
    level_transformation_needed: bool = False

    # Matching thresholds
    overlap_threshold: float = 1.0
    replacement_threshold: float = 0.60

    # Caps
    ftf_added_cap_pct: float = 0.20

    # Quantity adjustment
    max_adjustment_pct: float = 0.10

    # MOQ
    moq_per_option: int = 2500

    # Shelf life
    shelf_life_days: int = 60
    str_target: float = 0.75

    # Nearest neighbor
    min_nearest_neighbors: int = 3
    min_sales_days: int = 30
    k_neighbors: int = 5

    # Composite weights (stored as dicts)
    performance_weights: Dict[str, float] = field(default_factory=lambda: PERFORMANCE_WEIGHTS.copy())
    trend_multiplier_params: Dict = field(default_factory=lambda: TREND_MULTIPLIER_PARAMS.copy())
    rebalancing_weights: Dict[str, float] = field(default_factory=lambda: REBALANCING_WEIGHTS.copy())
    fallback_percentiles: Dict[str, int] = field(default_factory=lambda: FALLBACK_PERCENTILES.copy())
    ra_confidence_weights: Dict[str, float] = field(default_factory=lambda: RA_CONFIDENCE_WEIGHTS.copy())
    ftf_confidence_weights: Dict[str, float] = field(default_factory=lambda: FTF_CONFIDENCE_WEIGHTS.copy())
    aop_breakdown_params: Dict = field(default_factory=lambda: AOP_BREAKDOWN_PARAMS.copy())

    # Fashion grade mapping
    fashion_grade_map: Dict[str, str] = field(default_factory=lambda: FASHION_GRADE_MAP.copy())

    # Output level toggle — if different from input level, flag transformation
    output_level: Optional[str] = None  # "BRAND" or "CATEGORY" or None (same as input)

    def __post_init__(self):
        self.validate()

    def validate(self):
        """Validate configuration constraints per the prototype guide."""
        errors = []

        if self.replacement_threshold > self.overlap_threshold:
            errors.append(
                f"replacement_threshold ({self.replacement_threshold}) "
                f"must be <= overlap_threshold ({self.overlap_threshold})"
            )

        if not (0.50 <= self.str_target <= 1.00):
            errors.append(
                f"str_target ({self.str_target}) must be between 0.50 and 1.00"
            )

        if not (0.0 <= self.overlap_threshold <= 1.0):
            errors.append(
                f"overlap_threshold ({self.overlap_threshold}) must be between 0.0 and 1.0"
            )

        if not (0.0 <= self.replacement_threshold <= 1.0):
            errors.append(
                f"replacement_threshold ({self.replacement_threshold}) must be between 0.0 and 1.0"
            )

        if not (0.0 <= self.ftf_added_cap_pct <= 1.0):
            errors.append(
                f"ftf_added_cap_pct ({self.ftf_added_cap_pct}) must be between 0.0 and 1.0"
            )

        if self.moq_per_option < 0:
            errors.append(f"moq_per_option ({self.moq_per_option}) must be >= 0")

        for name, weights in [
            ("performance_weights", self.performance_weights),
            ("rebalancing_weights", self.rebalancing_weights),
        ]:
            total = sum(weights.values())
            if abs(total - 1.0) > 0.01:
                errors.append(f"{name} must sum to 1.0, got {total:.3f}")

        ra_w = self.ra_confidence_weights
        ra_total = ra_w["w_sales"] + ra_w["w_ros"] + ra_w["w_str"]
        if abs(ra_total - 1.0) > 0.01:
            errors.append(f"ra_confidence_weights must sum to 1.0, got {ra_total:.3f}")

        ftf_w = self.ftf_confidence_weights
        ftf_total = ftf_w["w_trend"] + ftf_w["w_confidence"] + ftf_w["w_lifecycle"]
        if abs(ftf_total - 1.0) > 0.01:
            errors.append(f"ftf_confidence_weights must sum to 1.0, got {ftf_total:.3f}")

        if errors:
            raise ValueError("Config validation failed:\n  " + "\n  ".join(errors))

    def detect_ra_level(self, ra_df) -> str:
        """Auto-detect RA input level from the data.

        If the Brand column is populated and unique per row → BRAND level.
        Otherwise → CATEGORY level (Segment × Family).
        """
        import pandas as pd

        brand_col = None
        for col in ra_df.columns:
            if col.lower().strip() in ("brand", "brand_name"):
                brand_col = col
                break

        if brand_col and ra_df[brand_col].notna().all():
            self.primary_key_type = "BRAND"
            self.primary_key_value_field = brand_col
        else:
            self.primary_key_type = "CATEGORY"
            self.primary_key_value_field = None  # use segment × family

        if self.output_level and self.output_level != self.primary_key_type:
            self.level_transformation_needed = True

        return self.primary_key_type

    def freeze(self) -> "FTFConfig":
        """Return self after final validation. Call after detect_ra_level."""
        self.validate()
        return self

    def summary(self) -> dict:
        """Return a dict summary of all config values for logging/audit."""
        return {
            "primary_key_type": self.primary_key_type,
            "level_transformation_needed": self.level_transformation_needed,
            "overlap_threshold": self.overlap_threshold,
            "replacement_threshold": self.replacement_threshold,
            "ftf_added_cap_pct": self.ftf_added_cap_pct,
            "max_adjustment_pct": self.max_adjustment_pct,
            "moq_per_option": self.moq_per_option,
            "shelf_life_days": self.shelf_life_days,
            "str_target": self.str_target,
            "k_neighbors": self.k_neighbors,
            "min_nearest_neighbors": self.min_nearest_neighbors,
            "min_sales_days": self.min_sales_days,
        }


def load_config(overrides: dict = None) -> FTFConfig:
    """Create a config with defaults, applying any overrides.

    Args:
        overrides: dict of parameter names → values to override defaults.
    """
    kwargs = {}
    if overrides:
        valid_fields = {f.name for f in fields(FTFConfig)}
        for key, value in overrides.items():
            if key in valid_fields:
                kwargs[key] = value
    return FTFConfig(**kwargs)
