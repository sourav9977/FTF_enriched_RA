"""Phase 2 tests — Factor Analysis (Step 3) with synthetic data.

Run with: pytest tests/test_phase2.py -v
"""

import pytest
import numpy as np
import pandas as pd

from src.module1.factor_analysis import (
    _prepare_sku_data,
    _build_attribute_matrix,
    _compute_correlations,
    _aggregate_to_attribute_weights,
    compute_brick_weights,
    run_factor_analysis,
    run_factor_analysis_sales_only,
    get_default_weights,
    BrickWeights,
    FactorAnalysisResult,
    MATCHABLE_ATTRIBUTES,
)


# ── Synthetic Data Builders ───────────────────────────────────────────

def make_sales_df(n=100):
    """Sales data with SKU, revenue, brick, color."""
    rng = np.random.default_rng(42)
    skus = [f"SKU{i:04d}" for i in range(n)]
    return pd.DataFrame({
        "sku": skus,
        "revenue": rng.uniform(500, 5000, n),
        "quantity": rng.integers(10, 200, n),
        "brick": rng.choice(["SHIRTS", "JEANS", "T-SHIRTS"], n),
        "color": rng.choice(["RED", "BLUE", "BLACK", "WHITE", "GREEN"], n),
    })


def make_catalog_df(n=100):
    """Catalog data with SKU + matchable attributes."""
    rng = np.random.default_rng(42)
    skus = [f"SKU{i:04d}" for i in range(n)]
    return pd.DataFrame({
        "sku_code": skus,
        "brick": rng.choice(["SHIRTS", "JEANS", "T-SHIRTS"], n),
        "pattern": rng.choice(["SOLID", "STRIPES", "CHECK", "PRINTED"], n),
        "color": rng.choice(["RED", "BLUE", "BLACK", "WHITE", "GREEN"], n),
        "fit": rng.choice(["REGULAR", "SLIM", "LOOSE"], n),
        "sleeve": rng.choice(["FULL", "HALF", "SLEEVELESS"], n),
        "neck_type": rng.choice(["ROUND", "V-NECK", "COLLAR"], n),
        "length": rng.choice(["SHORT", "REGULAR", "LONG"], n),
    })


# ── Test SKU Join ─────────────────────────────────────────────────────

class TestPrepareSkuData:

    def test_basic_join(self):
        sales = make_sales_df()
        catalog = make_catalog_df()
        merged = _prepare_sku_data(sales, catalog)
        assert len(merged) > 0
        assert "nsv" in merged.columns
        assert "brick" in merged.columns

    def test_join_produces_sku_level(self):
        sales = make_sales_df()
        catalog = make_catalog_df()
        merged = _prepare_sku_data(sales, catalog)
        assert merged["sku_code"].is_unique

    def test_raises_on_missing_sku(self):
        sales = pd.DataFrame({"not_sku": [1], "revenue": [100]})
        catalog = make_catalog_df()
        with pytest.raises(ValueError, match="missing SKU column"):
            _prepare_sku_data(sales, catalog)

    def test_raises_on_missing_revenue(self):
        sales = pd.DataFrame({"sku": ["X1"], "other": [1]})
        catalog = make_catalog_df()
        with pytest.raises(ValueError, match="missing revenue"):
            _prepare_sku_data(sales, catalog)

    def test_string_normalization(self):
        """SKU keys should be normalized to strings."""
        sales = pd.DataFrame({
            "sku": [1001, 1002],
            "revenue": [500.0, 800.0],
        })
        catalog = pd.DataFrame({
            "sku_code": ["1001", "1002"],
            "brick": ["SHIRTS", "SHIRTS"],
            "color": ["RED", "BLUE"],
        })
        merged = _prepare_sku_data(sales, catalog)
        assert len(merged) == 2


# ── Test Attribute Matrix ─────────────────────────────────────────────

class TestBuildAttributeMatrix:

    def test_binary_encoding(self):
        sales = make_sales_df(50)
        catalog = make_catalog_df(50)
        merged = _prepare_sku_data(sales, catalog)
        X, y = _build_attribute_matrix(merged, "SHIRTS")
        if not X.empty:
            assert all(X.isin([0, 1]).all())
            assert len(y) == len(X)

    def test_skips_low_coverage_attributes(self):
        """Attributes that are >90% null should be excluded."""
        df = pd.DataFrame({
            "brick": ["SHIRTS"] * 20,
            "nsv": range(20),
            "color": ["RED"] * 10 + ["BLUE"] * 10,
            "pattern": [None] * 19 + ["SOLID"],
        })
        X, _ = _build_attribute_matrix(df, "SHIRTS", ["color", "pattern"])
        col_prefixes = {c.split("_")[0] for c in X.columns}
        assert "color" in col_prefixes
        assert "pattern" not in col_prefixes

    def test_returns_empty_for_insufficient_skus(self):
        df = pd.DataFrame({
            "brick": ["SHIRTS"] * 3,
            "nsv": [100, 200, 300],
            "color": ["RED", "BLUE", "GREEN"],
        })
        X, y = _build_attribute_matrix(df, "SHIRTS")
        assert X.empty


# ── Test Correlation Analysis ─────────────────────────────────────────

class TestCorrelations:

    def test_basic_correlation(self):
        X = pd.DataFrame({"a_X": [1, 0, 1, 0], "a_Y": [0, 1, 0, 1]})
        y = pd.Series([10, 5, 12, 3])
        corrs = _compute_correlations(X, y)
        assert "a_X" in corrs
        assert "a_Y" in corrs

    def test_zero_variance_column(self):
        X = pd.DataFrame({"a_X": [1, 1, 1, 1]})
        y = pd.Series([10, 5, 12, 3])
        corrs = _compute_correlations(X, y)
        assert corrs["a_X"] == 0.0


# ── Test Weight Calculation ───────────────────────────────────────────

class TestWeightCalculation:

    def test_weights_sum_to_one(self):
        corrs = {
            "pattern_SOLID": 0.3, "pattern_STRIPES": 0.2,
            "color_RED": 0.5, "color_BLUE": -0.1,
        }
        weights, _ = _aggregate_to_attribute_weights(corrs, ["pattern", "color"])
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_handles_zero_correlations(self):
        corrs = {"pattern_SOLID": 0.0, "color_RED": 0.0}
        weights, _ = _aggregate_to_attribute_weights(corrs, ["pattern", "color"])
        assert abs(weights["pattern"] - 0.5) < 1e-6
        assert abs(weights["color"] - 0.5) < 1e-6


# ── Test Full Pipeline ────────────────────────────────────────────────

class TestFullPipeline:

    def test_run_factor_analysis(self):
        sales = make_sales_df(200)
        catalog = make_catalog_df(200)
        result = run_factor_analysis(sales, catalog, min_skus=5)
        assert isinstance(result, FactorAnalysisResult)
        assert len(result.brick_weights) > 0
        for bw in result.brick_weights.values():
            assert abs(sum(bw.weights.values()) - 1.0) < 1e-6
            assert bw.sample_size > 0

    def test_skipped_bricks_with_high_min(self):
        sales = make_sales_df(20)
        catalog = make_catalog_df(20)
        result = run_factor_analysis(sales, catalog, min_skus=999)
        assert len(result.brick_weights) == 0
        assert len(result.skipped_bricks) > 0

    def test_sales_only_fallback(self):
        sales = make_sales_df(200)
        result = run_factor_analysis_sales_only(sales, min_skus=5)
        assert len(result.brick_weights) > 0
        for bw in result.brick_weights.values():
            assert bw.method.endswith("(sales-only)")

    def test_default_weights(self):
        weights = get_default_weights()
        assert len(weights) == len(MATCHABLE_ATTRIBUTES)
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_brick_weights_audit_fields(self):
        sales = make_sales_df(200)
        catalog = make_catalog_df(200)
        result = run_factor_analysis(sales, catalog, min_skus=5)
        for bw in result.brick_weights.values():
            assert bw.calculation_date is not None
            assert bw.method is not None
            assert isinstance(bw.correlations, dict)
            assert isinstance(bw.attribute_coverage, dict)

    def test_summary_output(self):
        result = FactorAnalysisResult()
        result.brick_weights["SHIRTS"] = BrickWeights(
            brick="SHIRTS",
            weights={"color": 0.5, "pattern": 0.5},
            correlations={"color": 0.3, "pattern": 0.2},
            sample_size=50,
            attribute_coverage={"color": 1.0, "pattern": 0.8},
        )
        s = result.summary()
        assert "SHIRTS" in s
        assert "n=" in s
