"""Phase 4 tests — AOP Breakdown (Step 5) & Quantity Estimation (Steps 6-7).

Run with: pytest tests/test_phase4.py -v
"""

import pytest
import numpy as np
import pandas as pd

from src.config import FTFConfig, load_config
from src.module1.factor_analysis import FactorAnalysisResult, BrickWeights
from src.module2.aop_breakdown import (
    compute_brick_contributions,
    compute_attribute_contributions,
    apply_trend_adjustment,
    allocate_aop_to_bricks,
    validate_constraints,
    run_aop_breakdown,
)
from src.module2.quantity_estimation import (
    compute_performance_scores,
    adjust_approved_quantities,
    apply_shelf_life_cap,
    run_quantity_estimation,
)


# ── Helpers ───────────────────────────────────────────────────────────

def _make_sales(n=200):
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "sku": [f"SKU{i:04d}" for i in range(n)],
        "brick": rng.choice(["SHIRTS", "JEANS", "T-SHIRTS"], n),
        "color": rng.choice(["RED", "BLUE", "BLACK", "WHITE"], n),
        "revenue": rng.uniform(500, 5000, n).round(2),
        "quantity": rng.integers(10, 200, n),
        "day": pd.date_range("2025-01-01", periods=n, freq="h"),
    })


def _make_aop():
    return pd.DataFrame({
        "channel": ["STORE", "STORE", "ONLINE"],
        "fashion_grade": ["CORE", "FASHION", "CORE"],
        "period": ["Mar-26", "Mar-26", "Mar-26"],
        "revenue": [1_000_000, 500_000, 300_000],
    })


def _make_enriched_ra(n=20):
    rng = np.random.default_rng(42)
    statuses = (["APPROVED"] * 8 + ["ORIGINAL"] * 8 + ["ADDED"] * 4)[:n]
    return pd.DataFrame({
        "ra_id": [f"RA-{i:04d}" for i in range(n)],
        "unique_id": [f"UID_{i}" for i in range(n)],
        "brick": rng.choice(["SHIRTS", "JEANS", "T-SHIRTS"], n).tolist(),
        "ftf_status": statuses,
        "original_quantity": rng.integers(50, 300, n).tolist(),
        "trend_score": rng.uniform(0.3, 0.95, n).round(2).tolist(),
        "trend_stage": rng.choice(["RISE", "FLAT", "DECLINE"], n).tolist(),
        "trend_confidence": rng.choice(["HIGH", "MEDIUM", "LOW"], n).tolist(),
        "overlap_score": rng.uniform(0.5, 1.0, n).round(2).tolist(),
        "pattern_type": rng.choice(["SOLID", "STRIPES", "CHECK"], n).tolist(),
        "color": rng.choice(["RED", "BLUE", "BLACK"], n).tolist(),
        "fit": rng.choice(["REGULAR", "SLIM"], n).tolist(),
        "sleeve_type": rng.choice(["FULL", "HALF"], n).tolist(),
        "neck_type": rng.choice(["COLLAR", "ROUND"], n).tolist(),
        "length": rng.choice(["REGULAR", "SHORT"], n).tolist(),
        "segment": ["MENS CASUAL"] * n,
        "family": ["CASUAL WEAR"] * n,
        "brand": ["DNMX"] * n,
        "season": ["SS26"] * n,
    })


def _make_factor_result():
    weights = {"pattern": 0.25, "color": 0.30, "fit": 0.15,
               "sleeve": 0.12, "neck_type": 0.10, "length": 0.08}
    result = FactorAnalysisResult()
    for b in ["SHIRTS", "JEANS", "T-SHIRTS"]:
        result.brick_weights[b] = BrickWeights(
            brick=b, weights=weights, correlations={},
            sample_size=50, attribute_coverage={},
        )
    return result


# ══════════════════════════════════════════════════════════════════════
#  Test Step 5: AOP Breakdown
# ══════════════════════════════════════════════════════════════════════

class TestBrickContributions:

    def test_contributions_sum_to_one(self):
        sales = _make_sales()
        config = load_config()
        contribs = compute_brick_contributions(sales, config)
        assert abs(contribs["contribution"].sum() - 1.0) < 1e-6

    def test_all_bricks_present(self):
        sales = _make_sales()
        config = load_config()
        contribs = compute_brick_contributions(sales, config)
        assert set(contribs["brick"]) == {"SHIRTS", "JEANS", "T-SHIRTS"}

    def test_has_avg_asp(self):
        sales = _make_sales()
        config = load_config()
        contribs = compute_brick_contributions(sales, config)
        assert "avg_asp" in contribs.columns
        assert (contribs["avg_asp"] > 0).all()


class TestAttributeContributions:

    def test_within_brick_sums_to_one(self):
        sales = _make_sales()
        config = load_config()
        attr = compute_attribute_contributions(sales, None, group_by=["color"])
        if not attr.empty:
            for brick, grp in attr.groupby("brick"):
                assert abs(grp["attr_contribution"].sum() - 1.0) < 1e-6


class TestTrendAdjustment:

    def test_rising_trends_get_uplift(self):
        sales = _make_sales()
        config = load_config()
        contribs = compute_brick_contributions(sales, config)
        enriched = _make_enriched_ra()
        # Force all approved items to RISE
        enriched.loc[enriched["ftf_status"] == "APPROVED", "trend_stage"] = "RISE"
        enriched.loc[enriched["ftf_status"] == "APPROVED", "trend_score"] = 0.9

        adjusted = apply_trend_adjustment(contribs, enriched, config)
        assert "adjusted_contribution" in adjusted.columns
        assert abs(adjusted["adjusted_contribution"].sum() - 1.0) < 1e-6

    def test_no_rising_no_uplift(self):
        sales = _make_sales()
        config = load_config()
        contribs = compute_brick_contributions(sales, config)
        enriched = _make_enriched_ra()
        enriched["trend_stage"] = "DECLINE"

        adjusted = apply_trend_adjustment(contribs, enriched, config)
        assert (adjusted["trend_uplift"] == 0).all()


class TestAOPAllocation:

    def test_target_nsv_assigned(self):
        sales = _make_sales()
        config = load_config()
        contribs = compute_brick_contributions(sales, config)
        aop = _make_aop()

        targets, total = allocate_aop_to_bricks(contribs, aop, config)
        assert total == 1_800_000
        assert "target_nsv" in targets.columns
        assert abs(targets["target_nsv"].sum() - total) < 1.0


class TestConstraints:

    def test_moq_applied(self):
        config = load_config({"moq_per_option": 1000})
        targets = pd.DataFrame({
            "brick": ["SHIRTS", "JEANS"],
            "target_qty": [500, 2000],
        })
        validated = validate_constraints(targets, config)
        assert validated.loc[0, "target_qty"] == 1000  # bumped to MOQ
        assert validated.loc[1, "target_qty"] == 2000  # already above


class TestFullAOPPipeline:

    def test_run_aop_breakdown(self):
        sales = _make_sales()
        aop = _make_aop()
        enriched = _make_enriched_ra()
        config = load_config()

        result = run_aop_breakdown(aop, sales, enriched, config)
        assert not result.brick_targets.empty
        assert result.total_aop_target > 0
        assert "target_nsv" in result.brick_targets.columns


# ══════════════════════════════════════════════════════════════════════
#  Test Steps 6-7: Quantity Estimation
# ══════════════════════════════════════════════════════════════════════

class TestPerformanceScores:

    def test_scores_in_range(self):
        enriched = _make_enriched_ra()
        sales = _make_sales()
        config = load_config()
        scores = compute_performance_scores(enriched, sales, config)
        assert len(scores) == len(enriched)
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_no_sales_uses_trend(self):
        enriched = _make_enriched_ra()
        config = load_config()
        scores = compute_performance_scores(enriched, None, config)
        assert (scores >= 0).all() and (scores <= 1).all()


class TestApprovedAdjustment:

    def test_adjustment_bounded(self):
        enriched = _make_enriched_ra()
        config = load_config({"max_adjustment_pct": 0.10})
        perf = pd.Series(np.random.uniform(0, 1, len(enriched)), index=enriched.index)
        adjusted = adjust_approved_quantities(enriched, perf, config)

        approved = adjusted[adjusted["ftf_status"] == "APPROVED"]
        for _, row in approved.iterrows():
            orig = pd.to_numeric(row["original_quantity"], errors="coerce")
            adj = pd.to_numeric(row["adjusted_quantity"], errors="coerce")
            if pd.notna(orig) and orig > 0 and pd.notna(adj):
                pct_change = abs(adj - orig) / orig
                assert pct_change <= 0.10 + 0.01  # small float tolerance

    def test_original_items_unchanged(self):
        enriched = _make_enriched_ra()
        config = load_config()
        perf = pd.Series(0.5, index=enriched.index)
        adjusted = adjust_approved_quantities(enriched, perf, config)
        originals = adjusted[adjusted["ftf_status"] == "ORIGINAL"]
        for _, row in originals.iterrows():
            assert row["adjustment_factor"] == 0.0


class TestShelfLifeCap:

    def test_cap_applied_when_needed(self):
        enriched = pd.DataFrame({
            "ftf_status": ["APPROVED"],
            "brick": ["SHIRTS"],
            "adjusted_quantity": [999999],
            "original_quantity": [999999],
        })
        sales = pd.DataFrame({
            "sku": ["S1"] * 10,
            "brick": ["SHIRTS"] * 10,
            "quantity": [10] * 10,
            "day": pd.date_range("2025-01-01", periods=10),
        })
        config = load_config()
        capped, n_caps = apply_shelf_life_cap(enriched, sales, config)
        assert n_caps == 1
        assert capped.at[0, "cap_applied"] == True
        assert capped.at[0, "adjusted_quantity"] < 999999


class TestFullQuantityPipeline:

    def test_run_quantity_estimation(self):
        enriched = _make_enriched_ra()
        sales = _make_sales()
        config = load_config()
        factor = _make_factor_result()

        result = run_quantity_estimation(enriched, sales, config, factor)
        assert not result.final_ra.empty
        assert "adjusted_quantity" in result.final_ra.columns
        assert result.approved_adjustments > 0

    def test_all_items_have_quantity(self):
        enriched = _make_enriched_ra()
        sales = _make_sales()
        config = load_config()
        factor = _make_factor_result()

        result = run_quantity_estimation(enriched, sales, config, factor)
        non_null = result.final_ra["adjusted_quantity"].notna().sum()
        assert non_null == len(enriched)

    def test_summary(self):
        enriched = _make_enriched_ra()
        sales = _make_sales()
        config = load_config()

        result = run_quantity_estimation(enriched, sales, config)
        s = result.summary()
        assert "APPROVED" in s
        assert "ADDED" in s
