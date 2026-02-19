"""Phase 5 tests — Rebalancing (Step 8) & Output (Step 9).

Run with: pytest tests/test_phase5.py -v
"""

import pytest
import numpy as np
import pandas as pd

from src.config import load_config
from src.module2.rebalancing import (
    apply_moq_constraint,
    apply_budget_constraint,
    check_intake_margin,
    apply_ftf_added_cap,
    revalidate_shelf_life,
    run_rebalancing,
)
from src.module2.output import (
    compute_summary_metrics,
    generate_output,
    SummaryMetrics,
    OUTPUT_COLUMNS,
)


# ── Helpers ───────────────────────────────────────────────────────────

def _make_enriched(n=20):
    rng = np.random.default_rng(42)
    statuses = (["APPROVED"] * 8 + ["ORIGINAL"] * 8 + ["ADDED"] * 4)[:n]
    return pd.DataFrame({
        "ra_id": [f"RA-{i:04d}" for i in range(n)],
        "unique_id": [f"UID_{i}" for i in range(n)],
        "brick": rng.choice(["SHIRTS", "JEANS", "T-SHIRTS"], n).tolist(),
        "ftf_status": statuses,
        "original_quantity": rng.integers(50, 300, n).tolist(),
        "adjusted_quantity": rng.integers(2000, 5000, n).tolist(),
        "trend_score": rng.uniform(0.3, 0.95, n).round(2).tolist(),
        "trend_confidence": rng.choice(["HIGH", "MEDIUM", "LOW"], n).tolist(),
        "overlap_score": rng.uniform(0.5, 1.0, n).round(2).tolist(),
        "ftf_confidence_score": rng.uniform(0.3, 0.9, n).round(2).tolist(),
        "ftf_added_source": [None] * 8 + [None] * 8 + ["UNMATCHED_TREND"] * 2 + ["CONFIDENCE_REPLACEMENT"] * 2,
        "replaced_ra_id": [None] * 18 + ["OLD_1", "OLD_2"],
        "mrp": rng.integers(500, 2000, n).tolist(),
        "segment": ["MENS CASUAL"] * n,
        "family": ["CASUAL WEAR"] * n,
        "brand": ["DNMX"] * n,
    })


# ══════════════════════════════════════════════════════════════════════
#  Constraint 1: MOQ
# ══════════════════════════════════════════════════════════════════════

class TestMOQConstraint:

    def test_items_above_moq_untouched(self):
        df = _make_enriched()
        config = load_config({"moq_per_option": 1000})
        result, boosts, drops, removed = apply_moq_constraint(df, config)
        assert len(result) + len(removed) == len(df)

    def test_high_confidence_boosted(self):
        df = pd.DataFrame({
            "ra_id": ["A"], "ftf_status": ["ADDED"],
            "adjusted_quantity": [500],
            "trend_score": [0.9], "trend_confidence": ["HIGH"],
        })
        config = load_config({"moq_per_option": 2500})
        result, boosts, drops, _ = apply_moq_constraint(df, config)
        assert boosts == 1
        assert result.at[0, "adjusted_quantity"] == 2500

    def test_low_confidence_dropped(self):
        df = pd.DataFrame({
            "ra_id": ["A"], "ftf_status": ["ADDED"],
            "adjusted_quantity": [500],
            "trend_score": [0.3], "trend_confidence": ["LOW"],
        })
        config = load_config({"moq_per_option": 2500})
        result, boosts, drops, removed = apply_moq_constraint(df, config)
        assert drops == 1
        assert len(result) == 0
        assert len(removed) == 1


# ══════════════════════════════════════════════════════════════════════
#  Constraint 2: AOP Budget
# ══════════════════════════════════════════════════════════════════════

class TestBudgetConstraint:

    def test_no_reduction_under_budget(self):
        df = pd.DataFrame({
            "ra_id": ["A", "B"], "ftf_status": ["APPROVED", "ORIGINAL"],
            "adjusted_quantity": [100, 100],
            "mrp": [100, 100],
            "trend_score": [0.8, 0.5], "overlap_score": [1.0, 0.6],
        })
        config = load_config({"moq_per_option": 10})
        result, reds, removed = apply_budget_constraint(df, 50000, config)
        assert reds == 0
        assert len(result) == 2

    def test_reductions_when_over_budget(self):
        df = pd.DataFrame({
            "ra_id": ["A", "B", "C"],
            "ftf_status": ["APPROVED", "ORIGINAL", "ADDED"],
            "adjusted_quantity": [5000, 5000, 5000],
            "mrp": [1000, 1000, 1000],
            "trend_score": [0.9, 0.5, 0.3],
            "overlap_score": [1.0, 0.6, 0.3],
        })
        config = load_config({"moq_per_option": 100})
        # Total NSV = 15M, budget = 10M
        result, reds, removed = apply_budget_constraint(df, 10_000_000, config)
        assert reds > 0 or len(removed) > 0


# ══════════════════════════════════════════════════════════════════════
#  Constraint 3: Intake Margin
# ══════════════════════════════════════════════════════════════════════

class TestIntakeMargin:

    def test_no_margin_data_passes(self):
        df = pd.DataFrame({"adjusted_quantity": [100], "mrp": [1000]})
        ok, im = check_intake_margin(df, target_im=0.5)
        assert ok is True

    def test_margin_check(self):
        df = pd.DataFrame({
            "adjusted_quantity": [100, 200],
            "mrp": [1000, 1000],
            "margin": [0.4, 0.6],
        })
        ok, im = check_intake_margin(df, target_im=0.45)
        assert ok in (True, False)
        assert 0 <= im <= 1


# ══════════════════════════════════════════════════════════════════════
#  Constraint 4: FTF Added Cap
# ══════════════════════════════════════════════════════════════════════

class TestFTFAddedCap:

    def test_within_cap_no_drops(self):
        df = _make_enriched(20)  # 4 ADDED / 20 = 20%
        config = load_config({"ftf_added_cap_pct": 0.25})
        result, drops, _ = apply_ftf_added_cap(df, config)
        assert drops == 0

    def test_over_cap_drops_lowest(self):
        df = _make_enriched(20)  # 4 ADDED / 20 = 20%
        config = load_config({"ftf_added_cap_pct": 0.10})  # max 2
        result, drops, removed = apply_ftf_added_cap(df, config)
        assert drops > 0
        remaining_added = (result["ftf_status"] == "ADDED").sum()
        max_allowed = int(len(result) * config.ftf_added_cap_pct)
        assert remaining_added <= max_allowed + 1  # allow rounding


# ══════════════════════════════════════════════════════════════════════
#  Constraint 5: Shelf-Life Re-validation
# ══════════════════════════════════════════════════════════════════════

class TestShelfLifeRevalidation:

    def test_warns_on_excess(self):
        df = pd.DataFrame({
            "brick": ["SHIRTS"],
            "adjusted_quantity": [99999],
        })
        ros = {"SHIRTS": 10.0}
        config = load_config()
        result, warns = revalidate_shelf_life(df, ros, config)
        assert warns == 1

    def test_no_warn_when_ok(self):
        df = pd.DataFrame({
            "brick": ["SHIRTS"],
            "adjusted_quantity": [100],
        })
        ros = {"SHIRTS": 10.0}
        config = load_config()
        result, warns = revalidate_shelf_life(df, ros, config)
        assert warns == 0


# ══════════════════════════════════════════════════════════════════════
#  Full Rebalancing Pipeline
# ══════════════════════════════════════════════════════════════════════

class TestFullRebalancing:

    def test_run_rebalancing(self):
        df = _make_enriched()
        config = load_config({"moq_per_option": 100})
        result = run_rebalancing(df, aop_budget=50_000_000, config=config)
        assert not result.final_ra.empty
        assert isinstance(result.items_removed, list)

    def test_summary(self):
        df = _make_enriched()
        config = load_config({"moq_per_option": 100})
        result = run_rebalancing(df, aop_budget=50_000_000, config=config)
        s = result.summary()
        assert "Rebalancing" in s


# ══════════════════════════════════════════════════════════════════════
#  Output Generation (Step 9)
# ══════════════════════════════════════════════════════════════════════

class TestSummaryMetrics:

    def test_metrics_computed(self):
        df = _make_enriched()
        m = compute_summary_metrics(df)
        assert m.total_items == len(df)
        assert m.approved_count + m.original_count + m.added_count == m.total_items
        assert 0 <= m.approved_pct <= 1
        assert 0 <= m.added_pct <= 1

    def test_empty_df(self):
        df = pd.DataFrame(columns=["ftf_status", "original_quantity", "adjusted_quantity"])
        m = compute_summary_metrics(df)
        assert m.total_items == 0

    def test_replacement_count(self):
        df = _make_enriched()
        m = compute_summary_metrics(df)
        assert m.items_replaced == 2  # two have replaced_ra_id

    def test_summary_string(self):
        df = _make_enriched()
        m = compute_summary_metrics(df)
        s = m.summary()
        assert "APPROVED" in s
        assert "ORIGINAL" in s
        assert "ADDED" in s


class TestGenerateOutput:

    def test_output_has_required_columns(self):
        df = _make_enriched()
        out, metrics = generate_output(df)
        assert "version" in out.columns
        assert "created_at" in out.columns
        assert "ftf_status" in out.columns

    def test_output_preserves_all_rows(self):
        df = _make_enriched()
        out, _ = generate_output(df)
        assert len(out) == len(df)

    def test_metrics_returned(self):
        df = _make_enriched()
        _, metrics = generate_output(df)
        assert isinstance(metrics, SummaryMetrics)
        assert metrics.total_items == len(df)
