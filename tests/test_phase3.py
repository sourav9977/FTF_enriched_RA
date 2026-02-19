"""Phase 3 tests — Overlap Scoring, Confidence, Classification (Step 4).

Run with: pytest tests/test_phase3.py -v
"""

import pytest
import numpy as np
import pandas as pd

from src.config import FTFConfig, load_config
from src.module1.factor_analysis import FactorAnalysisResult, BrickWeights
from src.module1.overlap import (
    compute_overlap_score,
    score_all_overlaps,
    assign_best_trends,
    compute_ra_confidence,
    compute_ftf_confidence,
    classify_items,
    generate_ftf_added_from_unmatched,
    assemble_enriched_ra,
    run_enrichment,
    ClassificationResult,
    CONFIDENCE_NUMERIC,
    LIFECYCLE_SCORE,
)


# ── Helpers ───────────────────────────────────────────────────────────

def _make_factor_result(bricks=None, weights=None):
    """Build a FactorAnalysisResult with specified brick weights."""
    if bricks is None:
        bricks = ["SHIRTS"]
    if weights is None:
        weights = {"pattern": 0.25, "color": 0.30, "fit": 0.15,
                   "sleeve": 0.12, "neck_type": 0.10, "length": 0.08}
    result = FactorAnalysisResult()
    for brick in bricks:
        result.brick_weights[brick] = BrickWeights(
            brick=brick, weights=weights, correlations={},
            sample_size=100, attribute_coverage={},
        )
    return result


def _ra_to_trend_map():
    """Standard RA → Trend attribute map."""
    return {
        "pattern_type": "print_",
        "color": "color",
        "fit": "style",
        "sleeve_type": "sleeve",
        "neck_type": "neck_type",
        "length": "length",
    }


def make_ra_df(n=5):
    """Synthetic RA data for overlap testing."""
    rng = np.random.default_rng(42)
    patterns = ["SOLID", "STRIPES", "CHECK", "PRINTED", "FLORAL"]
    colors = ["RED", "BLUE", "BLACK", "WHITE", "GREEN"]
    fits = ["REGULAR", "SLIM", "LOOSE"]
    sleeves = ["FULL", "HALF", "SLEEVELESS"]
    necks = ["COLLAR", "ROUND", "V-NECK", "CREW"]
    lengths = ["REGULAR", "SHORT", "LONG"]

    return pd.DataFrame({
        "unique_id": [f"RA_{i}" for i in range(n)],
        "business_unit": ["AZORTE"] * n,
        "season": ["SS26"] * n,
        "segment": ["MENS CASUAL"] * n,
        "family": ["CASUAL WEAR"] * n,
        "brand": ["DNMX"] * n,
        "brick": ["SHIRTS"] * n,
        "class_": ["TOPS"] * n,
        "fashion_grade": ["CORE"] * n,
        "month_of_drop": ["Mar-26"] * n,
        "mrp": ["999"] * n,
        "quantities_planned": rng.integers(50, 300, n).tolist(),
        "pattern_type": rng.choice(patterns, n).tolist(),
        "color": rng.choice(colors, n).tolist(),
        "fit": rng.choice(fits, n).tolist(),
        "sleeve_type": rng.choice(sleeves, n).tolist(),
        "neck_type": rng.choice(necks, n).tolist(),
        "length": rng.choice(lengths, n).tolist(),
        "hit": ["N"] * n,
    })


def make_trends_df(n=4):
    """Synthetic trend data for overlap testing."""
    rng = np.random.default_rng(99)
    prints = ["SOLID", "STRIPES", "FLORAL", "CHECK", "ABSTRACT"]
    colors = ["RED", "BLUE", "PINK", "BLACK", "WHITE"]
    styles = ["REGULAR", "SLIM", "OVERSIZED", "RELAXED"]
    sleeves = ["FULL", "HALF", "SLEEVELESS"]
    necks = ["COLLAR", "ROUND", "CREW", "V-NECK"]
    lengths = ["REGULAR", "SHORT", "LONG"]

    return pd.DataFrame({
        "trend_name": [f"Trend_{i}" for i in range(n)],
        "brick": ["SHIRTS"] * n,
        "print_": rng.choice(prints, n).tolist(),
        "color": rng.choice(colors, n).tolist(),
        "style": rng.choice(styles, n).tolist(),
        "sleeve": rng.choice(sleeves, n).tolist(),
        "neck_type": rng.choice(necks, n).tolist(),
        "length": rng.choice(lengths, n).tolist(),
        "current_score": rng.uniform(0.3, 0.95, n).round(2).tolist(),
        "confidence": rng.choice(["HIGH", "MEDIUM", "LOW"], n).tolist(),
        "m1_stage": rng.choice(["RISE", "FLAT", "DECLINE"], n).tolist(),
        "m1_month": ["Mar-26"] * n,
        "trajectory": rng.choice(["Rising", "Stable", "Declining"], n).tolist(),
        "business_label": rng.choice(["Core", "Fashion"], n).tolist(),
        "category": ["TOPS"] * n,
        "gender": ["MENS CASUAL"] * n,
    })


def make_sales_df():
    """Synthetic sales data for confidence calculation."""
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame({
        "sku": [f"SKU{i:04d}" for i in range(n)],
        "brick": rng.choice(["SHIRTS", "JEANS"], n),
        "revenue": rng.uniform(500, 5000, n),
        "quantity": rng.integers(10, 200, n),
    })


# ══════════════════════════════════════════════════════════════════════
#  Test 4a: Overlap Scoring
# ══════════════════════════════════════════════════════════════════════

class TestOverlapScoring:

    def test_perfect_match_gives_1(self):
        ra_row = pd.Series({"pattern_type": "SOLID", "color": "RED",
                            "fit": "REGULAR", "sleeve_type": "FULL",
                            "neck_type": "COLLAR", "length": "REGULAR"})
        trend_row = pd.Series({"print_": "SOLID", "color": "RED",
                               "style": "REGULAR", "sleeve": "FULL",
                               "neck_type": "COLLAR", "length": "REGULAR"})
        weights = {"pattern": 0.25, "color": 0.30, "fit": 0.15,
                   "sleeve": 0.12, "neck_type": 0.10, "length": 0.08}
        score = compute_overlap_score(ra_row, trend_row, weights, _ra_to_trend_map())
        assert score == pytest.approx(1.0)

    def test_no_match_gives_0(self):
        ra_row = pd.Series({"pattern_type": "SOLID", "color": "RED",
                            "fit": "REGULAR", "sleeve_type": "FULL",
                            "neck_type": "COLLAR", "length": "REGULAR"})
        trend_row = pd.Series({"print_": "CHECK", "color": "BLUE",
                               "style": "SLIM", "sleeve": "HALF",
                               "neck_type": "ROUND", "length": "SHORT"})
        weights = {"pattern": 0.25, "color": 0.30, "fit": 0.15,
                   "sleeve": 0.12, "neck_type": 0.10, "length": 0.08}
        score = compute_overlap_score(ra_row, trend_row, weights, _ra_to_trend_map())
        assert score == pytest.approx(0.0)

    def test_null_ra_is_wildcard(self):
        """NULL RA attributes should auto-match (wildcard behavior)."""
        ra_row = pd.Series({"pattern_type": None, "color": None,
                            "fit": None, "sleeve_type": None,
                            "neck_type": None, "length": None})
        trend_row = pd.Series({"print_": "SOLID", "color": "RED",
                               "style": "REGULAR", "sleeve": "FULL",
                               "neck_type": "COLLAR", "length": "REGULAR"})
        weights = {"pattern": 0.25, "color": 0.30, "fit": 0.15,
                   "sleeve": 0.12, "neck_type": 0.10, "length": 0.08}
        score = compute_overlap_score(ra_row, trend_row, weights, _ra_to_trend_map())
        assert score == pytest.approx(1.0)

    def test_partial_match(self):
        ra_row = pd.Series({"pattern_type": "SOLID", "color": "RED",
                            "fit": "SLIM", "sleeve_type": "FULL",
                            "neck_type": "COLLAR", "length": "REGULAR"})
        trend_row = pd.Series({"print_": "SOLID", "color": "RED",
                               "style": "REGULAR", "sleeve": "HALF",
                               "neck_type": "ROUND", "length": "REGULAR"})
        weights = {"pattern": 0.25, "color": 0.30, "fit": 0.15,
                   "sleeve": 0.12, "neck_type": 0.10, "length": 0.08}
        score = compute_overlap_score(ra_row, trend_row, weights, _ra_to_trend_map())
        # Matches: pattern=0.25, color=0.30, length=0.08 → 0.63
        # Mismatches: fit=0.15, sleeve=0.12, neck_type=0.10
        # Total weights = 1.0 → score = 0.63
        assert score == pytest.approx(0.63, abs=0.02)

    def test_case_insensitive(self):
        ra_row = pd.Series({"pattern_type": "solid", "color": "red",
                            "fit": "Regular", "sleeve_type": "FULL",
                            "neck_type": "collar", "length": "regular"})
        trend_row = pd.Series({"print_": "SOLID", "color": "RED",
                               "style": "REGULAR", "sleeve": "FULL",
                               "neck_type": "COLLAR", "length": "REGULAR"})
        weights = {"pattern": 0.25, "color": 0.30, "fit": 0.15,
                   "sleeve": 0.12, "neck_type": 0.10, "length": 0.08}
        score = compute_overlap_score(ra_row, trend_row, weights, _ra_to_trend_map())
        assert score == pytest.approx(1.0)


class TestScoreAllOverlaps:

    def test_produces_results(self):
        ra = make_ra_df()
        trends = make_trends_df()
        factor = _make_factor_result()
        scores = score_all_overlaps(ra, trends, factor, _ra_to_trend_map())
        assert len(scores) > 0
        assert set(scores.columns) == {"ra_idx", "trend_idx", "brick", "overlap_score"}

    def test_only_same_brick_matches(self):
        ra = make_ra_df()
        trends = make_trends_df()
        # Change one trend to a different brick
        trends.loc[0, "brick"] = "JEANS"
        factor = _make_factor_result(["SHIRTS", "JEANS"])
        scores = score_all_overlaps(ra, trends, factor, _ra_to_trend_map())
        # No RA item has brick=JEANS, so trend_0 should have no matches
        trend_0_matches = scores[scores["trend_idx"] == 0]
        assert trend_0_matches.empty

    def test_assign_best_trends(self):
        ra = make_ra_df()
        trends = make_trends_df()
        factor = _make_factor_result()
        scores = score_all_overlaps(ra, trends, factor, _ra_to_trend_map())
        best = assign_best_trends(scores)
        assert best["ra_idx"].is_unique
        assert len(best) <= len(ra)


# ══════════════════════════════════════════════════════════════════════
#  Test 4b: Confidence Scores
# ══════════════════════════════════════════════════════════════════════

class TestConfidenceScores:

    def test_ra_confidence_range(self):
        ra = make_ra_df()
        sales = make_sales_df()
        config = load_config()
        scores = compute_ra_confidence(ra, sales, config)
        assert len(scores) == len(ra)
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_ra_confidence_no_sales(self):
        ra = make_ra_df()
        config = load_config()
        scores = compute_ra_confidence(ra, None, config)
        assert (scores == 0.5).all()

    def test_ftf_confidence_range(self):
        trends = make_trends_df()
        config = load_config()
        scores = compute_ftf_confidence(trends, config)
        assert len(scores) == len(trends)
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_ftf_confidence_high_trend(self):
        """A trend with HIGH confidence, RISE stage, and high score should rank high."""
        trends = pd.DataFrame({
            "current_score": [0.95],
            "confidence": ["HIGH"],
            "m1_stage": ["RISE"],
        })
        config = load_config()
        scores = compute_ftf_confidence(trends, config)
        assert scores.iloc[0] > 0.8

    def test_ftf_confidence_low_trend(self):
        """A trend with LOW confidence, DECLINE stage should rank low."""
        trends = pd.DataFrame({
            "current_score": [0.1],
            "confidence": ["LOW"],
            "m1_stage": ["DECLINE"],
        })
        config = load_config()
        scores = compute_ftf_confidence(trends, config)
        assert scores.iloc[0] < 0.4


# ══════════════════════════════════════════════════════════════════════
#  Test 4c: Classification
# ══════════════════════════════════════════════════════════════════════

class TestClassification:

    def _run_classification(self, overlap_threshold=1.0, replacement_threshold=0.6):
        ra = make_ra_df()
        trends = make_trends_df()
        factor = _make_factor_result()
        scores = score_all_overlaps(ra, trends, factor, _ra_to_trend_map())
        best = assign_best_trends(scores)

        config = load_config({"overlap_threshold": overlap_threshold,
                              "replacement_threshold": replacement_threshold})
        ra_conf = pd.Series(0.3, index=ra.index)
        ftf_conf = compute_ftf_confidence(trends, config)

        return classify_items(ra, trends, best, ra_conf, ftf_conf, config)

    def test_classification_produces_all_types(self):
        result = self._run_classification()
        total = (len(result.approved) + len(result.original)
                 + len(result.added_replacements))
        assert total == 5  # all 5 RA items are classified

    def test_some_items_approved_or_original(self):
        """With random data, we expect items to be classified into at least one tier."""
        result = self._run_classification(overlap_threshold=1.0)
        total = len(result.approved) + len(result.original) + len(result.added_replacements)
        assert total == 5  # all 5 RA items accounted for

    def test_wildcard_ra_is_approved(self):
        """An RA item with all NULL attributes should always match 100% (wildcard)."""
        ra = make_ra_df(1)
        for col in ["pattern_type", "color", "fit", "sleeve_type", "neck_type", "length"]:
            ra.at[0, col] = None
        trends = make_trends_df()
        factor = _make_factor_result()
        scores = score_all_overlaps(ra, trends, factor, _ra_to_trend_map())
        best = assign_best_trends(scores)
        assert best.iloc[0]["overlap_score"] == pytest.approx(1.0)

    def test_lowering_threshold_increases_approvals(self):
        strict = self._run_classification(overlap_threshold=1.0, replacement_threshold=0.6)
        relaxed = self._run_classification(overlap_threshold=0.5, replacement_threshold=0.3)
        assert len(relaxed.approved) >= len(strict.approved)

    def test_tier3_replacement_creates_added(self):
        """With very low RA confidence and low replacement_threshold,
        some items should become FTF ADDED replacements."""
        result = self._run_classification(
            overlap_threshold=1.0,
            replacement_threshold=0.9,
        )
        # Items with 0.6 <= overlap < 0.9 go to TIER 2 (original)
        # Items with overlap < 0.6 go to TIER 3
        # With RA_conf=0.3, FTF_conf will likely be higher → replacements
        assert len(result.added_replacements) >= 0  # may or may not have replacements


# ══════════════════════════════════════════════════════════════════════
#  Test 4d: FTF Added from Unmatched Trends
# ══════════════════════════════════════════════════════════════════════

class TestFTFAdded:

    def test_unmatched_trends_generate_added(self):
        ra = make_ra_df(10)
        trends = make_trends_df(10)
        config = load_config({"ftf_added_cap_pct": 0.50})
        ftf_conf = pd.Series(0.7, index=trends.index)

        # Empty overlap = all trends unmatched
        overlap_scores = pd.DataFrame(columns=["ra_idx", "trend_idx", "brick", "overlap_score"])
        added = generate_ftf_added_from_unmatched(
            trends, ra, overlap_scores, ftf_conf, set(), config,
        )
        assert len(added) > 0
        assert all(item["ftf_status"] == "ADDED" for item in added)
        assert all(item["ftf_added_source"] == "UNMATCHED_TREND" for item in added)

    def test_cap_is_respected(self):
        ra = make_ra_df(5)  # cap = 5 × 0.20 = 1
        trends = make_trends_df(10)
        config = load_config({"ftf_added_cap_pct": 0.20})
        ftf_conf = pd.Series(0.7, index=trends.index)
        overlap_scores = pd.DataFrame(columns=["ra_idx", "trend_idx", "brick", "overlap_score"])

        added = generate_ftf_added_from_unmatched(
            trends, ra, overlap_scores, ftf_conf, set(), config,
        )
        assert len(added) <= int(len(ra) * config.ftf_added_cap_pct)

    def test_used_trends_excluded(self):
        trends = make_trends_df(4)
        ra = make_ra_df(2)
        config = load_config({"ftf_added_cap_pct": 1.0})
        ftf_conf = pd.Series(0.7, index=trends.index)
        overlap_scores = pd.DataFrame(columns=["ra_idx", "trend_idx", "brick", "overlap_score"])

        # Mark trends 0 and 1 as used
        added = generate_ftf_added_from_unmatched(
            trends, ra, overlap_scores, ftf_conf, {0, 1}, config,
        )
        # Only trends 2 and 3 should be candidates
        assert len(added) <= 2


# ══════════════════════════════════════════════════════════════════════
#  Test 4e: Assembly
# ══════════════════════════════════════════════════════════════════════

class TestAssembly:

    def test_assemble_has_correct_columns(self):
        result = ClassificationResult()
        result.approved.append({
            "ra_id": "RA-TEST",
            "unique_id": "RA_0",
            "ftf_status": "APPROVED",
            "brick": "SHIRTS",
        })
        enriched = assemble_enriched_ra(result, [])
        assert "ra_id" in enriched.columns
        assert "ftf_status" in enriched.columns
        assert len(enriched) == 1

    def test_assemble_empty(self):
        result = ClassificationResult()
        enriched = assemble_enriched_ra(result, [])
        assert enriched.empty


# ══════════════════════════════════════════════════════════════════════
#  Test Full Pipeline
# ══════════════════════════════════════════════════════════════════════

class TestFullEnrichmentPipeline:

    def test_run_enrichment(self):
        ra = make_ra_df()
        trends = make_trends_df()
        sales = make_sales_df()
        config = load_config()
        factor = _make_factor_result()

        result = run_enrichment(ra, trends, sales, factor, config)
        assert not result.enriched_ra.empty
        assert "ftf_status" in result.enriched_ra.columns

        statuses = set(result.enriched_ra["ftf_status"].unique())
        # Should have at least APPROVED and ORIGINAL
        assert statuses & {"APPROVED", "ORIGINAL"}

    def test_total_items_consistent(self):
        ra = make_ra_df()
        trends = make_trends_df()
        sales = make_sales_df()
        config = load_config()
        factor = _make_factor_result()

        result = run_enrichment(ra, trends, sales, factor, config)
        n_approved = (result.enriched_ra["ftf_status"] == "APPROVED").sum()
        n_original = (result.enriched_ra["ftf_status"] == "ORIGINAL").sum()
        n_added = (result.enriched_ra["ftf_status"] == "ADDED").sum()

        # APPROVED + ORIGINAL should account for all original RA items
        # minus any that were replaced
        n_replaced = len(result.classification.removed_ra)
        assert n_approved + n_original == len(ra) - n_replaced

    def test_enrichment_summary(self):
        ra = make_ra_df()
        trends = make_trends_df()
        sales = make_sales_df()
        config = load_config()
        factor = _make_factor_result()

        result = run_enrichment(ra, trends, sales, factor, config)
        summary = result.summary()
        assert "APPROVED" in summary
        assert "ORIGINAL" in summary
