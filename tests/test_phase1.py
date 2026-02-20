"""Phase 1 tests — Config, Ingestion, Normalization with synthetic data.

Run with: pytest tests/test_phase1.py -v
"""

import pytest
import pandas as pd

from src.config import FTFConfig, load_config
from src.module1.ingest import (
    get_mapping, reload_mapping, map_columns, validate_schema,
    detect_ra_level, ingest_ra, ingest_trends, DataMapping,
)
from src.module1.normalize import (
    normalize_value, normalize_color, normalize_brick,
    normalize_fashion_grade, normalize_ra, normalize_trends,
    run_normalization,
)


# ── Synthetic Data Builders ───────────────────────────────────────────

def make_ra_df(n=5):
    """Build a small synthetic RA DataFrame matching the YAML schema.

    All required fields are populated with realistic non-null values.
    """
    return pd.DataFrame({
        "Unique Id": [f"RA_{i}" for i in range(n)],
        "Business Unit": ["AZORTE"] * n,
        "Sales Channel": ["AZORTE STORE"] * n,
        "Season": ["AWSS-26"] * n,
        "Segment": ["MENS CASUAL"] * n,
        "Family": ["CASUAL WEAR"] * n,
        "Brand": ["ALTHEORY BY AZORTE"] * n,
        "Brick": ["SHIRTS", "SHIRTS", "JEANS", "TROUSERS", "T-SHIRTS"][:n],
        "Class": ["TOPS", "TOPS", "BOTTOMS", "BOTTOMS", "TOPS"][:n],
        "Month of Drop": ["March"] * n,
        "Hit": ["H2", "H3", "H2", "H1", "H3"][:n],
        "Fashion Type": ["APL-FASHION"] * n,
        "Product Group": ["WOVEN", "WOVEN", "DENIM", "WOVEN", "KNIT"][:n],
        "MRP": ["1700-1799", "1800-1899", "1900-1999", "2200-2299", "999"][:n],
        "Fit": ["Slim", "Regular", "Straight", "Straight", "Oversized"][:n],
        "Neck Type": ["Shirt Collar", "spread collar", "Band Collar", "Band Collar", "round neck"][:n],
        "Waist": ["30", "32", "32", "34", "28"][:n],
        "Pattern Type": ["solid", "stripes", "solid", "checks", "color block"][:n],
        "Sleeve Type": ["full sleeves", "half sleeves", "full sleeves", "full sleeves", "short sleeve"][:n],
        "Wash Type": ["Normal", "Normal", "Dark wash", "Normal", "Normal"][:n],
        "Color": ["navy blue", "White", "dark blue", "Khaki", "Black"][:n],
        "Length": ["Regular", "Regular", "Ankle", "Regular", "Crop"][:n],
        "Emblishment": ["Plain", "Plain", "Plain", "Plain", "Plain"][:n],
        "Quantities Planned": [500, 450, 600, 550, 400][:n],
    })


def make_trend_df(n=4):
    """Build a small synthetic Trend Report DataFrame matching the YAML schema.

    Includes all M1-M4 forecast columns as required by the schema.
    """
    return pd.DataFrame({
        "Trend_ID": [f"TID_{i:04x}" for i in range(n)],
        "Gender": ["men", "men", "women", "men"][:n],
        "Category": ["topwear", "topwear", "topwear", "bottomwear"][:n],
        "Brick": ["SHIRTS", "T-SHIRTS", "TOPS", "JEANS"][:n],
        "Trend Name": [
            "men topwear | Solid Navy SHIRTS | Regular FULL SLEEVES | COLLAR",
            "men topwear | Color Block Black T-SHIRTS | Crop SLEEVELESS | ROUND NECK",
            "women topwear | Strappy COLOR BLOCK Black TOPS | Crop SLEEVELESS | HALTER",
            "men bottomwear | Slim SOLID Dark Blue JEANS | Ankle",
        ][:n],
        "Print": ["solid", "color block", "color block", "solid"][:n],
        "Pattern": ["solid", "color block", "color block", "solid"][:n],
        "Style": ["Regular", "Strappy", "Strappy", "Slim"][:n],
        "Neck Type": ["collar", "round", "halter", "round"][:n],
        "Length": ["Regular", "Crop", "Crop", "Ankle"][:n],
        "Sleeve": ["full sleeves", "sleeveless", "sleeveless", "full sleeves"][:n],
        "Color": ["navy", "Black", "Black", "dark blue"][:n],
        "Current Score": [0.82, 0.78, 0.85, 0.71][:n],
        # M1 forecast
        "Month": ["2026-03", "2026-03", "2026-02", "2026-03"][:n],
        "Score": [0.80, 0.75, 0.84, 0.69][:n],
        "Stage": ["Flat", "Rise", "Flat", "Rise"][:n],
        "TopK Prob": [0.10, 0.20, 0.15, 0.05][:n],
        # M2 forecast
        "Month_2": ["2026-04", "2026-04", "2026-03", "2026-04"][:n],
        "Score_2": [0.78, 0.77, 0.83, 0.70][:n],
        "Stage_2": ["Flat", "Rise", "Flat", "Flat"][:n],
        "TopK Prob_2": [0.10, 0.18, 0.14, 0.06][:n],
        # M3 forecast
        "Month_3": ["2026-05", "2026-05", "2026-04", "2026-05"][:n],
        "Score_3": [0.76, 0.79, 0.81, 0.68][:n],
        "Stage_3": ["Decline", "Rise", "Flat", "Flat"][:n],
        "TopK Prob_3": [0.08, 0.19, 0.12, 0.04][:n],
        # M4 forecast
        "Month_4": ["2026-06", "2026-06", "2026-05", "2026-06"][:n],
        "Score_4": [0.72, 0.80, 0.79, 0.65][:n],
        "Stage_4": ["Decline", "Flat", "Decline", "Decline"][:n],
        "TopK Prob_4": [0.06, 0.17, 0.10, 0.03][:n],
        # Business insights
        "Business Label": ["Stable", "Rising Star", "Stable Leaders", "Emerging"][:n],
        "Confidence": ["High", "Medium", "High", "Low"][:n],
        "Trajectory": ["Flat->Flat->Decline->Decline", "Rise->Rise->Rise->Flat",
                        "Flat->Flat->Flat->Decline", "Rise->Flat->Flat->Decline"][:n],
    })


def make_sales_df(n=10):
    """Build a small synthetic Sales DataFrame."""
    return pd.DataFrame({
        "channel": ["AZORTE"] * n,
        "store_code": ["T5UG", "T4QE"] * (n // 2),
        "sku": [f"SKU_{i}" for i in range(n)],
        "style_code": [f"STYLE_{i}" for i in range(n)],
        "day": pd.date_range("2026-01-01", periods=n),
        "quantity": [1, 2, 1, 3, 1, 2, 1, 1, 2, 1][:n],
        "revenue": [650, 1200, 850, 2100, 500, 1100, 750, 650, 900, 550][:n],
    })


def make_aop_df():
    """Build a small synthetic AOP DataFrame."""
    return pd.DataFrame({
        "store_code": ["T4QE", "T5UG"],
        "channel": ["AZORTE", "AZORTE"],
        "fashion_grade": ["APL-FASHION", "APL-FASHION"],
        "brand_segment": ["MENS WEAR", "MENS WEAR"],
        "family": ["FORMAL WEAR", "FORMAL WEAR"],
        "brand": ["ALTHEORY", "ALTHEORY"],
        "period": ["march", "march"],
        "year": [2026, 2026],
        "revenue": [13000000, 18000000],
    })


# ── Helper to get schemas from YAML ──────────────────────────────────

@pytest.fixture(autouse=True)
def fresh_mapping():
    """Ensure a fresh mapping is loaded for each test."""
    reload_mapping()
    yield


def _ra_schema():
    return get_mapping().ra_schema

def _trend_schema():
    return get_mapping().trend_schema

def _sales_schema():
    return get_mapping().sales_schema

def _aop_schema():
    return get_mapping().aop_schema


# ══════════════════════════════════════════════════════════════════════
#  TEST SUITE: YAML MAPPING
# ══════════════════════════════════════════════════════════════════════

class TestYAMLMapping:

    def test_mapping_loads(self):
        m = get_mapping()
        assert isinstance(m, DataMapping)

    def test_all_schemas_present(self):
        m = get_mapping()
        assert len(m.ra_schema) > 0
        assert len(m.trend_schema) > 0
        assert len(m.sales_schema) > 0
        assert len(m.catalog_schema) > 0
        assert len(m.aop_schema) > 0
        assert len(m.site_master_schema) > 0

    def test_ra_required_fields(self):
        m = get_mapping()
        required = {r.internal_name for r in m.ra_schema if r.required}
        # Identity & hierarchy — always required
        assert "unique_id" in required
        assert "brick" in required
        assert "quantities_planned" in required
        assert "hit" in required
        assert "business_unit" in required
        assert "season" in required
        assert "fashion_grade" in required
        assert "mrp" not in required  # MRP optional — null rows still accepted
        # Matchable attributes — optional (wildcard when null)
        assert "fit" not in required
        assert "neck_type" not in required
        assert "pattern_type" not in required
        assert "sleeve_type" not in required
        assert "color" not in required
        assert "length" not in required
        assert "waist" not in required
        assert "wash_type" not in required
        assert "embellishment" not in required
        # Other optional
        assert "sales_channel" not in required
        assert "product_group" not in required
        assert "fixture_type" not in required

    def test_trend_all_fields_required(self):
        m = get_mapping()
        required = {r.internal_name for r in m.trend_schema if r.required}
        optional = {r.internal_name for r in m.trend_schema if not r.required}
        assert "gender" in required
        assert "category" in required
        assert "brick" in required
        assert "trend_name" in required
        assert "trend_id" in required
        assert "print_" in required
        assert "pattern" in required
        assert "style" in required
        assert "color" in required
        assert "current_score" in required
        assert "business_label" in required
        assert "confidence" in required
        # neck_type and sleeve are optional (null for bottomwear)
        assert "neck_type" in optional
        assert "sleeve" in optional
        assert "trajectory" in required

    def test_cross_mapping_ra_to_trend(self):
        m = get_mapping()
        rt = m.ra_to_trend_map
        assert rt["segment"] == "gender"
        assert rt["class_"] == "category"
        assert rt["brick"] == "brick"
        assert rt["pattern_type"] == "print_"
        assert rt["fit"] == "style"
        assert rt["neck_type"] == "neck_type"
        assert rt["sleeve_type"] == "sleeve"
        assert rt["color"] == "color"
        assert rt["length"] == "length"
        assert rt["month_of_drop"] == "m1_month"

    def test_cross_mapping_trend_to_ra(self):
        m = get_mapping()
        tr = m.trend_to_ra_map
        assert tr["gender"] == "segment"
        assert tr["category"] == "class_"
        assert tr["brick"] == "brick"
        assert tr["print_"] == "pattern_type"
        assert tr["style"] == "fit"
        assert tr["sleeve"] == "sleeve_type"
        assert tr["color"] == "color"
        assert tr["length"] == "length"
        assert tr["m1_month"] == "month_of_drop"

    def test_trend_dedup_map(self):
        m = get_mapping()
        assert m.trend_dedup_map["month"] == "m1_month"
        assert m.trend_dedup_map["score_2"] == "m2_score"
        assert m.trend_dedup_map["stage_3"] == "m3_stage"
        assert m.trend_dedup_map["topk prob_4"] == "m4_topk_prob"

    def test_trend_header_keywords(self):
        m = get_mapping()
        assert "gender" in m.trend_header_keywords
        assert "trend name" in m.trend_header_keywords
        assert "colour" in m.trend_header_keywords


# ══════════════════════════════════════════════════════════════════════
#  TEST SUITE: CONFIGURATION (Step 0)
# ══════════════════════════════════════════════════════════════════════

class TestConfig:

    def test_default_config_loads(self):
        cfg = load_config()
        assert cfg.overlap_threshold == 1.0
        assert cfg.replacement_threshold == 0.60
        assert cfg.str_target == 0.75
        assert cfg.moq_per_option == 2500

    def test_config_with_overrides(self):
        cfg = load_config({"overlap_threshold": 0.80, "moq_per_option": 5000})
        assert cfg.overlap_threshold == 0.80
        assert cfg.moq_per_option == 5000

    def test_config_validation_passes(self):
        cfg = load_config()
        cfg.validate()

    def test_replacement_must_be_leq_overlap(self):
        with pytest.raises(ValueError, match="replacement_threshold"):
            load_config({"replacement_threshold": 0.90, "overlap_threshold": 0.80})

    def test_str_target_range(self):
        with pytest.raises(ValueError, match="str_target"):
            load_config({"str_target": 0.30})

    def test_weight_sums_validated(self):
        with pytest.raises(ValueError, match="performance_weights"):
            load_config({"performance_weights": {
                "ros": 0.50, "str": 0.50, "turns": 0.50,
                "contribution": 0.0, "trend_score": 0.0
            }})

    def test_detect_ra_level_brand(self):
        cfg = load_config()
        ra = make_ra_df()
        ra = map_columns(ra, _ra_schema())
        level = cfg.detect_ra_level(ra)
        assert level == "BRAND"

    def test_detect_ra_level_category(self):
        cfg = load_config()
        ra = make_ra_df()
        ra = map_columns(ra, _ra_schema())
        ra["brand"] = None
        level = cfg.detect_ra_level(ra)
        assert level == "CATEGORY"

    def test_config_summary(self):
        cfg = load_config()
        s = cfg.summary()
        assert "overlap_threshold" in s
        assert s["moq_per_option"] == 2500

    def test_config_freeze(self):
        cfg = load_config()
        frozen = cfg.freeze()
        assert frozen is cfg


# ══════════════════════════════════════════════════════════════════════
#  TEST SUITE: INGESTION (Step 1)
# ══════════════════════════════════════════════════════════════════════

class TestIngestion:

    def test_ra_column_mapping(self):
        ra = make_ra_df()
        mapped = map_columns(ra, _ra_schema())
        assert "unique_id" in mapped.columns
        assert "brick" in mapped.columns
        assert "quantities_planned" in mapped.columns
        assert "fashion_grade" in mapped.columns
        assert "hit" in mapped.columns

    def test_ra_validation_passes(self):
        ra = make_ra_df()
        mapped = map_columns(ra, _ra_schema())
        df, result = validate_schema(mapped, _ra_schema(), "RA")
        assert result.is_valid
        assert result.accepted_rows == 5
        assert result.rejected_rows == 0

    def test_ra_rejects_missing_mandatory(self):
        ra = make_ra_df()
        ra.loc[0, "Brick"] = None
        ra.loc[1, "Quantities Planned"] = None
        mapped = map_columns(ra, _ra_schema())
        df, result = validate_schema(mapped, _ra_schema(), "RA")
        assert result.rejected_rows >= 2
        assert result.accepted_rows <= 3

    def test_trend_column_mapping(self):
        trends = make_trend_df()
        mapped = map_columns(trends, _trend_schema())
        assert "gender" in mapped.columns
        assert "brick" in mapped.columns
        assert "current_score" in mapped.columns
        assert "print_" in mapped.columns
        assert "confidence" in mapped.columns

    def test_trend_validation_passes(self):
        from src.module1.ingest import _map_trend_dedup_columns
        trends = make_trend_df()
        trends = _map_trend_dedup_columns(trends)
        mapped = map_columns(trends, _trend_schema())
        df, result = validate_schema(mapped, _trend_schema(), "Trends")
        assert result.is_valid
        assert result.accepted_rows == 4

    def test_detect_ra_level(self):
        ra = make_ra_df()
        mapped = map_columns(ra, _ra_schema())
        assert detect_ra_level(mapped) == "BRAND"

    def test_sales_validation(self):
        sales = make_sales_df()
        mapped = map_columns(sales, _sales_schema())
        df, result = validate_schema(mapped, _sales_schema(), "Sales")
        assert result.is_valid
        assert result.accepted_rows == 10

    def test_aop_validation(self):
        aop = make_aop_df()
        mapped = map_columns(aop, _aop_schema())
        df, result = validate_schema(mapped, _aop_schema(), "AOP")
        assert result.is_valid
        assert result.accepted_rows == 2

    def test_ingest_ra_from_csv(self, tmp_path):
        ra = make_ra_df()
        csv_path = tmp_path / "test_ra.csv"
        ra.to_csv(csv_path, index=False)
        df, result = ingest_ra(str(csv_path))
        assert result.is_valid
        assert result.accepted_rows == 5

    def test_ingest_trends_from_csv(self, tmp_path):
        trends = make_trend_df()
        csv_path = tmp_path / "test_trends.csv"
        trends.to_csv(csv_path, index=False)
        df, result = ingest_trends(str(csv_path))
        assert result.is_valid
        assert result.accepted_rows == 4

    def test_validation_result_summary(self):
        ra = make_ra_df()
        mapped = map_columns(ra, _ra_schema())
        _, result = validate_schema(mapped, _ra_schema(), "RA")
        summary = result.summary()
        assert "RA" in summary
        assert "Total rows" in summary


# ══════════════════════════════════════════════════════════════════════
#  TEST SUITE: NORMALIZATION (Step 2)
# ══════════════════════════════════════════════════════════════════════

class TestNormalization:

    def test_normalize_pattern_synonyms(self):
        assert normalize_value("solid", "pattern_type") == "SOLID"
        assert normalize_value("Stripes", "pattern_type") == "STRIPES"
        assert normalize_value("color block", "pattern_type") == "COLOR BLOCK"
        assert normalize_value("tie & dye", "pattern_type") == "TIE DYE"

    def test_normalize_sleeve_synonyms(self):
        assert normalize_value("full sleeves", "sleeve_type") == "FULL SLEEVES"
        assert normalize_value("short sleeve", "sleeve_type") == "HALF SLEEVES"
        assert normalize_value("sleeveless", "sleeve_type") == "SLEEVELESS"
        assert normalize_value("balloon sleeve", "sleeve_type") == "BALLOON SLEEVE"

    def test_normalize_neck_synonyms(self):
        assert normalize_value("Shirt Collar", "neck_type") == "SHIRT COLLAR"
        assert normalize_value("spread collar", "neck_type") == "SPREAD COLLAR"
        assert normalize_value("round", "neck_type") == "ROUND NECK"
        assert normalize_value("v-neck", "neck_type") == "V NECK"
        assert normalize_value("halter", "neck_type") == "HALTER"

    def test_normalize_fit_synonyms(self):
        assert normalize_value("slim", "fit") == "SLIM"
        assert normalize_value("Regular", "fit") == "REGULAR"
        assert normalize_value("oversized", "fit") == "OVERSIZED"
        assert normalize_value("Strappy", "style") == "STRAPPY"

    def test_normalize_length_synonyms(self):
        assert normalize_value("crop", "length") == "CROPPED"
        assert normalize_value("Ankle", "length") == "ANKLE LENGTH"
        assert normalize_value("Regular", "length") == "REGULAR"

    def test_normalize_color_exact(self):
        assert normalize_color("Black") == "BLACK"
        assert normalize_color("White") == "WHITE"
        assert normalize_color("Red") == "RED"

    def test_normalize_color_family(self):
        assert normalize_color("navy blue") == "BLUE"
        assert normalize_color("olive") == "GREEN"
        assert normalize_color("Khaki") == "BROWN"
        assert normalize_color("dark purple") == "PURPLE"
        assert normalize_color("cream") == "WHITE"

    def test_normalize_color_null(self):
        assert normalize_color(None) is None
        assert normalize_color("") is None
        assert normalize_color("nan") is None

    def test_normalize_brick(self):
        assert normalize_brick("SHIRTS") == "SHIRTS"
        assert normalize_brick("shirt") == "SHIRTS"
        assert normalize_brick("t-shirt") == "T-SHIRTS"
        assert normalize_brick("jean") == "JEANS"
        assert normalize_brick("pants") == "TROUSERS"

    def test_normalize_fashion_grade(self):
        assert normalize_fashion_grade("APL-FASHION") == "FASHION"
        assert normalize_fashion_grade("APL-CORE") == "CORE"
        assert normalize_fashion_grade("FASHION") == "FASHION"

    def test_null_returns_none(self):
        assert normalize_value(None) is None
        assert normalize_value("nan") is None
        assert normalize_value("") is None

    def test_normalize_ra_dataframe(self):
        ra = make_ra_df()
        ra = map_columns(ra, _ra_schema())
        cfg = load_config()
        norm = normalize_ra(ra, cfg)

        assert norm["brick"].iloc[0] == "SHIRTS"
        assert norm["brick"].iloc[2] == "JEANS"
        assert norm["fashion_grade"].iloc[0] == "FASHION"
        assert norm["pattern_type"].iloc[0] == "SOLID"
        assert norm["color"].iloc[0] == "BLUE"
        assert norm["color"].iloc[3] == "BROWN"
        assert norm["fit"].iloc[0] == "SLIM"
        assert norm["neck_type"].iloc[0] == "SHIRT COLLAR"
        assert norm["sleeve_type"].iloc[0] == "FULL SLEEVES"

    def test_normalize_trends_dataframe(self):
        trends = make_trend_df()
        mapped = map_columns(trends, _trend_schema())
        cfg = load_config()
        norm = normalize_trends(mapped, cfg)

        assert norm["brick"].iloc[0] == "SHIRTS"
        assert norm["color"].iloc[0] == "BLUE"
        assert norm["pattern"].iloc[0] == "SOLID"
        assert norm["neck_type"].iloc[0] == "COLLAR"
        assert norm["sleeve"].iloc[0] == "FULL SLEEVES"
        assert norm["current_score"].iloc[0] == 0.82

    def test_run_normalization_pipeline(self):
        ra = map_columns(make_ra_df(), _ra_schema())
        trends = map_columns(make_trend_df(), _trend_schema())
        cfg = load_config()

        result = run_normalization(ra, trends, cfg)
        assert "ra" in result
        assert "trends" in result
        assert len(result["ra"]) == 5
        assert len(result["trends"]) == 4

    def test_ra_and_trends_share_vocabulary_after_normalization(self):
        """After normalization, RA and Trends should use the same canonical
        values for shared attributes, enabling overlap matching."""
        ra = map_columns(make_ra_df(), _ra_schema())
        trends = map_columns(make_trend_df(), _trend_schema())
        cfg = load_config()
        result = run_normalization(ra, trends, cfg)

        ra_bricks = set(result["ra"]["brick"].dropna())
        trend_bricks = set(result["trends"]["brick"].dropna())
        shared_bricks = ra_bricks & trend_bricks
        assert len(shared_bricks) > 0, "RA and Trends should share at least one brick"

        ra_colors = set(result["ra"]["color"].dropna())
        trend_colors = set(result["trends"]["color"].dropna())
        shared_colors = ra_colors & trend_colors
        assert len(shared_colors) > 0, "RA and Trends should share at least one color"
