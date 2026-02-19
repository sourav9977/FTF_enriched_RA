"""Step 2: Data Normalization.

Standardize attribute values across RA and Trend Report so matching is
possible. Handles synonym mapping, color family mapping, brick taxonomy
alignment, fashion grade normalization, and level transformation.
"""

import logging
from typing import Dict, Optional

import pandas as pd
import numpy as np

from src.config import FTFConfig
from src.pipeline_logger import get_logger

logger = get_logger(__name__)


# ── Color Family Mapping ─────────────────────────────────────────────

COLOR_FAMILY_MAP = {
    # Browns
    "dk. brown": "BROWN", "dark brown": "BROWN", "lt. brown": "BROWN",
    "light brown": "BROWN", "chocolate": "BROWN", "tan": "BROWN",
    "camel": "BROWN", "khaki": "BROWN", "mocha": "BROWN", "coffee": "BROWN",
    # Blacks
    "jet black": "BLACK", "charcoal black": "BLACK",
    # Blues
    "navy": "BLUE", "navy blue": "BLUE", "royal blue": "BLUE",
    "sky blue": "BLUE", "lt. blue": "BLUE", "light blue": "BLUE",
    "dk. blue": "BLUE", "dark blue": "BLUE", "cobalt": "BLUE",
    "indigo": "BLUE", "teal": "BLUE", "denim blue": "BLUE",
    # Greens
    "olive": "GREEN", "olive green": "GREEN", "lt olive": "GREEN",
    "lt. olive": "GREEN", "army green": "GREEN", "forest green": "GREEN",
    "sage": "GREEN", "mint": "GREEN", "emerald": "GREEN",
    # Reds
    "maroon": "RED", "burgundy": "RED", "wine": "RED", "crimson": "RED",
    "rust": "RED", "coral": "RED",
    # Whites
    "off white": "WHITE", "off-white": "WHITE", "cream": "WHITE",
    "ivory": "WHITE", "ecru": "WHITE",
    # Greys
    "grey": "GREY", "gray": "GREY", "charcoal": "GREY", "silver": "GREY",
    "lt. grey": "GREY", "dk. grey": "GREY", "ash": "GREY",
    # Pinks
    "rose": "PINK", "blush": "PINK", "dusty pink": "PINK",
    "hot pink": "PINK", "fuchsia": "PINK", "magenta": "PINK",
    # Yellows
    "mustard": "YELLOW", "gold": "YELLOW", "lemon": "YELLOW",
    # Purples
    "lavender": "PURPLE", "violet": "PURPLE", "plum": "PURPLE",
    "mauve": "PURPLE", "lilac": "PURPLE", "dark purple": "PURPLE",
    # Oranges
    "peach": "ORANGE", "tangerine": "ORANGE", "apricot": "ORANGE",
    # Beiges
    "beige": "BEIGE", "sand": "BEIGE", "nude": "BEIGE", "taupe": "BEIGE",
}


# ── Synonym Dictionaries ─────────────────────────────────────────────

PATTERN_SYNONYMS = {
    "solid": "SOLID", "plain": "SOLID",
    "stripes": "STRIPES", "stripe": "STRIPES", "striped": "STRIPES",
    "checks": "CHECKS", "check": "CHECKS", "checked": "CHECKS",
    "checkered": "CHECKS", "plaid": "CHECKS",
    "color block": "COLOR BLOCK", "colour block": "COLOR BLOCK",
    "colorblock": "COLOR BLOCK",
    "tie & dye": "TIE DYE", "tie and dye": "TIE DYE",
    "tie-dye": "TIE DYE", "tie dye": "TIE DYE",
    "floral": "FLORAL", "flower": "FLORAL",
    "geometric": "GEOMETRIC", "geo": "GEOMETRIC",
    "abstract": "ABSTRACT",
    "polka dot": "POLKA DOT", "polka dots": "POLKA DOT", "dotted": "POLKA DOT",
    "self design": "SELF DESIGN", "self-design": "SELF DESIGN",
    "printed": "PRINTED", "print": "PRINTED",
    "dyed": "DYED",
    "ombre": "OMBRE",
    "camo": "CAMOUFLAGE", "camouflage": "CAMOUFLAGE",
    "paisley": "PAISLEY",
    "tropical": "TROPICAL",
    "animal print": "ANIMAL PRINT",
}

SLEEVE_SYNONYMS = {
    "full sleeves": "FULL SLEEVES", "long sleeves": "FULL SLEEVES",
    "long sleeve": "FULL SLEEVES", "full sleeve": "FULL SLEEVES",
    "half sleeves": "HALF SLEEVES", "short sleeves": "HALF SLEEVES",
    "short sleeve": "HALF SLEEVES", "half sleeve": "HALF SLEEVES",
    "sleeveless": "SLEEVELESS", "no sleeve": "SLEEVELESS",
    "3/4 sleeves": "THREE QUARTER SLEEVES",
    "three quarter sleeves": "THREE QUARTER SLEEVES",
    "3/4 sleeve": "THREE QUARTER SLEEVES",
    "balloon sleeve": "BALLOON SLEEVE", "balloon sleeves": "BALLOON SLEEVE",
    "puff sleeve": "PUFF SLEEVE", "puff sleeves": "PUFF SLEEVE",
    "cap sleeve": "CAP SLEEVE", "cap sleeves": "CAP SLEEVE",
    "roll up sleeves": "ROLL UP SLEEVES", "roll-up sleeves": "ROLL UP SLEEVES",
}

NECK_SYNONYMS = {
    "round": "ROUND NECK", "round neck": "ROUND NECK",
    "v-neck": "V NECK", "v neck": "V NECK", "vneck": "V NECK",
    "collar": "COLLAR", "shirt collar": "SHIRT COLLAR",
    "spread collar": "SPREAD COLLAR",
    "mandarin": "MANDARIN COLLAR", "mandarin collar": "MANDARIN COLLAR",
    "band collar": "BAND COLLAR",
    "henley": "HENLEY", "henley neck": "HENLEY",
    "crew": "CREW NECK", "crew neck": "CREW NECK",
    "polo": "POLO COLLAR", "polo collar": "POLO COLLAR",
    "halter": "HALTER", "halter neck": "HALTER",
    "off-shoulder": "OFF SHOULDER", "off shoulder": "OFF SHOULDER",
    "boat neck": "BOAT NECK",
    "square neck": "SQUARE NECK",
    "cowl neck": "COWL NECK",
    "turtle neck": "TURTLE NECK", "turtleneck": "TURTLE NECK",
    "mock neck": "MOCK NECK",
    "hooded": "HOODED", "hood": "HOODED",
}

LENGTH_SYNONYMS = {
    "regular": "REGULAR", "standard": "REGULAR", "normal": "REGULAR",
    "crop": "CROPPED", "cropped": "CROPPED",
    "long": "LONG",
    "short": "SHORT",
    "midi": "MIDI",
    "maxi": "MAXI",
    "mini": "MINI",
    "ankle": "ANKLE LENGTH", "ankle length": "ANKLE LENGTH",
    "knee": "KNEE LENGTH", "knee length": "KNEE LENGTH",
}

FIT_SYNONYMS = {
    "slim": "SLIM", "slim fit": "SLIM",
    "regular": "REGULAR", "regular fit": "REGULAR",
    "relaxed": "RELAXED", "relaxed fit": "RELAXED", "loose": "RELAXED",
    "skinny": "SKINNY", "skinny fit": "SKINNY",
    "oversized": "OVERSIZED", "oversize": "OVERSIZED",
    "tailored": "TAILORED", "tailored fit": "TAILORED",
    "straight": "STRAIGHT", "straight fit": "STRAIGHT",
    "button down": "BUTTON DOWN",
    "strappy": "STRAPPY",
    "blouson": "BLOUSON",
    "wrap": "WRAP",
    "a-line": "A LINE", "a line": "A LINE",
    "fit & flare": "FIT AND FLARE", "fit and flare": "FIT AND FLARE",
}

ATTRIBUTE_SYNONYM_MAP = {
    "pattern_type": PATTERN_SYNONYMS,
    "pattern": PATTERN_SYNONYMS,
    "print_": PATTERN_SYNONYMS,
    "sleeve_type": SLEEVE_SYNONYMS,
    "sleeve": SLEEVE_SYNONYMS,
    "neck_type": NECK_SYNONYMS,
    "color": COLOR_FAMILY_MAP,
    "length": LENGTH_SYNONYMS,
    "fit": FIT_SYNONYMS,
    "style": FIT_SYNONYMS,
}

# Brick taxonomy alignment — map variant brick names to canonical forms
BRICK_SYNONYMS = {
    "t-shirt": "T-SHIRTS", "tshirt": "T-SHIRTS", "t shirt": "T-SHIRTS",
    "t-shirts": "T-SHIRTS", "tshirts": "T-SHIRTS",
    "shirt": "SHIRTS", "shirts": "SHIRTS",
    "jean": "JEANS", "jeans": "JEANS", "denim": "JEANS",
    "trouser": "TROUSERS", "trousers": "TROUSERS", "pant": "TROUSERS",
    "pants": "TROUSERS", "chino": "TROUSERS", "chinos": "TROUSERS",
    "shorts": "SHORTS", "short": "SHORTS",
    "jacket": "JACKETS", "jackets": "JACKETS",
    "sweater": "SWEATERS", "sweaters": "SWEATERS",
    "sweatshirt": "SWEATSHIRTS", "sweatshirts": "SWEATSHIRTS",
    "hoodie": "HOODIES", "hoodies": "HOODIES",
    "top": "TOPS", "tops": "TOPS",
    "dress": "DRESSES", "dresses": "DRESSES",
    "skirt": "SKIRTS", "skirts": "SKIRTS",
    "kurta": "KURTAS", "kurtas": "KURTAS",
    "polo": "POLOS", "polos": "POLOS",
    "blazer": "BLAZERS", "blazers": "BLAZERS",
    "coat": "COATS", "coats": "COATS",
}


# ── Core Normalization Functions ──────────────────────────────────────

def normalize_value(value, attribute_type: str = None) -> Optional[str]:
    """Normalize a single attribute value using synonym dictionaries.

    Returns None for null/empty/NaN values (wildcard in matching).
    """
    if value is None:
        return None
    val = str(value).strip()
    if not val or val.lower() in ("nan", "none", "null", ""):
        return None

    if attribute_type and attribute_type in ATTRIBUTE_SYNONYM_MAP:
        synonyms = ATTRIBUTE_SYNONYM_MAP[attribute_type]
        lookup = val.lower()
        if lookup in synonyms:
            return synonyms[lookup]

    return val.upper()


def normalize_color(value) -> Optional[str]:
    """Normalize color to a canonical family name.

    Exact-match colors pass through uppercased. Variant colors are
    mapped to their family (e.g., "navy blue" → "BLUE").
    """
    if value is None:
        return None
    val = str(value).strip()
    if not val or val.lower() in ("nan", "none", "null"):
        return None
    lookup = val.lower()
    if lookup in COLOR_FAMILY_MAP:
        return COLOR_FAMILY_MAP[lookup]
    return val.upper()


def normalize_brick(value) -> Optional[str]:
    """Normalize a brick name to canonical form."""
    if value is None:
        return None
    val = str(value).strip()
    if not val or val.lower() in ("nan", "none", "null"):
        return None
    lookup = val.lower()
    if lookup in BRICK_SYNONYMS:
        return BRICK_SYNONYMS[lookup]
    return val.upper()


def normalize_fashion_grade(value, grade_map: Dict[str, str] = None) -> Optional[str]:
    """Normalize fashion grade (e.g., APL-FASHION → FASHION)."""
    if value is None:
        return None
    val = str(value).strip()
    if not val or val.lower() in ("nan", "none", "null"):
        return None
    if grade_map is None:
        from src.config import FASHION_GRADE_MAP
        grade_map = FASHION_GRADE_MAP
    return grade_map.get(val, val.upper())


# ── DataFrame-Level Normalization ─────────────────────────────────────

def normalize_ra(ra_df: pd.DataFrame, config: FTFConfig) -> pd.DataFrame:
    """Normalize all attributes in the RA DataFrame.

    - Maps synonyms/variants to canonical values
    - Normalizes fashion grade
    - Normalizes brick taxonomy
    - Uppercases string fields for consistency
    """
    df = ra_df.copy()

    if "brick" in df.columns:
        df["brick"] = df["brick"].apply(normalize_brick)

    if "fashion_grade" in df.columns:
        df["fashion_grade"] = df["fashion_grade"].apply(
            lambda x: normalize_fashion_grade(x, config.fashion_grade_map)
        )

    attr_columns = {
        "pattern_type": "pattern_type",
        "fit": "fit",
        "sleeve_type": "sleeve_type",
        "neck_type": "neck_type",
        "length": "length",
        "color": "color",
    }
    for col, attr_type in attr_columns.items():
        if col in df.columns:
            if col == "color":
                df[col] = df[col].apply(normalize_color)
            else:
                df[col] = df[col].apply(lambda x: normalize_value(x, attr_type))

    # Uppercase key string columns
    for col in ["business_unit", "sales_channel", "season", "segment",
                "family", "brand", "product_group"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: str(x).strip().upper() if pd.notna(x) and str(x).strip() else None
            )

    if "mrp" in df.columns:
        df["mrp"] = pd.to_numeric(df["mrp"], errors="coerce")

    if "quantities_planned" in df.columns:
        df["quantities_planned"] = (
            pd.to_numeric(df["quantities_planned"], errors="coerce")
            .fillna(0)
            .astype(int)
        )

    logger.info(f"Normalized RA: {len(df)} rows, {df['brick'].nunique() if 'brick' in df.columns else '?'} bricks")
    for col, attr_type in attr_columns.items():
        if col in df.columns:
            non_null = df[col].notna().sum()
            n_unique = df[col].nunique()
            logger.debug(f"  RA norm [{col}]: {non_null}/{len(df)} non-null, {n_unique} unique values")
            if n_unique <= 15:
                logger.debug(f"    values: {df[col].value_counts().to_dict()}")
    return df


def normalize_trends(trend_df: pd.DataFrame, config: FTFConfig) -> pd.DataFrame:
    """Normalize all attributes in the Trend Report DataFrame.

    - Maps synonyms/variants to canonical values
    - Normalizes brick taxonomy
    - Coerces scores to float
    """
    df = trend_df.copy()

    if "brick" in df.columns:
        df["brick"] = df["brick"].apply(normalize_brick)

    attr_columns = {
        "pattern": "pattern",
        "print_": "print_",
        "style": "style",
        "sleeve": "sleeve",
        "neck_type": "neck_type",
        "length": "length",
        "color": "color",
    }
    for col, attr_type in attr_columns.items():
        if col in df.columns:
            if col == "color":
                df[col] = df[col].apply(normalize_color)
            else:
                df[col] = df[col].apply(lambda x: normalize_value(x, attr_type))

    # Coerce score columns to float
    score_cols = ["current_score", "m1_score", "m2_score", "m3_score", "m4_score"]
    for col in score_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Uppercase key string columns
    for col in ["gender", "category", "business_label", "confidence", "trajectory"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: str(x).strip().upper() if pd.notna(x) and str(x).strip() else None
            )

    logger.info(f"Normalized Trends: {len(df)} rows, {df['brick'].nunique() if 'brick' in df.columns else '?'} bricks")
    for col, attr_type in attr_columns.items():
        if col in df.columns:
            non_null = df[col].notna().sum()
            n_unique = df[col].nunique()
            logger.debug(f"  Trend norm [{col}]: {non_null}/{len(df)} non-null, {n_unique} unique values")
    return df


def normalize_catalog(catalog_df: pd.DataFrame, config: FTFConfig) -> pd.DataFrame:
    """Normalize product catalog attributes."""
    df = catalog_df.copy()

    if "brick" in df.columns:
        df["brick"] = df["brick"].apply(normalize_brick)

    attr_columns = {
        "pattern": "pattern",
        "fit": "fit",
        "sleeve": "sleeve",
        "neck_type": "neck_type",
        "length": "length",
        "color": "color",
    }
    for col, attr_type in attr_columns.items():
        if col in df.columns:
            if col == "color":
                df[col] = df[col].apply(normalize_color)
            else:
                df[col] = df[col].apply(lambda x: normalize_value(x, attr_type))

    if "fashion_grade" in df.columns:
        df["fashion_grade"] = df["fashion_grade"].apply(
            lambda x: normalize_fashion_grade(x, config.fashion_grade_map)
        )

    logger.info(f"Normalized Catalog: {len(df)} rows")
    return df


# ── Level Transformation ──────────────────────────────────────────────

def aggregate_brand_to_category(ra_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Brand-level RA to Category level (Segment × Family).

    Sums quantities, takes first value for non-aggregatable fields.
    """
    group_cols = ["segment", "family", "brick", "fashion_grade",
                  "pattern_type", "color", "fit", "neck_type",
                  "sleeve_type", "length", "month_of_drop", "product_group"]
    available = [c for c in group_cols if c in ra_df.columns]

    agg_dict = {}
    if "quantities_planned" in ra_df.columns:
        agg_dict["quantities_planned"] = "sum"
    if "mrp" in ra_df.columns:
        agg_dict["mrp"] = "mean"

    first_cols = [c for c in ra_df.columns if c not in available and c not in agg_dict]
    for c in first_cols:
        agg_dict[c] = "first"

    result = ra_df.groupby(available, dropna=False).agg(agg_dict).reset_index()
    logger.info(f"Aggregated Brand→Category: {len(ra_df)} → {len(result)} rows")
    return result


def disaggregate_category_to_brand(ra_df: pd.DataFrame,
                                   brand_share: pd.DataFrame) -> pd.DataFrame:
    """Disaggregate Category-level RA to Brand level using historical brand share.

    brand_share should have columns: [segment, family, brick, brand, share]
    where share is the proportion (0-1) of that brand within the category.
    """
    if brand_share is None or brand_share.empty:
        logger.warning("No brand share data provided; cannot disaggregate")
        return ra_df

    merge_cols = ["segment", "family", "brick"]
    available = [c for c in merge_cols if c in ra_df.columns and c in brand_share.columns]

    result = ra_df.merge(brand_share[available + ["brand", "share"]], on=available, how="left")

    if "quantities_planned" in result.columns and "share" in result.columns:
        result["quantities_planned"] = (result["quantities_planned"] * result["share"]).round().astype(int)

    result = result.drop(columns=["share"], errors="ignore")
    logger.info(f"Disaggregated Category→Brand: {len(ra_df)} → {len(result)} rows")
    return result


# ── Top-Level Normalization Pipeline ──────────────────────────────────

def run_normalization(ra_df: pd.DataFrame,
                      trend_df: pd.DataFrame,
                      config: FTFConfig,
                      catalog_df: pd.DataFrame = None,
                      brand_share_df: pd.DataFrame = None,
                      ) -> Dict[str, pd.DataFrame]:
    """Run the full normalization pipeline (Step 2).

    Returns a dict with keys: "ra", "trends", and optionally "catalog".
    """
    norm_ra = normalize_ra(ra_df, config)
    norm_trends = normalize_trends(trend_df, config)

    norm_catalog = None
    if catalog_df is not None:
        norm_catalog = normalize_catalog(catalog_df, config)

    # Level transformation if needed
    if config.level_transformation_needed and config.output_level:
        if config.primary_key_type == "BRAND" and config.output_level == "CATEGORY":
            norm_ra = aggregate_brand_to_category(norm_ra)
        elif config.primary_key_type == "CATEGORY" and config.output_level == "BRAND":
            norm_ra = disaggregate_category_to_brand(norm_ra, brand_share_df)

    result = {"ra": norm_ra, "trends": norm_trends}
    if norm_catalog is not None:
        result["catalog"] = norm_catalog

    return result
