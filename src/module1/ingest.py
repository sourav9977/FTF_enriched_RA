"""Step 1: Data Ingestion & Validation.

Ingest RA, Trend Report, AOP, Sales Data, Product Catalog, and Site Master.
Validate completeness and schema conformity. Produce a rejection log.

All schemas and column aliases are loaded from config/data_mapping.yaml.
"""

import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import yaml
import pandas as pd

from src.pipeline_logger import get_logger, StepLogger

logger = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_MAPPING_FILE = _PROJECT_ROOT / "config" / "data_mapping.yaml"


# ── Schema Loading from YAML ─────────────────────────────────────────

@dataclass
class ColumnRule:
    internal_name: str
    required: bool = True
    dtype: str = "str"
    aliases: List[str] = field(default_factory=list)
    maps_to_trend: Optional[str] = None
    maps_to_ra: Optional[str] = None


def _load_mapping_yaml(path: Path = None) -> dict:
    """Load the data mapping YAML file."""
    p = path or _MAPPING_FILE
    if not p.exists():
        raise FileNotFoundError(f"Data mapping file not found: {p}")
    with open(p, "r") as f:
        return yaml.safe_load(f)


def _parse_schema(entity_config: dict) -> List[ColumnRule]:
    """Parse a YAML entity config into a list of ColumnRule objects."""
    rules = []
    for col_def in entity_config.get("columns", []):
        rules.append(ColumnRule(
            internal_name=col_def["name"],
            required=col_def.get("required", True),
            dtype=col_def.get("dtype", "str"),
            aliases=col_def.get("aliases", []),
            maps_to_trend=col_def.get("maps_to_trend"),
            maps_to_ra=col_def.get("maps_to_ra"),
        ))
    return rules


class DataMapping:
    """Loads and holds all schemas from the YAML config file."""

    def __init__(self, yaml_path: Path = None):
        self._raw = _load_mapping_yaml(yaml_path)
        self.ra_schema = _parse_schema(self._raw["ra"])
        self.trend_schema = _parse_schema(self._raw["trend_report"])
        self.sales_schema = _parse_schema(self._raw["sales"])
        self.catalog_schema = _parse_schema(self._raw["catalog"])
        self.aop_schema = _parse_schema(self._raw["aop"])
        self.site_master_schema = _parse_schema(self._raw["site_master"])

        self.trend_dedup_map = self._raw.get("trend_report", {}).get("dedup_column_map", {})
        self.trend_header_keywords = set(
            self._raw.get("trend_report", {}).get("header_keywords", [])
        )
        self.attribute_mapping = self._raw.get("attribute_mapping", {})

    @property
    def ra_to_trend_map(self) -> Dict[str, str]:
        return self.attribute_mapping.get("ra_to_trend", {})

    @property
    def trend_to_ra_map(self) -> Dict[str, str]:
        return self.attribute_mapping.get("trend_to_ra", {})

    def get_schema(self, entity_name: str) -> List[ColumnRule]:
        lookup = {
            "ra": self.ra_schema,
            "trend_report": self.trend_schema,
            "trends": self.trend_schema,
            "sales": self.sales_schema,
            "catalog": self.catalog_schema,
            "aop": self.aop_schema,
            "site_master": self.site_master_schema,
        }
        if entity_name.lower() not in lookup:
            raise ValueError(f"Unknown entity: {entity_name}")
        return lookup[entity_name.lower()]


# Singleton — loaded once at module level
_mapping: DataMapping = None


def get_mapping(yaml_path: Path = None) -> DataMapping:
    """Return the global DataMapping singleton (lazy-loaded)."""
    global _mapping
    if _mapping is None:
        _mapping = DataMapping(yaml_path)
    return _mapping


def reload_mapping(yaml_path: Path = None) -> DataMapping:
    """Force-reload the mapping from YAML."""
    global _mapping
    _mapping = DataMapping(yaml_path)
    return _mapping


# ── Validation Result ─────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """Captures the outcome of validating a single dataset."""
    entity_name: str
    total_rows: int = 0
    accepted_rows: int = 0
    rejected_rows: int = 0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    rejection_log: List[Dict] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> str:
        lines = [
            f"--- {self.entity_name} ---",
            f"  Total rows:    {self.total_rows}",
            f"  Accepted rows: {self.accepted_rows}",
            f"  Rejected rows: {self.rejected_rows}",
            f"  Warnings:      {len(self.warnings)}",
            f"  Errors:        {len(self.errors)}",
        ]
        for w in self.warnings[:5]:
            lines.append(f"    [WARN] {w}")
        for e in self.errors[:5]:
            lines.append(f"    [ERROR] {e}")
        if len(self.errors) > 5:
            lines.append(f"    ... and {len(self.errors) - 5} more errors")
        return "\n".join(lines)


@dataclass
class IngestionResult:
    """Full result of ingesting all datasets."""
    ra: Optional[Tuple[pd.DataFrame, ValidationResult]] = None
    trends: Optional[Tuple[pd.DataFrame, ValidationResult]] = None
    sales: Optional[Tuple[pd.DataFrame, ValidationResult]] = None
    catalog: Optional[Tuple[pd.DataFrame, ValidationResult]] = None
    aop: Optional[Tuple[pd.DataFrame, ValidationResult]] = None
    site_master: Optional[Tuple[pd.DataFrame, ValidationResult]] = None
    detected_ra_level: Optional[str] = None

    def all_valid(self) -> bool:
        for attr in ["ra", "trends", "sales", "catalog", "aop", "site_master"]:
            pair = getattr(self, attr)
            if pair is not None and not pair[1].is_valid:
                return False
        return True

    def summary(self) -> str:
        lines = [f"RA Level Detected: {self.detected_ra_level}", ""]
        for attr in ["ra", "trends", "sales", "catalog", "aop", "site_master"]:
            pair = getattr(self, attr)
            if pair is not None:
                lines.append(pair[1].summary())
                lines.append("")
        return "\n".join(lines)


# ── Core Ingestion Functions ──────────────────────────────────────────

def read_file(path: str) -> pd.DataFrame:
    """Read a CSV or Excel file into a DataFrame."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = p.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    elif suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    elif suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    else:
        raise ValueError(f"Unsupported file format: {suffix}. Use CSV, TSV, or Excel.")


def _build_alias_map(schema: List[ColumnRule]) -> Dict[str, str]:
    """Build a lowercased alias -> internal_name mapping from a schema."""
    alias_map = {}
    for rule in schema:
        alias_map[rule.internal_name.lower()] = rule.internal_name
        for alias in rule.aliases:
            alias_map[alias.lower()] = rule.internal_name
    return alias_map


def map_columns(df: pd.DataFrame, schema: List[ColumnRule]) -> pd.DataFrame:
    """Rename DataFrame columns to internal names using the schema's aliases.

    Uses first-match-wins to prevent duplicate target columns.
    """
    alias_map = _build_alias_map(schema)
    rename = {}
    mapped_targets = set()

    for col in df.columns:
        key = str(col).strip().lower()
        if key in alias_map:
            target = alias_map[key]
            if target not in mapped_targets:
                rename[col] = target
                mapped_targets.add(target)

    logger.debug(f"  Column mapping: {len(rename)} of {len(df.columns)} cols mapped")
    for raw, internal in sorted(rename.items(), key=lambda x: x[1]):
        logger.debug(f"    {raw!r:30s} -> {internal}")
    unmapped = [c for c in df.columns if c not in rename]
    if unmapped:
        logger.debug(f"  Unmapped columns ({len(unmapped)}): {unmapped[:10]}")

    return df.rename(columns=rename)


def validate_schema(df: pd.DataFrame, schema: List[ColumnRule],
                    entity_name: str) -> Tuple[pd.DataFrame, ValidationResult]:
    """Validate a DataFrame against a schema. Returns (cleaned_df, result).

    - Rows with missing mandatory fields are rejected.
    - Optional null columns are flagged as warnings.
    - Type coercion is attempted for numeric/date columns.
    """
    result = ValidationResult(entity_name=entity_name, total_rows=len(df))

    for rule in schema:
        if rule.required and rule.internal_name not in df.columns:
            result.errors.append(f"Required column '{rule.internal_name}' is missing")

    if result.errors:
        result.accepted_rows = 0
        result.rejected_rows = len(df)
        return df, result

    for rule in schema:
        if rule.internal_name not in df.columns:
            continue
        col = rule.internal_name
        if rule.dtype == "int":
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif rule.dtype == "float":
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif rule.dtype == "date":
            df[col] = pd.to_datetime(df[col], errors="coerce")

    required_cols = [r.internal_name for r in schema if r.required and r.internal_name in df.columns]
    mask_valid = pd.Series(True, index=df.index)

    for col in required_cols:
        null_mask = df[col].isna()
        if null_mask.any():
            null_indices = df.index[null_mask].tolist()
            for idx in null_indices:
                result.rejection_log.append({
                    "row": idx,
                    "reason": f"Missing mandatory field: {col}",
                })
            mask_valid &= ~null_mask

    optional_cols = [r.internal_name for r in schema
                     if not r.required and r.internal_name in df.columns]
    for col in optional_cols:
        null_pct = df[col].isna().mean()
        if null_pct == 1.0:
            result.warnings.append(f"Optional column '{col}' is entirely null")
        elif null_pct > 0.5:
            result.warnings.append(
                f"Optional column '{col}' is {null_pct:.0%} null — treated as wildcard in matching"
            )

    accepted = df[mask_valid].copy()
    rejected_count = len(df) - len(accepted)

    result.accepted_rows = len(accepted)
    result.rejected_rows = rejected_count
    if rejected_count > 0:
        result.warnings.append(f"{rejected_count} rows rejected for missing mandatory fields")

    logger.debug(
        f"  Validation [{entity_name}]: {result.accepted_rows}/{result.total_rows} accepted, "
        f"{result.rejected_rows} rejected, {len(result.warnings)} warnings, {len(result.errors)} errors"
    )
    for w in result.warnings:
        logger.debug(f"    WARN: {w}")
    for e in result.errors:
        logger.debug(f"    ERROR: {e}")

    return accepted, result


# ── Trend Report Special Handling ─────────────────────────────────────

def _auto_detect_trend_header(df: pd.DataFrame,
                               header_keywords: set = None) -> pd.DataFrame:
    """Handle merged Excel headers in trend reports.

    Scans the first few rows for known header keywords and promotes
    that row to the column header. De-duplicates repeating column
    names (Month, Score, Stage, etc.) with _2, _3, _4 suffixes.
    """
    if header_keywords is None:
        header_keywords = get_mapping().trend_header_keywords

    current_cols_lower = {str(c).strip().lower() for c in df.columns}
    if len(current_cols_lower & header_keywords) >= 3:
        df.columns = _make_unique(list(df.columns))
        return df

    for i in range(min(5, len(df))):
        row_vals = {str(v).strip().lower() for v in df.iloc[i] if pd.notna(v)}
        matches = row_vals & header_keywords
        if len(matches) >= 3:
            raw_names = [
                str(v).strip() if pd.notna(v) else f"col_{j}"
                for j, v in enumerate(df.iloc[i])
            ]
            df.columns = _make_unique(raw_names)
            df = df.iloc[i + 1:].reset_index(drop=True)
            return df

    return df


def _make_unique(columns: list) -> list:
    """Append _2, _3, etc. to duplicate column names."""
    seen = {}
    result = []
    for c in columns:
        key = str(c).lower()
        if key in seen:
            seen[key] += 1
            result.append(f"{c}_{seen[key]}")
        else:
            seen[key] = 1
            result.append(c)
    return result


def _map_trend_dedup_columns(df: pd.DataFrame,
                              dedup_map: Dict[str, str] = None) -> pd.DataFrame:
    """Map de-duplicated M1-M4 forecast columns to internal names."""
    if dedup_map is None:
        dedup_map = get_mapping().trend_dedup_map
    rename = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in dedup_map:
            rename[col] = dedup_map[key]
    return df.rename(columns=rename)


# ── RA Level Detection ────────────────────────────────────────────────

def detect_ra_level(ra_df: pd.DataFrame) -> str:
    """Detect whether the RA is at Brand or Category level.

    Brand level: the 'brand' column is populated and unique per row.
    Category level: otherwise (uses Segment x Family as primary key).
    """
    if "brand" in ra_df.columns and ra_df["brand"].notna().all():
        return "BRAND"
    return "CATEGORY"


# ── Top-Level Ingest Functions ────────────────────────────────────────

def ingest_ra(path: str) -> Tuple[pd.DataFrame, ValidationResult]:
    """Ingest and validate an RA file."""
    m = get_mapping()
    df = read_file(path)
    df = map_columns(df, m.ra_schema)
    df, result = validate_schema(df, m.ra_schema, "RA")
    return df, result


def ingest_trends(path: str) -> Tuple[pd.DataFrame, ValidationResult]:
    """Ingest and validate a Trend Report file."""
    m = get_mapping()
    df = read_file(path)
    df = _auto_detect_trend_header(df, m.trend_header_keywords)
    df = _map_trend_dedup_columns(df, m.trend_dedup_map)
    df = map_columns(df, m.trend_schema)
    df, result = validate_schema(df, m.trend_schema, "Trend Report")
    return df, result


def ingest_sales(path: str) -> Tuple[pd.DataFrame, ValidationResult]:
    """Ingest and validate a Sales Data file."""
    m = get_mapping()
    df = read_file(path)
    df = map_columns(df, m.sales_schema)
    df, result = validate_schema(df, m.sales_schema, "Sales Data")
    return df, result


def ingest_catalog(path: str) -> Tuple[pd.DataFrame, ValidationResult]:
    """Ingest and validate a Product Catalog file."""
    m = get_mapping()
    df = read_file(path)
    df = map_columns(df, m.catalog_schema)
    df, result = validate_schema(df, m.catalog_schema, "Product Catalog")
    return df, result


def ingest_aop(path: str) -> Tuple[pd.DataFrame, ValidationResult]:
    """Ingest and validate an AOP file."""
    m = get_mapping()
    df = read_file(path)
    df = map_columns(df, m.aop_schema)
    df, result = validate_schema(df, m.aop_schema, "AOP")
    return df, result


def ingest_site_master(path: str) -> Tuple[pd.DataFrame, ValidationResult]:
    """Ingest and validate a Site Master file."""
    m = get_mapping()
    df = read_file(path)
    df = map_columns(df, m.site_master_schema)
    df, result = validate_schema(df, m.site_master_schema, "Site Master")
    return df, result
