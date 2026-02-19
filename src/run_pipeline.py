"""Full FTF Pipeline Orchestrator with step-level logging.

Runs Steps 0-9 end-to-end, wrapping each step in a StepLogger for
comprehensive calculation tracing that logs to both console and file.

Usage:
    python -m src.run_pipeline \
        --ra path/to/ra.csv \
        --trends path/to/trends.xlsx \
        --sales path/to/sales.csv \
        --catalog path/to/catalog.csv \
        --aop path/to/aop.csv \
        --output output/enriched_ra.xlsx
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.pipeline_logger import setup_logging, StepLogger, get_logger, ItemTracer
from src.config import FTFConfig
from src.module1.ingest import (
    get_mapping, read_file, map_columns, validate_schema,
    _auto_detect_trend_header, _map_trend_dedup_columns,
)
from src.module1.normalize import run_normalization
from src.module1.factor_analysis import run_factor_analysis, run_factor_analysis_sales_only
from src.module1.overlap import (
    score_all_overlaps, compute_ra_confidence, compute_ftf_confidence,
    classify_items, generate_ftf_added_from_unmatched, assemble_enriched_ra,
)
from src.module2.aop_breakdown import run_aop_breakdown
from src.module2.quantity_estimation import run_quantity_estimation
from src.module2.rebalancing import run_rebalancing
from src.module2.output import generate_output, export_to_file

logger = get_logger(__name__)


def run_pipeline(
    ra_path: str,
    trends_path: str,
    sales_path: str,
    catalog_path: str = None,
    aop_path: str = None,
    output_path: str = "output/enriched_ra.xlsx",
    config_overrides: dict = None,
    trace_item_id: str = None,
) -> pd.DataFrame:
    """Execute the full FTF pipeline.

    Args:
        ra_path: Path to RA file (CSV/XLSX).
        trends_path: Path to Trend Report file (CSV/XLSX).
        sales_path: Path to Sales data file (CSV/XLSX).
        catalog_path: Path to Product Catalog file (CSV/XLSX).
        aop_path: Path to AOP targets file (CSV/XLSX).
        output_path: Path to write the output file.
        config_overrides: Optional dict of config parameter overrides.
        trace_item_id: Optional RA item ID to trace through the pipeline.

    Returns:
        The final Enriched RA DataFrame.
    """
    setup_logging()
    mapping = get_mapping()
    tracer = ItemTracer(trace_item_id) if trace_item_id else None

    # ── Step 0: Configuration ─────────────────────────────────────────
    with StepLogger("0", "Configuration", "src.step_0") as step:
        config = FTFConfig(**(config_overrides or {}))
        step.log_input("config_overrides", config_overrides or {})
        step.log_summary(
            overlap_threshold=config.overlap_threshold,
            replacement_threshold=config.replacement_threshold,
            ftf_added_cap=config.ftf_added_cap_pct,
            moq=config.moq_per_option,
            shelf_life=config.shelf_life_days,
        )

    # ── Step 1: Data Ingestion & Validation ───────────────────────────
    with StepLogger("1", "Data Ingestion & Validation", "src.step_1") as step:
        ra_raw = read_file(ra_path)
        ra_schema = mapping.get_schema("ra")
        ra_df = map_columns(ra_raw, ra_schema)
        ra_df, ra_val = validate_schema(ra_df, ra_schema, entity_name="RA")
        step.log_input("RA", ra_df, f"from {ra_path}")

        trends_raw = read_file(trends_path)
        trends_raw = _auto_detect_trend_header(trends_raw)
        trends_raw = _map_trend_dedup_columns(trends_raw)
        trend_schema = mapping.get_schema("trend_report")
        trends_df = map_columns(trends_raw, trend_schema)
        trends_df, trend_val = validate_schema(trends_df, trend_schema, entity_name="Trend Report")
        step.log_input("Trends", trends_df, f"from {trends_path}")

        sales_raw = read_file(sales_path)
        sales_schema = mapping.get_schema("sales")
        sales_df = map_columns(sales_raw, sales_schema)
        sales_df, sales_val = validate_schema(sales_df, sales_schema, entity_name="Sales")
        step.log_input("Sales", sales_df, f"from {sales_path}")

        catalog_df = None
        if catalog_path:
            cat_raw = read_file(catalog_path)
            cat_schema = mapping.get_schema("catalog")
            catalog_df = map_columns(cat_raw, cat_schema)
            catalog_df, cat_val = validate_schema(catalog_df, cat_schema, entity_name="Catalog")
            step.log_input("Catalog", catalog_df, f"from {catalog_path}")

        aop_df = None
        if aop_path:
            aop_raw = read_file(aop_path)
            aop_schema = mapping.get_schema("aop")
            aop_df = map_columns(aop_raw, aop_schema)
            aop_df, aop_val = validate_schema(aop_df, aop_schema, entity_name="AOP")
            step.log_input("AOP", aop_df, f"from {aop_path}")

        level = config.detect_ra_level(ra_df)
        step.log_summary(
            ra_level=level,
            ra_rows=ra_val.accepted_rows,
            ra_rejected=ra_val.rejected_rows,
            trend_rows=trend_val.accepted_rows,
            sales_rows=sales_val.accepted_rows,
        )

    # ── Step 2: Data Normalization ────────────────────────────────────
    with StepLogger("2", "Data Normalization", "src.step_2") as step:
        norm_result = run_normalization(ra_df, trends_df, config, catalog_df)
        ra_df = norm_result["ra"]
        trends_df = norm_result["trends"]
        catalog_df = norm_result.get("catalog", catalog_df)
        step.log_output("RA", ra_df, f"{ra_df['brick'].nunique()} bricks" if "brick" in ra_df.columns else "")
        step.log_output("Trends", trends_df)

    # ── Step 3: Factor Analysis ───────────────────────────────────────
    with StepLogger("3", "Factor Analysis", "src.step_3") as step:
        step.log_input("Sales", sales_df)
        step.log_input("Catalog", catalog_df if catalog_df is not None else "None")

        factor_result = run_factor_analysis(sales_df, catalog_df)
        if not factor_result.brick_weights:
            step.log_warning("Combined mode yielded no weights, trying sales-only mode")
            factor_result = run_factor_analysis_sales_only(sales_df)

        step.log_summary(
            bricks_with_weights=len(factor_result.brick_weights),
            bricks_analyzed=factor_result.bricks_analyzed,
            bricks_skipped=factor_result.bricks_skipped,
        )
        for bw in factor_result.brick_weights.values():
            step.log_calc(
                "brick_weights",
                brick=bw.brick,
                n=bw.sample_size,
                detail=", ".join(f"{k}={v:.3f}" for k, v in sorted(bw.weights.items(), key=lambda x: -x[1])),
            )

    # ── Step 4a: Overlap Scoring ──────────────────────────────────────
    with StepLogger("4a", "Overlap Scoring", "src.step_4a") as step:
        overlap_scores = score_all_overlaps(ra_df, trends_df, factor_result)
        step.log_output("overlap_pairs", overlap_scores)
        if not overlap_scores.empty:
            step.log_summary(
                pairs=len(overlap_scores),
                avg_score=overlap_scores["overlap_score"].mean(),
                max_score=overlap_scores["overlap_score"].max(),
                above_threshold=(overlap_scores["overlap_score"] >= config.overlap_threshold).sum(),
            )

    # ── Step 4b: Confidence Scores ────────────────────────────────────
    with StepLogger("4b", "Confidence Scores", "src.step_4b") as step:
        ra_confidence = compute_ra_confidence(ra_df, sales_df, config)
        ftf_confidence = compute_ftf_confidence(trends_df, config)
        step.log_output("ra_confidence", ra_confidence)
        step.log_output("ftf_confidence", ftf_confidence)

    # ── Step 4c: Classification ───────────────────────────────────────
    with StepLogger("4c", "Three-Tier Classification", "src.step_4c") as step:
        classification = classify_items(
            ra_df, trends_df, overlap_scores,
            ra_confidence, ftf_confidence,
            factor_result, config,
        )
        step.log_summary(
            approved=len(classification.approved),
            original=len(classification.original),
            replaced=len(classification.added_replacements),
            removed_ra=len(classification.removed_ra),
        )

    # ── Step 4d: FTF Added from Unmatched ─────────────────────────────
    with StepLogger("4d", "FTF Added from Unmatched", "src.step_4d") as step:
        added_items = generate_ftf_added_from_unmatched(
            trends_df, ra_df, overlap_scores,
            ftf_confidence,
            classification._used_trend_indices,
            config,
        )
        step.log_output("added_from_unmatched", f"{len(added_items)} items")

        enriched_ra = assemble_enriched_ra(classification, added_items)
        step.log_output("enriched_ra", enriched_ra)
        step.log_summary(
            total=len(enriched_ra),
            approved=(enriched_ra["ftf_status"] == "APPROVED").sum(),
            original=(enriched_ra["ftf_status"] == "ORIGINAL").sum(),
            added=(enriched_ra["ftf_status"] == "ADDED").sum(),
        )

    # ── Step 5: AOP Breakdown ─────────────────────────────────────────
    if aop_df is not None:
        with StepLogger("5", "AOP Breakdown", "src.step_5") as step:
            aop_result = run_aop_breakdown(
                aop_df, sales_df, enriched_ra, config, catalog_df,
            )
            step.log_output("brick_targets", aop_result.brick_targets)
            step.log_summary(
                total_aop=aop_result.total_aop_target,
                total_projected=aop_result.total_projected,
                scaling=aop_result.scaling_factor,
            )
    else:
        aop_result = None
        logger.info("Skipping Step 5 (AOP Breakdown) — no AOP file provided")

    # ── Steps 6-7: Quantity Estimation ────────────────────────────────
    with StepLogger("6-7", "Quantity Estimation", "src.step_67") as step:
        qty_result = run_quantity_estimation(
            enriched_ra, sales_df, config, factor_result, catalog_df,
        )
        enriched_ra = qty_result.final_ra
        step.log_output("enriched_ra", enriched_ra)
        step.log_summary(
            approved_adj=qty_result.approved_adjustments,
            added_est=qty_result.added_estimates,
            shelf_caps=qty_result.shelf_life_caps_applied,
            fallbacks=qty_result.fallback_count,
        )

    # ── Step 8: Rebalancing ───────────────────────────────────────────
    with StepLogger("8", "Constraint Optimization & Rebalancing", "src.step_8") as step:
        aop_budget = aop_result.total_aop_target if aop_result else 1e12
        rebal_result = run_rebalancing(enriched_ra, aop_budget, config)
        enriched_ra = rebal_result.final_ra
        step.log_output("enriched_ra", enriched_ra)
        step.log_summary(
            retained=len(enriched_ra),
            removed=len(rebal_result.items_removed),
            moq_boosts=rebal_result.moq_boosts,
            moq_drops=rebal_result.moq_drops,
            budget_cuts=rebal_result.budget_reductions,
            ftf_cap_drops=rebal_result.ftf_cap_drops,
        )

    # ── Step 9: Output Generation ─────────────────────────────────────
    with StepLogger("9", "Output Generation", "src.step_9") as step:
        final_df, metrics = generate_output(
            enriched_ra,
            items_removed=len(rebal_result.items_removed),
            shelf_life_caps=qty_result.shelf_life_caps_applied,
        )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fmt = "xlsx" if output_path.endswith(".xlsx") else "csv"
        export_to_file(final_df, metrics, output_path, format=fmt)
        step.log_output("final_ra", final_df)
        step.log_summary(
            total_items=metrics.total_items,
            approved=f"{metrics.approved_count} ({metrics.approved_pct:.1%})",
            original=f"{metrics.original_count} ({metrics.original_pct:.1%})",
            added=f"{metrics.added_count} ({metrics.added_pct:.1%})",
            qty_delta=f"{metrics.qty_delta_pct:+.1%}",
        )

    if tracer:
        logger.info(tracer.report())

    logger.info(f"\nPipeline complete — output written to {output_path}")
    return final_df


def main():
    parser = argparse.ArgumentParser(description="FTF Enriched RA Pipeline")
    parser.add_argument("--ra", required=True, help="Path to RA file")
    parser.add_argument("--trends", required=True, help="Path to Trend Report file")
    parser.add_argument("--sales", required=True, help="Path to Sales data file")
    parser.add_argument("--catalog", help="Path to Product Catalog file")
    parser.add_argument("--aop", help="Path to AOP targets file")
    parser.add_argument("--output", default="output/enriched_ra.xlsx", help="Output file path")
    parser.add_argument("--trace-item", help="RA item ID to trace through pipeline")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])

    args = parser.parse_args()
    setup_logging(console_level=args.log_level)

    run_pipeline(
        ra_path=args.ra,
        trends_path=args.trends,
        sales_path=args.sales,
        catalog_path=args.catalog,
        aop_path=args.aop,
        output_path=args.output,
        trace_item_id=args.trace_item,
    )


if __name__ == "__main__":
    main()
