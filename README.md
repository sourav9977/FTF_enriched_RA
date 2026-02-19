# FTF Enriched RA — Prototype Pipeline

A modular prototype for the **Fashion Trend Forecasting (FTF) Range Architecture** system. It ingests a Range Architecture (RA) plan and a Trend Report, enriches the RA with trend intelligence, estimates optimized quantities, and enforces business constraints — all within an AOP (Annual Operating Plan) budget envelope.

---

## Architecture

The pipeline is split into two independent modules at the **Enriched RA** boundary, enabling A/B testing of different enrichment strategies.

```
┌─────────────────────────────────────────────────┐
│         MODULE 1: ENRICHMENT ENGINE              │
│         (Swappable for A/B testing)              │
│                                                  │
│   Config → Ingest → Normalize → Factor Analysis  │
│                → Enriched RA Generation           │
│                         │                        │
│                         ▼                        │
│               ┌─────────────────┐                │
│               │  ENRICHED RA    │  ← A/B point   │
│               └────────┬────────┘                │
└────────────────────────┼─────────────────────────┘
                         │
┌────────────────────────┼─────────────────────────┐
│         MODULE 2: QUANTITY & CONSTRAINT ENGINE    │
│         (Common across systems)                  │
│                         │                        │
│   AOP Breakdown → Qty Estimation (Approved)      │
│     → Qty Estimation (Added) → Constraints       │
│       → Final Output                             │
└──────────────────────────────────────────────────┘
```

---

## Module 1 — Enrichment Engine

| Step | Name | Purpose |
|------|------|---------|
| 0 | Configuration | Load and freeze all parameters (thresholds, weights, caps) |
| 1 | Data Ingestion & Validation | Schema validation, reject/flag rows, detect RA level |
| 2 | Data Normalization | Synonym mapping, taxonomy alignment, fashion grade standardization |
| 3 | Factor Analysis | Data-driven attribute weighting per Brick via correlation analysis |
| 4 | Enriched RA Generation | Overlap scoring, confidence scoring, 3-tier classification, FTF Added generation |

**Output:** Enriched RA — a single dataset containing ORIGINAL, FTF APPROVED, and FTF ADDED items with trend metadata and confidence scores.

### Item Classification

- **FTF APPROVED** — RA item fully matches a trend (overlap >= threshold)
- **ORIGINAL** — RA item partially matches or outperforms the trend on confidence
- **FTF ADDED** — New item sourced from an unmatched trend or a confidence-based replacement

---

## Module 2 — Quantity & Constraint Engine

| Step | Name | Purpose |
|------|------|---------|
| 5 | AOP Breakdown | Decompose AOP targets to Brick/attribute level using historical patterns |
| 6 | Qty Estimation (Approved) | Adjust planned quantities using performance score + shelf life cap |
| 7 | Qty Estimation (Added) | Estimate quantities via nearest neighbor / fallback hierarchy + trend influence |
| 8 | Constraint Optimization | Enforce MOQ, AOP budget, intake margin, FTF cap; rebalance as needed |
| 9 | Output Generation | Final RA file with full schema + summary metrics |

---

## Input Files

| File | Description |
|------|-------------|
| RA (Range Architecture) | Base range plan with planned quantities per option |
| Trend Report | Trend intelligence — scores, confidence levels, lifecycle stages |
| Product Catalog | SKU-level attributes for normalization and factor analysis |
| Sales Data | Historical performance (NSV, ROS, STR, turns) |
| AOP | Budget envelope at Brand/Category × Format × Fashion Grade × Month |
| Site Master | Store-level data (placeholder for prototype) |

---

## Key Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `overlap_threshold` | 100% | Minimum overlap to classify as FTF APPROVED |
| `replacement_threshold` | 60% | Below this, confidence-based replacement triggers |
| `ftf_added_cap_pct` | 20% | Max FTF Added items as % of total RA |
| `max_adjustment_pct` | ±10% | Max quantity adjustment for Approved items |
| `moq_per_option` | 2500 | Minimum order quantity per item |
| `shelf_life_days` | 60 | Target shelf life window |
| `str_target` | 75% | Sell-through rate target within shelf life |
| `k_neighbors` | 5 | Neighbors for quantity estimation of Added items |

---

## Project Structure

```
FTF_enriched_RA/
├── README.md
├── config/                  # Configuration files and defaults
├── data/                    # Input data files (not committed)
├── src/
│   ├── module1/             # Enrichment Engine (Steps 0–4)
│   │   ├── config.py
│   │   ├── ingest.py
│   │   ├── normalize.py
│   │   ├── factor_analysis.py
│   │   └── enrichment.py
│   ├── module2/             # Quantity & Constraint Engine (Steps 5–9)
│   │   ├── aop_breakdown.py
│   │   ├── qty_approved.py
│   │   ├── qty_added.py
│   │   ├── constraints.py
│   │   └── output.py
│   └── utils/               # Shared utilities
├── tests/                   # Unit and integration tests
├── notebooks/               # Exploration and verification notebooks
├── output/                  # Generated output files (not committed)
└── requirements.txt
```

---

## Prototype Scope

**Included:**
- RA + Trend Report ingestion through to enriched RA
- Quantity estimation with shelf life / ROS optimization
- AOP-constrained final quantities
- Full audit trail (scores, reasons, estimation methods)

**Excluded (future phases):**
- Approval workflow
- Stock / inventory data integration
- Seasonality and event-based adjustments

---

## Getting Started

```bash
# Clone the repository
git clone https://github.com/sourav9977/FTF_enriched_RA.git
cd FTF_enriched_RA

# Install dependencies
pip install -r requirements.txt

# Place input files in data/ directory

# Run the pipeline (instructions will be updated as modules are built)
```

---

## License

Internal use only.
