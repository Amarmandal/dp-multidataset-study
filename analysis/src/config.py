"""Configuration switches for the paper artifact pipeline.

Everything downstream honours these three flags. Nothing here changes the
underlying CSVs -- only what is displayed and which rows are included.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Referee-facing switches
# --------------------------------------------------------------------------

INCLUDE_LUNG_CANCER = True   # flip to False to drop it from leakage tables/figures
UTILITY_LABEL = "ACL"        # display label; do NOT rename the CSV column
RANDOM_FPR = 0.01            # reference FPR for TPR@1%

# Reason surfaced in .tex footnotes when INCLUDE_LUNG_CANCER is False.
LUNG_CANCER_EXCLUSION_REASON = (
    "Lung Cancer is excluded from this table: it is the only three-class dataset "
    "in the study and its non-private targets reach 100\\% test accuracy for four "
    "of five model families, so its leakage numbers are not comparable with the "
    "five binary datasets."
)

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[2]
ANALYSIS = REPO / "analysis"

TABLES_CSV = ANALYSIS / "tables" / "csv"
TABLES_TEX = ANALYSIS / "tables" / "tex"
FIGURES_PDF = ANALYSIS / "figures" / "pdf"
FIGURES_PNG = ANALYSIS / "figures" / "png"
LOGS = ANALYSIS / "logs"

UTILITY_CSV = REPO / "Results" / "dataset_results" / "consolidated_data.csv"
MIA_CSV = REPO / "Results" / "attack_results" / "consolidated_mia_data.csv"

WITHIN_PAIR = ANALYSIS / "stats" / "within_pair"
CORRELATION_STATS = ANALYSIS / "stats" / "correlation_stats"
RQ_STATS = ANALYSIS / "stats" / "rq"

# --------------------------------------------------------------------------
# Grids
# --------------------------------------------------------------------------

# Nine-point grid: the intersection of the utility sweep and the attack grid.
PAIRED_EPSILONS = [0.1, 0.2, 0.4, 0.8, 1.0, 2.0, 4.0, 8.0, 10.0]

EPS_TARGET = 1.0             # epsilon used by the residual-leakage tables/figures

# Literal placeholder written into any cell whose value is not on disk.
MISSING = "[MISSING: {what}]"
CONFLICT = "[CONFLICT: see gaps.md]"
