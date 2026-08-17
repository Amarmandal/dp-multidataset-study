"""
Data Preparation Script for Membership Inference Attack (MIA) Results.

Consolidates all three attacks — LiRA, Yeom threshold MIA, and Shokri
shadow-model MIA — across all six datasets and all five model families into a
single tidy CSV.

Output format
-------------
One long/tidy table, one row per (attack, dataset, model, variant, epsilon):

    attack, dataset, dataset_dir, n_samples, n_classes, n_train, n_test,
    model, variant, epsilon, n_runs, <metric columns...>

``variant`` is ``standard`` for the non-private target (blank ``epsilon``) and
``dp`` for each privacy budget. Metric columns are the union across all three
attacks; cells are blank where a given attack does not produce that metric
(e.g. ``sd_in``/``sd_out`` are LiRA-only, ``loss_gap`` is Yeom-only).

Metric naming
-------------
All three attacks now write a bare metric column plus a ``<metric>_std``. LiRA
and Shokri vary run to run through shadow-model reseeding against a fixed target;
Yeom is deterministic (fixed exported target + fixed 80/20 split + a threshold
rule with no randomness), so it reports ``n_runs=1``, ``*_std=0.0`` and
``deterministic=True``. LiRA's and Yeom's bare names are aliased into the
canonical ``*_mean`` columns. Yeom's ``train_loss_mean`` / ``test_loss_mean`` are
left as-is: those are means over *samples*, not runs.

Provenance
----------
Reads ONLY the per-dataset CSVs under ``Attack/<attack>/results/<DATASET>/`` —
the single ground truth for attack metrics. The dataset-level header fields
(``n_classes`` / ``n_train`` / ``n_test``) come from each
``<DATASET>/data/processed_data.pkl``, which is the actual source of those
values.

Deliberately NOT read: any ``*.json`` (the drivers no longer emit any; the files
still on disk are stale pre-patch leftovers), the ``*_comparison.csv`` roll-ups,
``results_prepatch/``, and ``prereplication_backups/``. Adding any of those to a
glob here would reintroduce stale results.

Usage:
    python Results/MIA/prepare_mia_data.py

Output:
    Results/MIA/consolidated_mia_data.csv
"""

import csv
import pickle
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[2]
ATTACK_DIR = BASE_DIR / "Attack"
OUTPUT_FILE = Path(__file__).resolve().parent / "consolidated_mia_data.csv"

# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

# attack key -> (results subdirectory, per-dataset CSV filename template)
ATTACKS = {
    "lira": ("LiRA", "{ds}_lira_results.csv"),
    "yeom": ("MIA_YEOM", "{ds}_mia_results.csv"),
    "shokri": ("MIA_Shokri", "{ds}_results.csv"),
}

# directory name -> (display name, n_samples)
DATASETS = {
    "BCP": ("Breast Cancer", 569),
    "DIABETES": ("Diabetes", 70692),
    "CANCER_RISK": ("Cancer Risk", 1500),
    "GALLSTONE": ("Gallstone", 319),
    "KIDNEY_STONE": ("Kidney Stone", 4000),
    "LUNG_CANCER": ("Lung Cancer", 50000),
}

# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------

ID_COLUMNS = [
    "attack",
    "dataset",
    "dataset_dir",
    "n_samples",
    "n_classes",
    "n_train",
    "n_test",
    "model",
    "variant",
    "epsilon",
    "n_runs",
    # Yeom only. True records that the row came from a fixed exported target and
    # a fixed member/non-member split, so a repeat run reproduces it exactly and
    # every *_std is 0.0 by construction. Blank for LiRA/Shokri, which are
    # genuinely stochastic through shadow-model reseeding. Must not be dropped,
    # renamed, or inferred -- it is the evidence that Yeom's zero dispersion is
    # intentional.
    "deterministic",
]

# Shared across attacks
SHARED_METRICS = [
    "attack_auc_mean",
    "attack_auc_std",
    "advantage_mean",
    "advantage_std",
    "tpr_mean",
    "tpr_std",
    "fpr_mean",
    "fpr_std",
    "attack_accuracy_mean",
    "attack_accuracy_std",
    "attack_precision_mean",
    "attack_precision_std",
    "train_acc_mean",
    "train_acc_std",
    "test_acc_mean",
    "test_acc_std",
    "gen_gap_mean",
    "gen_gap_std",
    "n_members",
    "n_nonmembers",
]

# LiRA only
LIRA_METRICS = [
    "tpr_at_10pct",
    "tpr_at_10pct_std",
    "tpr_at_1pct",
    "tpr_at_1pct_std",
    "tpr_at_0p1pct",
    "tpr_at_0p1pct_std",
    "sd_in",
    "sd_in_std",
    "sd_out",
    "sd_out_std",
    "n_shadow_trained",
    "online",
    "balanced",
]

# Yeom only
YEOM_METRICS = [
    "threshold_tau",
    "threshold_tau_std",
    "train_loss_mean",
    "train_loss_std",
    "test_loss_mean",
    "test_loss_std",
    "loss_gap",
    "loss_gap_std",
]

METRIC_COLUMNS = SHARED_METRICS + LIRA_METRICS + YEOM_METRICS
COLUMNS = ID_COLUMNS + METRIC_COLUMNS

# Single-run column -> canonical *_mean column, per attack
ALIASES = {
    "lira": {
        "attack_auc": "attack_auc_mean",
        "advantage": "advantage_mean",
        "test_acc": "test_acc_mean",
    },
    "yeom": {
        "attack_auc": "attack_auc_mean",
        "advantage": "advantage_mean",
        "tpr": "tpr_mean",
        "fpr": "fpr_mean",
        "test_acc": "test_acc_mean",
    },
    "shokri": {},
}

NON_METRIC_CSV_FIELDS = {"dataset", "model", "variant", "epsilon", "n_runs"}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def to_number(value):
    """Coerce a CSV string to float/bool where possible, else return as-is."""
    if value is None or value == "":
        return None
    if value in ("True", "False"):
        return value == "True"
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


_HEADER_CACHE: dict = {}


def read_header(dataset_dir: str) -> dict:
    """Pull dataset-level metadata from ``<DATASET>/data/processed_data.pkl``.

    This pickle is the source of truth for the split: the attack drivers derive
    n_classes / n_train / n_test from exactly these arrays. Cached because all
    three attacks ask for the same dataset.
    """
    if dataset_dir in _HEADER_CACHE:
        return _HEADER_CACHE[dataset_dir]

    path = BASE_DIR / dataset_dir / "data" / "processed_data.pkl"
    if not path.exists():
        _HEADER_CACHE[dataset_dir] = {}
        return {}

    with path.open("rb") as f:
        blob = pickle.load(f)
    y_train = np.asarray(blob["y_train"]).ravel()
    y_test = np.asarray(blob["y_test"]).ravel()
    header = {
        "n_classes": int(len(np.unique(y_train))),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }
    _HEADER_CACHE[dataset_dir] = header
    return header


def blank_row() -> dict:
    return {column: "" for column in COLUMNS}


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def build_rows() -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    warnings: list[str] = []
    unmapped: set[str] = set()

    for attack, (subdir, csv_template) in ATTACKS.items():
        results_dir = ATTACK_DIR / subdir / "results"
        alias = ALIASES[attack]

        for dataset_dir, (display_name, n_samples) in DATASETS.items():
            csv_path = results_dir / dataset_dir / csv_template.format(ds=dataset_dir)

            if not csv_path.exists():
                warnings.append(f"{attack}/{dataset_dir}: missing {csv_path.name}")
                continue

            header = read_header(dataset_dir)
            if not header:
                warnings.append(
                    f"{attack}/{dataset_dir}: no processed_data.pkl; "
                    "n_classes/n_train/n_test left blank"
                )

            with csv_path.open(newline="") as f:
                records = list(csv.DictReader(f))

            for record in records:
                row = blank_row()
                row["attack"] = attack
                row["dataset"] = display_name
                row["dataset_dir"] = dataset_dir
                row["n_samples"] = n_samples
                for key in ("n_classes", "n_train", "n_test"):
                    value = header.get(key)
                    row[key] = "" if value is None else value

                row["model"] = record.get("model", "")
                row["variant"] = record.get("variant", "")
                epsilon = to_number(record.get("epsilon"))
                row["epsilon"] = "" if epsilon is None else epsilon
                n_runs = to_number(record.get("n_runs"))
                row["n_runs"] = "" if n_runs is None else n_runs

                for key, raw in record.items():
                    if key in NON_METRIC_CSV_FIELDS:
                        continue
                    column = alias.get(key, key)
                    if column not in COLUMNS:
                        unmapped.add(f"{attack}:{key}")
                        continue
                    value = to_number(raw)
                    if value is not None:
                        row[column] = value

                rows.append(row)

    for key in sorted(unmapped):
        warnings.append(f"unmapped CSV column dropped: {key}")

    return rows, warnings


def main() -> None:
    rows, warnings = build_rows()

    with OUTPUT_FILE.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {OUTPUT_FILE.relative_to(BASE_DIR)}")
    print(f"  {len(rows)} rows, {len(COLUMNS)} columns")
    for attack in ATTACKS:
        count = sum(1 for r in rows if r["attack"] == attack)
        print(f"    {attack:8s} {count} rows")
    n_standard = sum(1 for r in rows if r["variant"] == "standard")
    print(f"  {n_standard} standard / {len(rows) - n_standard} DP")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")


if __name__ == "__main__":
    main()
