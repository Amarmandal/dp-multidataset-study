"""
Data Preparation Script for Cross-Dataset DP Comparison.

Walks every dataset/model directory, merges each model's ``dp_*_report.json``
with its ``dp_*results.csv`` (and each ``std_*_report.json`` with its
``baseline*results.csv``), normalises the differing key names into one schema,
recomputes ACL against the standard-model baseline, and writes a single tidy
CSV ready for plotting.

Output format
-------------
One long/tidy table, one row per (dataset, model, variant, epsilon):

    dataset, dataset_dir, n_samples, n_features, model, model_family,
    variant, epsilon, n_runs, <metric columns...>, source_dp, source_baseline

``variant`` is ``standard`` for the non-private baseline (blank ``epsilon``,
``ACL`` = 0) and ``dp`` for each privacy budget. Metric columns are the union
of everything recorded anywhere; cells are left blank where a given
dataset/model never recorded that metric.

Notes on data provenance
------------------------
* JSON and CSV were verified to agree on every shared field for all 30
  dataset/model pairs, so either may serve as the source. The CSV is preferred
  for metrics because it is the wider of the two; the JSON supplies ``n_runs``.
* ``ACL`` is always recomputed here as ``1 - acc_dp / acc_std`` using the
  baseline accuracy from the standard-model report. The per-model CSVs
  disagree with each other on which baseline they used, so their stored ACL is
  ignored (kept as ``ACL_as_reported`` is deliberately *not* done — see the
  repo notes; recomputation is the single source of truth).
* ``GALLSTONE/DNN`` has no ``dp_dnn_report.json``; its DP rows come from the
  CSV alone and ``n_runs`` is blank.

Usage:
    python Results/prepare_data.py

Output:
    Results/consolidated_data.csv
"""

import csv
import glob
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_FILE = Path(__file__).resolve().parent / "consolidated_data.csv"
MEASURED_EXPORTED_FILE = (
    Path(__file__).resolve().parent / "exported_model_accuracy_measured.csv"
)

# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

DATASETS = {
    "Breast Cancer": {"dir": "BCP", "n_samples": 569, "n_features": 30},
    "Diabetes": {"dir": "DIABETES", "n_samples": 70692, "n_features": 21},
    "Cancer Risk": {"dir": "CANCER_RISK", "n_samples": 1500, "n_features": 8},
    "Gallstone": {"dir": "GALLSTONE", "n_samples": 319, "n_features": 38},
    "Kidney Stone": {"dir": "KIDNEY_STONE", "n_samples": 4000, "n_features": 23},
    "Lung Cancer": {"dir": "LUNG_CANCER", "n_samples": 50000, "n_features": 23},
}

# model display name -> directory name
MODELS = {
    "DP-RF": "RandomForest",
    "DP-LR": "LR",
    "DP-GNB": "GaussianNB",
    "DP-SVM": "SVM",
    "DP-DNN": "DNN",
}

# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------

ID_COLUMNS = [
    "dataset",
    "dataset_dir",
    "n_samples",
    "n_features",
    "model",
    "model_family",
    "variant",
    "epsilon",
    "n_runs",
]

METRIC_COLUMNS = [
    "accuracy_mean",
    "accuracy_std",
    "f1_score_mean",
    "f1_score_std",
    "precision_mean",
    "precision_std",
    "recall_mean",
    "recall_std",
    "balanced_accuracy_mean",
    "balanced_accuracy_std",
    "roc_auc_mean",
    "roc_auc_std",
    "train_accuracy_mean",
    "train_time_mean",
    "train_time_std",
    "ACL",
    "accuracy_loss_pct",
    # The exported DP artifact is the run-0 draw (seed = 0*10 + 42 = 42): every
    # DP_*.ipynb calls export_model() inside `if run == 0:` in the same loop
    # iteration that appends to per_run_accuracies. So per_run_accuracies[0] is
    # that one artifact's accuracy, while accuracy_mean averages all N_RUNS
    # draws. Surfacing it lets a leakage number be paired with the accuracy of
    # the *same* pickle the MIA attacked, instead of a mean over models that
    # were never saved. Blank where the report has no per_run_accuracies.
    "exported_model_accuracy",
    "ACL_exported",
    # "report JSON (per_run_accuracies[0])" where the run recorded its per-run
    # list, or "measured from exported artifact" for the DP-DNN rows of BCP,
    # CANCER_RISK, DIABETES and GALLSTONE, whose reports predate that line and
    # whose run-0 accuracy was instead recovered by scoring the exported .pt /
    # .pkl itself (see measure_exported_accuracy.py). Lets a referee separate
    # what was logged at run time from what was read back off the artifact.
    "exported_model_accuracy_source",
]

PROVENANCE_COLUMNS = ["source_dp", "source_baseline"]

COLUMNS = ID_COLUMNS + METRIC_COLUMNS + PROVENANCE_COLUMNS

# Per-epsilon DP key aliases -> canonical name
DP_ALIAS = {
    "f1_mean": "f1_score_mean",
    "f1_std": "f1_score_std",
    "accuracy_loss_mean": "ACL",
}

# Baseline (single-row) key aliases -> canonical name
BASELINE_ALIAS = {
    "accuracy": "accuracy_mean",
    "f1_score": "f1_score_mean",
    "f1_std": "f1_score_std",
    "precision": "precision_mean",
    "recall": "recall_mean",
    "balanced_accuracy": "balanced_accuracy_mean",
    "roc_auc": "roc_auc_mean",
    "train_accuracy": "train_accuracy_mean",
    "train_time": "train_time_mean",
}


# --------------------------------------------------------------------------
# File discovery
# --------------------------------------------------------------------------


def find_files(dataset_dir: str, model_dir: str, pattern: str) -> list[Path]:
    """Locate result files under <dataset>/<model>/ or <dataset>/<model>/*/."""
    root = BASE_DIR / dataset_dir / model_dir
    hits = sorted(glob.glob(str(root / pattern))) + sorted(
        glob.glob(str(root / "*" / pattern))
    )
    # Skip archived/superseded reports such as old_dp_svm_report.json
    return [Path(h) for h in hits if not Path(h).name.startswith("old_")]


def find_one(dataset_dir: str, model_dir: str, pattern: str) -> Path | None:
    hits = find_files(dataset_dir, model_dir, pattern)
    if not hits:
        return None
    # Prefer a file inside an output/ subdirectory when duplicates exist
    for h in hits:
        if h.parent.name in ("output", "data"):
            return h
    return hits[0]


def rel(path: Path | None) -> str:
    return "" if path is None else str(path.relative_to(BASE_DIR))


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------


def read_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def to_number(value):
    """Coerce a CSV string to float/int where possible, else return as-is."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def normalise(record: dict, alias: dict) -> dict:
    """Apply key aliases and drop nested / per-run bookkeeping fields."""
    out = {}
    for key, value in record.items():
        if key.startswith("per_run") or isinstance(value, (dict, list)):
            continue
        out[alias.get(key, key)] = value
    return out


def load_dp_records(dataset_dir: str, model_dir: str) -> tuple[dict, Path | None, Path | None]:
    """Return {epsilon: merged_record}, plus the JSON and CSV paths used."""
    json_path = find_one(dataset_dir, model_dir, "dp_*_report.json")
    csv_path = find_one(dataset_dir, model_dir, "dp_*results.csv")

    by_epsilon: dict[float, dict] = {}

    if json_path is not None:
        with json_path.open() as f:
            report = json.load(f)
        results = report.get("results") or report.get("epsilon_results") or []
        if isinstance(results, dict):
            results = list(results.values())
        for raw in results:
            record = normalise(raw, DP_ALIAS)
            eps = float(record["epsilon"])
            # normalise() strips every per_run_* key, so pull the run-0 entry out
            # of the raw record first. This is the only source: no dp_*results.csv
            # in the repo carries per-run values, mean/std only.
            per_run = raw.get("per_run_accuracies")
            if isinstance(per_run, list) and per_run:
                record["exported_model_accuracy"] = float(per_run[0])
            by_epsilon[eps] = record

    if csv_path is not None:
        for raw in read_csv_rows(csv_path):
            record = {DP_ALIAS.get(k, k): to_number(v) for k, v in raw.items()}
            eps = float(record["epsilon"])
            # CSV is the wider source, so it wins on any shared key
            by_epsilon.setdefault(eps, {}).update(
                {k: v for k, v in record.items() if v is not None}
            )

    return by_epsilon, json_path, csv_path


def load_measured_exported() -> dict[tuple[str, str, float], float]:
    """{(dataset, model, epsilon): accuracy} measured from the exported artifact.

    Consulted **only** where a ``dp_*_report.json`` carries no
    ``per_run_accuracies``; the report always wins where it has a value. The
    exported file is itself the run-0 draw, so scoring it measures the same
    object the attacks target rather than estimating it -- and on the 18
    configurations where both sources exist the two agree exactly. See
    ``measure_exported_accuracy.py``, which regenerates the input CSV.
    """
    if not MEASURED_EXPORTED_FILE.exists():
        return {}
    table: dict[tuple[str, str, float], float] = {}
    for row in read_csv_rows(MEASURED_EXPORTED_FILE):
        accuracy = row.get("exported_model_accuracy")
        if not accuracy:
            continue
        table[(row["dataset"], row["model"], float(row["epsilon"]))] = float(accuracy)
    return table


def load_baseline(dataset_dir: str, model_dir: str) -> tuple[dict, Path | None, Path | None]:
    """Return the merged standard-model record, plus the JSON and CSV paths."""
    json_path = find_one(dataset_dir, model_dir, "std_*_report.json")
    csv_path = find_one(dataset_dir, model_dir, "baseline*results.csv")

    record: dict = {}

    if json_path is not None:
        with json_path.open() as f:
            report = json.load(f)
        flat = {k: v for k, v in report.items() if isinstance(v, (int, float, str))}
        flat.update(report.get("metrics", {}))
        record.update(normalise(flat, BASELINE_ALIAS))

    if csv_path is not None:
        rows = read_csv_rows(csv_path)
        if rows:
            flat = {BASELINE_ALIAS.get(k, k): to_number(v) for k, v in rows[0].items()}
            record.update({k: v for k, v in flat.items() if v is not None})

    return record, json_path, csv_path


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def blank_row() -> dict:
    return {column: "" for column in COLUMNS}


def build_rows() -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    warnings: list[str] = []
    measured_exported = load_measured_exported()

    for dataset_name, info in DATASETS.items():
        dataset_dir = info["dir"]

        for model_name, model_dir in MODELS.items():
            baseline, base_json, base_csv = load_baseline(dataset_dir, model_dir)
            dp_records, dp_json, dp_csv = load_dp_records(dataset_dir, model_dir)

            if not baseline:
                warnings.append(f"{dataset_name}/{model_name}: no baseline found")
            if not dp_records:
                warnings.append(f"{dataset_name}/{model_name}: no DP results found")
                continue
            if dp_json is None:
                warnings.append(
                    f"{dataset_name}/{model_name}: no dp report JSON; using CSV only"
                )

            baseline_accuracy = baseline.get("accuracy_mean")
            if baseline_accuracy in (None, "", 0):
                warnings.append(
                    f"{dataset_name}/{model_name}: baseline accuracy unavailable; "
                    "ACL left blank"
                )
                baseline_accuracy = None

            identity = {
                "dataset": dataset_name,
                "dataset_dir": dataset_dir,
                "n_samples": info["n_samples"],
                "n_features": info["n_features"],
                "model": model_name,
                "model_family": model_dir,
                "source_dp": rel(dp_csv or dp_json),
                "source_baseline": rel(base_csv or base_json),
            }

            # --- baseline row -------------------------------------------------
            if baseline:
                row = blank_row()
                row.update(identity)
                row["variant"] = "standard"
                row["epsilon"] = ""
                row["n_runs"] = baseline.get("n_runs", "")
                for column in METRIC_COLUMNS:
                    if column in baseline and baseline[column] is not None:
                        row[column] = baseline[column]
                row["ACL"] = 0.0
                row["accuracy_loss_pct"] = 0.0
                rows.append(row)

            # --- DP rows ------------------------------------------------------
            for epsilon in sorted(dp_records):
                record = dp_records[epsilon]
                row = blank_row()
                row.update(identity)
                row["variant"] = "dp"
                row["epsilon"] = epsilon
                row["n_runs"] = record.get("n_runs", "")

                for column in METRIC_COLUMNS:
                    if column in record and record[column] is not None:
                        row[column] = record[column]

                dp_accuracy = record.get("accuracy_mean")
                if baseline_accuracy and dp_accuracy is not None:
                    acl = 1.0 - float(dp_accuracy) / float(baseline_accuracy)
                    row["ACL"] = acl
                    row["accuracy_loss_pct"] = acl * 100.0
                else:
                    row["ACL"] = ""
                    row["accuracy_loss_pct"] = ""

                # Same relative form as ACL above, but for the single exported
                # artifact rather than the N_RUNS mean. Prefer the value logged
                # at run time; where the report predates that logging, fall back
                # to scoring the exported artifact itself. Still left blank (not
                # zero, not estimated) if neither source has it.
                exported_accuracy = record.get("exported_model_accuracy")
                if exported_accuracy is not None:
                    row["exported_model_accuracy_source"] = (
                        "report JSON (per_run_accuracies[0])"
                    )
                else:
                    exported_accuracy = measured_exported.get(
                        (dataset_name, model_name, epsilon)
                    )
                    if exported_accuracy is not None:
                        row["exported_model_accuracy"] = exported_accuracy
                        row["exported_model_accuracy_source"] = (
                            "measured from exported artifact"
                        )

                if baseline_accuracy and exported_accuracy is not None:
                    row["ACL_exported"] = (
                        float(baseline_accuracy) - float(exported_accuracy)
                    ) / float(baseline_accuracy)
                else:
                    row["ACL_exported"] = ""

                rows.append(row)

    return rows, warnings


def main() -> None:
    rows, warnings = build_rows()

    with OUTPUT_FILE.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    n_baseline = sum(1 for r in rows if r["variant"] == "standard")
    n_dp = sum(1 for r in rows if r["variant"] == "dp")

    print(f"Wrote {OUTPUT_FILE.relative_to(BASE_DIR)}")
    print(f"  {len(rows)} rows ({n_baseline} baseline, {n_dp} DP)")
    print(f"  {len(COLUMNS)} columns")
    print(f"  {len(DATASETS)} datasets x {len(MODELS)} models")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")


if __name__ == "__main__":
    main()
