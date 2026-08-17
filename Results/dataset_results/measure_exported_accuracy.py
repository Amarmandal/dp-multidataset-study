"""
Measure the test accuracy of each exported DP-DNN artifact.

Why this exists
---------------
``exported_model_accuracy`` in ``consolidated_data.csv`` is the accuracy of the
single exported run-0 draw -- the artifact the membership inference attacks
actually target. Its normal source is ``per_run_accuracies[0]`` in each
``dp_*_report.json``. Four ``dp_dnn_report.json`` files (BCP, CANCER_RISK,
DIABETES, GALLSTONE) predate the notebook line that writes that key, so the
column was blank on 36 of the 414 DP rows.

The value is nonetheless recoverable **without retraining**, because the
artifact itself is on disk: ``dp_dnn_model_eps_<eps>.pt`` is the run-0 model,
exported inside ``if run == 0:`` in the same loop iteration that appended to
``per_run_accuracies``. Loading it and scoring it on the same test split is a
*measurement of the same object*, not a reconstruction or an estimate.

Author confirmation
-------------------
``recorded_run0_accuracy`` is a boolean and is ``True`` on **every** row. It
asserts the provenance claim -- *this number is the run-0 draw's accuracy* --
which the authors confirmed directly (2026-08-16) and which the code bears out:
every ``DP_*.ipynb`` calls ``export_model()`` inside ``if run == 0:``, so the
artifact on disk is the run-0 model for all six datasets. Whether a particular
run's report JSON also *logged* that number is a separate question, and a
missing log entry is not evidence against the property. The recorded float, where
one exists, is kept in ``report_json_run0_accuracy`` as corroborating evidence.

Self-validating
---------------
KIDNEY_STONE and LUNG_CANCER do record ``per_run_accuracies``, so this script
measures all six datasets and checks its own output against the recorded value
wherever one exists. Those 18 configurations are the control: if any of them
disagrees, the measurement procedure is wrong and the script exits non-zero
rather than writing a CSV.

Evaluation matches the notebooks exactly (``DP_DNN.ipynb``, ``evaluate_model``):
``model.eval()``, forward pass under ``no_grad``, ``argmax`` over logits,
``sklearn.metrics.accuracy_score`` against ``y_test`` from
``<DATASET>/data/processed_data.pkl``.

Usage:
    python Results/dataset_results/measure_exported_accuracy.py

Output:
    Results/dataset_results/exported_model_accuracy_measured.csv
"""

import csv
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_FILE = Path(__file__).resolve().parent / "exported_model_accuracy_measured.csv"

# ``exported_models`` redefines DNNClassifier / DPDNNClassifier and registers
# them into __main__ so the notebook-pickled objects resolve. It is the repo's
# single source of truth for the exported-target layout -- reuse it rather than
# keeping a second copy of those class definitions here.
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "Attack" / "MIA_Shokri"))
import exported_models  # noqa: E402  (import also registers the classes)

DATASETS = {
    "Breast Cancer": "BCP",
    "Cancer Risk": "CANCER_RISK",
    "Diabetes": "DIABETES",
    "Gallstone": "GALLSTONE",
    "Kidney Stone": "KIDNEY_STONE",
    "Lung Cancer": "LUNG_CANCER",
}

# The DP-DNN sweep grid (the attack grid), not the 15-value utility grid: the
# DNN notebooks sweep these nine budgets only.
EPSILONS = [0.1, 0.2, 0.4, 0.8, 1.0, 2.0, 4.0, 8.0, 10.0]

COLUMNS = [
    "dataset",
    "dataset_dir",
    "model",
    "epsilon",
    "exported_model_accuracy",
    # Boolean, True on every row: this value IS the run-0 draw's accuracy.
    # Author-confirmed 2026-08-16 -- every DP_*.ipynb exports inside
    # `if run == 0:`, so the artifact on disk is the run-0 model for all six
    # datasets, whether or not that run's report happened to log the number.
    # The flag is the provenance claim; the two columns below are the evidence.
    "recorded_run0_accuracy",
    # The report JSON's own per_run_accuracies[0], where it recorded one. Blank
    # on the 36 recovered cells -- absence of a log entry, not absence of the
    # property the flag asserts. Retained so the exact-match check stays
    # reproducible from the CSV alone.
    "report_json_run0_accuracy",
    "agreement",
    "n_test",
    "model_file",
    "measured_utc",
]

TOLERANCE = 1e-9


def load_test_split(dataset_dir: str) -> tuple[torch.Tensor, np.ndarray]:
    with (BASE_DIR / dataset_dir / "data" / "processed_data.pkl").open("rb") as f:
        data = pickle.load(f)
    return torch.FloatTensor(data["X_test"]), np.asarray(data["y_test"])


def recorded_run0(dataset_dir: str) -> dict[float, float]:
    """{epsilon: per_run_accuracies[0]} from the report JSON, where present."""
    path = BASE_DIR / dataset_dir / "DNN" / "output" / "dp_dnn_report.json"
    if not path.exists():
        return {}
    with path.open() as f:
        report = json.load(f)
    results = report.get("results") or report.get("epsilon_results") or []
    if isinstance(results, dict):
        results = list(results.values())
    out = {}
    for record in results:
        per_run = record.get("per_run_accuracies")
        if isinstance(per_run, list) and per_run:
            out[float(record["epsilon"])] = float(per_run[0])
    return out


def measure(X_test: torch.Tensor, y_test: np.ndarray, model_path: Path) -> float:
    """Score one exported artifact exactly as DP_DNN.ipynb's evaluate_model does."""
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
    preds = torch.max(logits, 1)[1].cpu().numpy()
    return float(accuracy_score(y_test, preds))


def main() -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows: list[dict] = []
    mismatches: list[str] = []
    missing: list[str] = []

    for dataset_name, dataset_dir in DATASETS.items():
        X_test, y_test = load_test_split(dataset_dir)
        known = recorded_run0(dataset_dir)

        for epsilon in EPSILONS:
            # Resolve via the attack loader's own path rule rather than a local
            # copy of it: the older exports (BCP, CANCER_RISK, DIABETES) carry a
            # .pkl suffix and the newer ones .pt, and model_path() already
            # falls back across the two.
            model_path = Path(
                exported_models.model_path(
                    str(BASE_DIR), dataset_dir, "DNN", "dp", epsilon
                )
            )
            if not model_path.exists():
                missing.append(f"{dataset_name} eps={epsilon}: {model_path} not found")
                continue

            accuracy = measure(X_test, y_test, model_path)
            reference = known.get(epsilon)

            if reference is None:
                agreement = (
                    "recovered cell; run-0 provenance author-confirmed, "
                    "no report JSON value to cross-check against"
                )
            elif abs(reference - accuracy) <= TOLERANCE:
                agreement = "matches recorded per_run_accuracies[0]"
            else:
                agreement = f"MISMATCH (delta={accuracy - reference:+.3e})"
                mismatches.append(
                    f"{dataset_name} eps={epsilon}: "
                    f"recorded {reference:.6f} vs measured {accuracy:.6f}"
                )

            rows.append(
                {
                    "dataset": dataset_name,
                    "dataset_dir": dataset_dir,
                    "model": "DP-DNN",
                    "epsilon": epsilon,
                    "exported_model_accuracy": accuracy,
                    "recorded_run0_accuracy": True,
                    "report_json_run0_accuracy": (
                        "" if reference is None else reference
                    ),
                    "agreement": agreement,
                    "n_test": len(y_test),
                    "model_file": str(model_path.relative_to(BASE_DIR)),
                    "measured_utc": stamp,
                }
            )

    n_control = sum(1 for r in rows if r["report_json_run0_accuracy"] != "")
    n_recovered = len(rows) - n_control

    if mismatches:
        print("VALIDATION FAILED -- measurement does not reproduce recorded run-0:")
        for line in mismatches:
            print(f"  - {line}")
        print("\nNo CSV written.")
        sys.exit(1)

    with OUTPUT_FILE.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {OUTPUT_FILE.relative_to(BASE_DIR)}")
    print(f"  {len(rows)} exported DP-DNN artifacts measured")
    print(f"  {n_control} control cells reproduced recorded run-0 exactly (<= {TOLERANCE:g})")
    print(f"  {n_recovered} cells recovered where no run-0 was recorded")

    if missing:
        print("\nMissing artifacts:")
        for line in missing:
            print(f"  - {line}")


if __name__ == "__main__":
    main()
