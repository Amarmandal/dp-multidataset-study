"""Parse the per-model report JSONs into a flat provenance record.

Feeds ``config_inventory``.  Reads only ``*_report.json`` files found beneath
``<DATASET>/<FAMILY_DIR>/`` (see ``loaders.find_report``) plus the two
consolidated CSVs for timing.  No model is ever loaded.

Anything not present on disk becomes the literal ``[MISSING: ...]`` string.
Where a report JSON carries no ``parameters`` block at all, the hyperparameters
are transcribed from the corresponding notebook's constructor call (read as
source -- no notebook is ever executed); see ``NOTEBOOK_PARAMS``.  The report
JSON always wins where it has a value, and the ``*_source`` columns record
which of the two each cell came from.
Software versions (NumPy, scikit-learn, PyTorch, Opacus, diffprivlib) are NOT
recorded in any per-model report; they are instead taken from the pinned
(``==``) dependencies resolved in this repository's own ``pyproject.toml`` /
``uv.lock`` (see ``SOFTWARE_VERSIONS`` below), on the basis that every
dependency in that file is an exact pin, not a range, so the resolved version
is the version any environment built from this lockfile would install.
Python itself is pinned only as a floor in ``pyproject.toml``
(``requires-python = ">=3.13"``), and ``uv.lock`` resolves against both cp313
and cp314 wheel sets without recording which interpreter actually ran;
``version_python`` is recorded as 3.13 on the author's direct confirmation of
the interpreter used, not derived from the lockfile alone.
"""

from __future__ import annotations

import json

import pandas as pd

import config as C
import loaders as L

SOFTWARE_FIELDS = ["python", "numpy", "scikit_learn", "pytorch", "opacus", "diffprivlib"]

# Exact pins from pyproject.toml / uv.lock (checked 2026-08-13). The five
# library versions are ``==`` pins resolved in the lockfile, not ranges.
# `requires-python = ">=3.13"` is a floor only -- uv.lock resolves against
# both cp313 and cp314 wheel sets and does not itself record which
# interpreter was invoked -- but the author has confirmed 3.13 as the actual
# interpreter used, so it is recorded here rather than left MISSING.
SOFTWARE_VERSIONS = {
    "python": "3.13",
    "numpy": "2.4.6",
    "scikit_learn": "1.6.1",
    "pytorch": "2.11.0",
    "opacus": "1.6.0",
    "diffprivlib": "0.6.6",
}

# --------------------------------------------------------------------------
# Notebook-sourced hyperparameters (fallback only)
# --------------------------------------------------------------------------
# 18 of the 60 hyperparameter cells have no ``parameters`` block in their
# report JSON.  The values below are transcribed from the constructor call in
# the corresponding notebook, read as source -- no notebook is executed.  They
# are used ONLY where the report JSON carries nothing, never to override a
# recorded value, and every row that uses one is marked in the
# ``*_hyperparameters_source`` column so a referee can tell the two apart.
#
# Each entry is (params_string, notebook_path).  Where a constructor was called
# bare or relied on a library default, that is stated as a default rather than
# presented as an explicit choice.
#
# Confidence check: GALLSTONE DP-DNN's recorded batch size (31) is reproduced
# exactly by the notebook's own rule min(256, max(16, n_train // 8)) with
# n_train = 255, so the notebooks and the recorded runs agree where they overlap.
_GNB_BARE = ("model=GaussianNB; constructor called with no arguments "
             "(scikit-learn defaults: priors=None, var_smoothing=1e-09)")

NOTEBOOK_PARAMS: dict[tuple[str, str, str], tuple[str, str]] = {
    # ---- non-private (11) ------------------------------------------------
    ("GALLSTONE", "RF", "std"): (
        "n_estimators=20; max_depth=4; random_state=42; n_jobs=-1",
        "GALLSTONE/RandomForest/STD_RF.ipynb"),
    ("GALLSTONE", "LR", "std"): (
        "max_iter=1000; random_state=42; solver=lbfgs (scikit-learn default)",
        "GALLSTONE/LR/STD_LR.ipynb"),
    ("GALLSTONE", "GNB", "std"): (_GNB_BARE, "GALLSTONE/GaussianNB/STD_GNB.ipynb"),
    ("GALLSTONE", "SVM", "std"): (
        "C=1.0; kernel=rbf; gamma=scale; random_state=42; probability=True",
        "GALLSTONE/SVM/STD_SVM.ipynb"),
    ("GALLSTONE", "DNN", "std"): (
        "architecture=DNN; hidden_sizes=[64,32]; input_size=38; num_classes=2; "
        "dropout=0.3; learning_rate=0.001; optimizer=Adam; num_epochs=50; "
        "batch_size=32; n_runs=30",
        "GALLSTONE/DNN/STD_DNN.ipynb"),
    ("LUNG_CANCER", "RF", "std"): (
        "n_estimators=20; max_depth=4; random_state=42; n_jobs=-1",
        "LUNG_CANCER/RandomForest/STD_RF.ipynb"),
    ("LUNG_CANCER", "LR", "std"): (
        "max_iter=1000; random_state=42; solver=lbfgs; multi_class=multinomial",
        "LUNG_CANCER/LR/STD_LR.ipynb"),
    ("LUNG_CANCER", "GNB", "std"): (_GNB_BARE, "LUNG_CANCER/GaussianNB/STD_GNB.ipynb"),
    ("LUNG_CANCER", "SVM", "std"): (
        "C=1.0; kernel=rbf; gamma=scale; random_state=42; probability=True",
        "LUNG_CANCER/SVM/STD_SVM.ipynb"),
    ("CANCER_RISK", "GNB", "std"): (
        _GNB_BARE + "; n_runs=30", "CANCER_RISK/GaussianNB/STD_GNB.ipynb"),
    ("KIDNEY_STONE", "GNB", "std"): (
        _GNB_BARE + "; n_runs=30", "KIDNEY_STONE/GaussianNB/STD_GNB.ipynb"),

    # ---- DP (7) ----------------------------------------------------------
    ("GALLSTONE", "RF", "dp"): (
        "n_estimators=20; max_depth=4; bounds=per-feature, from "
        "processed_data.pkl; random_state=run*10+42; n_jobs=-1; n_runs=30",
        "GALLSTONE/RandomForest/DP_RF.ipynb"),
    ("GALLSTONE", "LR", "dp"): (
        "max_iter=1000; data_norm=1.0; random_state=run*10+42; n_runs=30",
        "GALLSTONE/LR/DP_LR.ipynb"),
    ("GALLSTONE", "GNB", "dp"): (
        "bounds=per-feature, from processed_data.pkl; model=DP-GaussianNB; "
        "random_state not set (diffprivlib default); n_runs=30",
        "GALLSTONE/GaussianNB/DP_GNB.ipynb"),
    ("GALLSTONE", "SVM", "dp"): (
        "Lambda=0.01; h=0.5; normalize=True; fit_intercept=False; "
        "random_state=run*10+42; method=Objective Perturbation (Chaudhuri et al. "
        "2011, Algorithm 2); model=DP-SVM with Huber Loss; n_runs=30",
        "GALLSTONE/SVM/DP_SVM.ipynb"),
    ("KIDNEY_STONE", "RF", "dp"): (
        "n_estimators=20; max_depth=4; bounds=per-feature, from "
        "processed_data.pkl; random_state=run*10+42; n_jobs=-1; n_runs=30",
        "KIDNEY_STONE/RandomForest/DP_RF.ipynb"),
    ("KIDNEY_STONE", "LR", "dp"): (
        "max_iter=1000; data_norm=1.0; random_state=run*10+42; n_runs=30",
        "KIDNEY_STONE/LR/DP_LR.ipynb"),
    ("KIDNEY_STONE", "GNB", "dp"): (
        "bounds=(bounds[0], bounds[1]) from processed_data.pkl; "
        "model=DP-GaussianNB; random_state not set (diffprivlib default; "
        "np.random.seed(run*10+42) set by the notebook); n_runs=30",
        "KIDNEY_STONE/GaussianNB/DP_GNB.ipynb"),
}

# Notebook-sourced DNN batch sizes, same rules as NOTEBOOK_PARAMS above.
# The DP figures are the notebook's own min(256, max(16, n_train // 8)) rule
# evaluated at the recorded n_train (GALLSTONE 255 -> 31, LUNG_CANCER 40000 -> 256).
NOTEBOOK_BATCH: dict[tuple[str, str], tuple[str, str]] = {
    ("GALLSTONE", "STD"): ("32", "GALLSTONE/DNN/STD_DNN.ipynb"),
    ("LUNG_CANCER", "STD"): ("256", "LUNG_CANCER/DNN/STD_DNN.ipynb"),
    ("LUNG_CANCER", "DP"): ("256", "LUNG_CANCER/DNN/DP_DNN.ipynb"),
}

# Hyperparameter keys worth surfacing, in a stable display order.
_HP_ORDER = [
    "n_estimators", "max_depth", "criterion",
    "max_iter", "solver", "C", "kernel", "gamma", "data_norm",
    "var_smoothing", "priors", "bounds",
    "architecture", "hidden_sizes", "input_size", "input_features",
    "num_classes", "dropout", "lr", "learning_rate", "optimizer",
    "epochs", "num_epochs", "batch_size",
    "Lambda", "h", "method", "model",
    "max_grad_norm", "delta", "random_state", "probability", "n_jobs", "n_runs",
]


def _collect_params(report: dict | None) -> dict:
    """Pull the parameter dict out of a report, wherever the notebook put it.

    Layouts seen on disk:
      * top-level ``parameters``            (most STD reports, DP-DNN)
      * ``metrics`` sibling, params at top  (KIDNEY_STONE)
      * ``epsilon_results.<eps>.parameters`` (DP-RF / DP-LR / DP-GNB / DP-SVM)
      * nothing at all                      (GALLSTONE / LUNG_CANCER STD-DNN)
    """
    if report is None:
        return {}
    params: dict = {}

    top = report.get("parameters")
    if isinstance(top, dict):
        params.update(top)

    # Some reports (LUNG_CANCER STD) carry the architecture at the top level.
    for k in ("architecture", "input_features", "n_classes"):
        if k in report and not isinstance(report[k], (dict, list)):
            params.setdefault(k, report[k])
    if isinstance(report.get("architecture"), list):
        params.setdefault("architecture", report["architecture"])

    er = report.get("epsilon_results")
    if isinstance(er, dict) and er:
        # Parameters are identical across epsilons in every file inspected;
        # take the first epsilon that carries them.
        for _eps, blk in sorted(er.items(), key=lambda kv: float(kv[0])):
            if isinstance(blk.get("parameters"), dict):
                params.update(blk["parameters"])
                break

    return params


def _fmt_params(params: dict) -> str:
    if not params:
        return C.MISSING.format(what="hyperparameters not recorded in report JSON")
    keys = [k for k in _HP_ORDER if k in params]
    keys += [k for k in params if k not in keys]
    parts = []
    for k in keys:
        v = params[k]
        if isinstance(v, (list, dict)):
            v = json.dumps(v, separators=(",", ":"))
        parts.append(f"{k}={v}")
    return "; ".join(parts)


def _batch_size(std_params: dict, dp_params: dict, family: str,
                ds: str) -> tuple[str, str]:
    """Return (batch_size_string, source_string).

    Falls back to ``NOTEBOOK_BATCH`` only where the report JSON records no
    batch size; the source string names the notebook whenever it does.
    """
    if family != "DNN":
        return "n/a (not a minibatch learner)", "n/a"
    bits, srcs = [], []
    for label, params in (("STD", std_params), ("DP", dp_params)):
        bs = params.get("batch_size")
        if bs is not None:
            bits.append(f"{label}={bs}")
            srcs.append(f"{label}=report JSON")
            continue
        nb = NOTEBOOK_BATCH.get((ds, label))
        if nb is not None:
            bits.append(f"{label}={nb[0]}")
            srcs.append(f"{label}=notebook source ({nb[1]})")
        else:
            bits.append(f"{label}=" + C.MISSING.format(what="batch_size"))
            srcs.append(f"{label}=" + C.MISSING.format(what="batch_size source"))
    return "; ".join(bits), "; ".join(srcs)


def _params_with_fallback(report_params: dict, ds: str, fam: str,
                          variant: str) -> tuple[str, str]:
    """Format hyperparameters, falling back to the notebook source.

    The report JSON always wins.  ``NOTEBOOK_PARAMS`` is consulted only when
    the report carries no ``parameters`` block at all, and the returned source
    string records which of the two was used.
    """
    if report_params:
        return _fmt_params(report_params), "report JSON"
    nb = NOTEBOOK_PARAMS.get((ds, fam, variant))
    if nb is not None:
        return nb[0], f"notebook source ({nb[1]})"
    return (C.MISSING.format(what="hyperparameters not recorded in report JSON"),
            C.MISSING.format(what="hyperparameter source"))


def _privacy_note(dp_report: dict | None, family: str) -> str:
    """Mechanism string, preferring what the report itself declares."""
    declared = None
    if dp_report is not None:
        declared = dp_report.get("privacy_mechanism")
    base = L.DP_MECHANISM[family]
    if declared:
        return f"{base} [report: {declared}]"
    return base


def build_provenance() -> pd.DataFrame:
    u = L.load_utility()
    std_u = u[u["variant"] == "standard"]
    dp_u = u[(u["variant"] == "dp") & (u["epsilon"] == C.EPS_TARGET)]

    rows = []
    for ds in L.DATASET_DIRS:
        disp = L.DIR_TO_DISPLAY[ds]
        for fam in L.FAMILIES:
            std_r = L.read_report(ds, fam, "std")
            dp_r = L.read_report(ds, fam, "dp")
            std_p = _collect_params(std_r)
            dp_p = _collect_params(dp_r)

            su = std_u[(std_u["dataset"] == disp)
                       & (std_u["model"] == L.SHORT_TO_DP[fam])]
            du = dp_u[(dp_u["dataset"] == disp)
                      & (dp_u["model"] == L.SHORT_TO_DP[fam])]

            def _num(frame, col, what):
                if len(frame) == 1 and pd.notna(frame.iloc[0][col]):
                    return float(frame.iloc[0][col])
                return C.MISSING.format(what=what)

            std_hp, std_hp_src = _params_with_fallback(std_p, ds, fam, "std")
            dp_hp, dp_hp_src = _params_with_fallback(dp_p, ds, fam, "dp")
            bs, bs_src = _batch_size(std_p, dp_p, fam, ds)

            rows.append({
                "dataset": disp,
                "dataset_dir": ds,
                "model": L.SHORT_TO_DP[fam],
                "family": fam,
                "nonprivate_hyperparameters": std_hp,
                "nonprivate_hyperparameters_source": std_hp_src,
                "dp_hyperparameters": dp_hp,
                "dp_hyperparameters_source": dp_hp_src,
                "dp_mechanism": _privacy_note(dp_r, fam),
                "batch_size": bs,
                "batch_size_source": bs_src,
                "std_train_time_s": _num(su, "train_time_mean", "std train_time"),
                "train_time_mean": _num(du, "train_time_mean",
                                        f"DP train_time_mean at eps={C.EPS_TARGET}"),
                "train_time_std": _num(du, "train_time_std",
                                       f"DP train_time_std at eps={C.EPS_TARGET}"),
                **{f"version_{k}": (SOFTWARE_VERSIONS[k]
                                     if SOFTWARE_VERSIONS[k] is not None
                                     else C.MISSING.format(what="software version"))
                   for k in SOFTWARE_FIELDS},
                "source_std_report": (std_r or {}).get(
                    "__path__", C.MISSING.format(what="std report JSON")),
                "source_dp_report": (dp_r or {}).get(
                    "__path__", C.MISSING.format(what="dp report JSON")),
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build_provenance()
    pd.set_option("display.width", 200)
    print(df[["dataset", "model", "batch_size", "std_train_time_s",
              "train_time_mean", "train_time_std"]].to_string(index=False))
    for col in ("nonprivate_hyperparameters", "dp_hyperparameters"):
        miss = int(df[col].str.startswith("[MISSING").sum())
        nb = int(df[f"{col}_source"].str.startswith("notebook").sum())
        print(f"\n{col}: {miss} missing, {nb} from notebook source, "
              f"{len(df) - miss - nb} from report JSON  (rows={len(df)})")
