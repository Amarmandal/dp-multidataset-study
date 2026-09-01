"""Emit analysis/MANIFEST.csv -- one row per generated artifact.

Run last, after tables, figures and logs exist.  Every row is checked against
the filesystem: a declared artifact that is not on disk is an error.
"""

from __future__ import annotations

import csv

import config as C
import loaders as L

COLUMNS = ["artifact_path", "type", "referee_items", "manuscript_location",
           "source_files", "generated_by", "notes"]

UTIL = "Results/dataset_results/consolidated_data.csv"
MIA = "Results/attack_results/consolidated_mia_data.csv"
ATTACKS = ("Attack/LiRA/results/<DS>/<DS>_lira_results.csv; "
           "Attack/MIA_Shokri/results/<DS>/<DS>_results.csv; "
           "Attack/MIA_YEOM/results/<DS>/<DS>_mia_results.csv")

# (stem, referee items, manuscript location, sources, notes)
TABLES = [
    ("dataset_characteristics", "[48][49][50]", "Section 3.1 (Datasets)",
     f"<DS>/data/processed_data.pkl; {UTIL}; analysis/stats/correlation_stats/majority_class_accuracy.csv",
     "Real/synthetic is [MISSING] for all six datasets: not recorded in the "
     "repository and not derivable from the source URL. URLs reproduced verbatim."),
    ("baseline_accuracy_matrix", "[11][51]", "Section 4.1 (Non-private baselines)",
     f"{UTIL}; <DS>/<FAMILY>/**/std_*_report.json; Attack/MIA_Shokri/results/<DS>/<DS>_results.csv",
     "Carries both the 30-run mean accuracy and the exported run-0 artefact "
     "accuracy the attacks target; they differ by at most 0.0099. Neither is "
     "chosen for the reader. See gaps.md 4.1."),
    ("baseline_leakage_all_pairs", "[10]", "Section 4.3 (Leakage, non-private targets)",
     f"{MIA}; {ATTACKS}",
     "All 30 dataset x family pairs, all three attacks, each metric with its SD."),
    ("evaluation_set_sizes", "[8]", "Section 3.4 (Attack protocol)",
     f"{ATTACKS}; <DS>/data/processed_data.pkl",
     "Sets are 4:1 imbalanced, not the balanced 1:1 design requested; "
     "balanced == False on all 300 LiRA rows and cannot be fixed post hoc. "
     "Gallstone's minimum resolvable FPR (0.0156) exceeds 1%. See gaps.md 1.1."),
    ("residual_leakage_eps1", "[14][15]", "Section 4.4 (Residual leakage under DP)",
     f"{MIA}; {ATTACKS}",
     "DP targets at eps=1.0. The LiRA 95% CI bootstraps saved seed-level "
     "integer outcomes; repeated evaluations of the same records are not pooled."),
    ("rho_by_epsilon", "[19]", "Section 4.5 (Utility-leakage association)",
     "analysis/stats/correlation_stats/rho_by_epsilon.csv",
     "Primary all-pairs rows use artifact-matched ACL_exported at all nine "
     "budgets. P-values exactly permute the six dataset blocks; 95% intervals "
     "bootstrap whole datasets. Markers are uncorrected; balanced-accuracy "
     "rows are explicitly labeled sensitivity rows."),
    ("within_pair_correlations", "[18]", "Section 4.5 (Utility-leakage association)",
     "analysis/stats/within_pair/summary.csv; analysis/stats/within_pair/pair_correlations.csv",
     "Uses artifact-matched ACL_exported across nine budgets inside each pair. "
     "One Shokri pair is undefined. Significance counts are uncorrected."),
    ("bound_violations", "[3][33]", "Section 5 (Discussion) / [TBD]",
     f"{MIA}; Attack/LiRA/results/<DS>/<DS>_lira_results.csv",
     "Descriptive screening only; no confidence or guarantee-exceedance claim "
     "is computed. See the table note and gaps.md 4.2."),
    ("config_inventory", "[9][59][61]", "Appendix A (Reproducibility) / [TBD]",
     f"<DS>/<FAMILY>/**/std_*_report.json; <DS>/<FAMILY>/**/dp_*_report.json; {UTIL}",
     "All software-version fields are [MISSING]: not recorded in any report and "
     "deliberately not read from the current machine. Hyperparameters missing "
     "on 11 non-private and 7 DP configs. See gaps.md 2."),
]

FIGURES = [
    ("gap_vs_leakage", "[51][65]", "Section 4.3 (Memorisation and leakage)",
     "analysis/tables/csv/baseline_accuracy_matrix.csv; "
     "analysis/tables/csv/baseline_leakage_all_pairs.csv",
     "Two panels: train-test gap vs baseline LiRA TPR@1% and vs baseline LiRA "
     "AUC. Spearman rho, n, p annotated on each axis. No in-image title."),
    ("gap_vs_leakage_vs_N", "[51][65]", "Section 4.3 (Memorisation and leakage)",
     "analysis/tables/csv/baseline_accuracy_matrix.csv; "
     "analysis/tables/csv/baseline_leakage_all_pairs.csv; " + UTIL,
     "Places the gap-based relationship beside the existing N-based one on a "
     "shared y axis, for direct comparison."),
    ("residual_floor_ci", "[14][15]", "Section 4.4 (Residual leakage under DP)",
     "analysis/tables/csv/residual_leakage_eps1.csv; "
     "analysis/tables/csv/baseline_leakage_all_pairs.csv",
     "RQ2b common-floor figure with run-level bootstrap error bars on every point. "
     "Floor estimator (median, with IQR) stated in the axis annotation."),
    ("rq1c_utility_vs_protection_exported", "[72][73][75]", "Section 4.5 (RQ1c)",
     "analysis/stats/rq/figure_input.csv; analysis/stats/rq/correlations.csv",
     "Uses ACL_exported, which matches the artefact the attacks target. "
     "Dataset-block permutation p-value and dataset-cluster bootstrap interval."),
    ("rq2a_benefit_vs_baseline", "[72][73][75]", "Section 4.5 (RQ2a)",
     "analysis/stats/rq/figure_input.csv; analysis/stats/rq/correlations.csv",
     "Regenerated from the RQ figure input; cluster-aware recorded inference asserted."),
    ("l4_avg_vs_worst_case", "[72][73][75]", "Section 4.3 (Attack agreement)",
     "analysis/stats/rq/figure_input.csv; analysis/stats/rq/correlations.csv",
     "Agreement between the two average-case attack summaries, Yeom advantage "
     "and LiRA AUC, on the non-private targets. Cluster-aware inference asserted."),
    ("l6_baseline_leakage_vs_N", "[72][73][75]", "Section 4.3 (Dataset size)",
     "analysis/stats/rq/figure_input.csv; analysis/stats/rq/correlations.csv",
     "Non-private RF low-FPR leakage against N. Six points; exact permutation "
     "p-value and observation-bootstrap interval asserted."),
]

UPLOAD_TABLES = [
    ("table_11_artifact_matched_acl_exported_epsilon_1", "Table 11",
     "analysis/stats/correlation_stats/rho_by_epsilon.csv",
     "Fixed-budget correlations with dataset-cluster-aware p-values and intervals."),
    ("table_12_artifact_matched_acl_exported_all_budgets", "Table 12",
     "analysis/stats/correlation_stats/rho_by_epsilon.csv",
     "Nine-budget rho matrix; stars use exact dataset-block permutation p-values."),
    ("table_13_artifact_matched_acl_exported_within_pair", "Table 13",
     "analysis/stats/within_pair/summary.csv",
     "Within-pair nine-budget correlation summary."),
]


def rows() -> list[dict]:
    out = []
    for stem, items, loc, src, note in TABLES:
        out.append({
            "artifact_path": f"analysis/tables/csv/{stem}.csv",
            "type": "table (CSV)", "referee_items": items,
            "manuscript_location": loc, "source_files": src,
            "generated_by": "analysis/src/build_tables.py",
            "notes": note})

    for stem, table_name, src, note in UPLOAD_TABLES:
        out.append({
            "artifact_path": f"analysis/tables/csv/{stem}.csv",
            "type": "upload-ready manuscript table (CSV)",
            "referee_items": "[5]",
            "manuscript_location": table_name,
            "source_files": src,
            "generated_by": "analysis/src/build_tables.py",
            "notes": note})

    for stem, items, loc, src, note in FIGURES:
        for ext, kind in (("pdf", "figure (PDF, vector)"), ("png", "figure (PNG, 300 dpi)")):
            out.append({
                "artifact_path": f"analysis/figures/{ext}/{stem}.{ext}",
                "type": kind, "referee_items": items,
                "manuscript_location": loc, "source_files": src,
                "generated_by": "analysis/src/build_figures.py",
                "notes": note})

    roc_sources = "; ".join(
        L.LIRA_ROC_CSV.format(ds=ds) for ds in L.active_dataset_dirs()
    )
    for ext, kind in (("pdf", "figure (PDF, vector)"),
                      ("png", "figure (PNG, 600 dpi)")):
        out.append({
            "artifact_path": f"analysis/figures/{ext}/loglog_roc_grid.{ext}",
            "type": kind,
            "referee_items": "[62][78]",
            "manuscript_location": "Section 4.3 (LiRA ROC) / Figure 8",
            "source_files": roc_sources,
            "generated_by": "analysis/src/make_roc_grid.py",
            "notes": ("Six standard-target LiRA ROC panels redrawn from persisted "
                      "FPR/TPR coordinates; no attack-side raster is embedded.")})

    src_notes = {
        "config.py": ("Switches: INCLUDE_LUNG_CANCER, UTILITY_LABEL, RANDOM_FPR. "
                      "Also holds paths, the epsilon grids and the [MISSING]/"
                      "[CONFLICT] placeholder strings."),
        "common.py": ("Loads the consolidated inputs and implements the exact "
                      "dataset-block permutation, dataset-cluster bootstrap, "
                      "small-n exact permutation and within-pair helpers."),
        "within_pair_correlations.py": ("Computes artifact-matched correlations "
                                        "across nine budgets inside each pair."),
        "correlation_statistics.py": ("Computes and records all cross-sectional "
                                      "cluster-aware inference and sensitivities."),
        "rq_figures.py": ("Builds the authoritative RQ figure inputs and records "
                          "their cluster-aware correlations."),
        "loaders.py": ("Single source of truth for canonical naming (DP-RF vs RF, "
                       "BCP vs Breast Cancer) and for all input loading. Reads "
                       "processed_data.pkl through a restricted unpickler that "
                       "stubs every non-numpy class, so no estimator is ever "
                       "constructed. Run directly for a coverage report."),
        "build_tables.py": ("Builds nine core tables and three upload-ready "
                            "manuscript tables as CSV files."),
        "build_figures.py": "Builds the eight generated analysis figures.",
        "extract_provenance.py": ("Parses the report JSONs (globbed -- they sit at "
                                  "inconsistent depths) into config_inventory."),
        "verify.py": ("Re-derives every numeric table cell independently and writes "
                      "logs/verification.txt. Exits non-zero on any failure."),
        "make_manifest.py": "Emits this file.",
    }
    for name, note in src_notes.items():
        out.append({
            "artifact_path": f"analysis/src/{name}", "type": "source",
            "referee_items": "n/a", "manuscript_location": "n/a",
            "source_files": "n/a", "generated_by": "hand-written", "notes": note})

    out.append({
        "artifact_path": "analysis/logs/verification.txt", "type": "log",
        "referee_items": "n/a", "manuscript_location": "n/a",
        "source_files": "analysis/tables/csv/*.csv and their primary sources",
        "generated_by": "analysis/src/verify.py",
        "notes": ("One line per checked value: table | row | column | value | "
                  "source file | source row/filter. Failures listed at the top.")})
    out.append({
        "artifact_path": "analysis/logs/gaps.md", "type": "documentation",
        "referee_items": "[3][5][6][8][9][22][48][49][50][59][61][62]",
        "manuscript_location": "Response letter",
        "source_files": "analysis/logs/verification.txt; analysis/logs/_figure_gaps.json",
        "generated_by": "hand-written from the pipeline's recorded gaps",
        "notes": ("Everything that could not be produced without compute, with "
                  "reasons. Written to be lifted into the response letter. "
                  "CONFLICTS section: none found.")})
    out.append({
        "artifact_path": "analysis/README.md", "type": "documentation",
        "referee_items": "n/a", "manuscript_location": "n/a",
        "source_files": "n/a", "generated_by": "hand-written",
        "notes": "What the pipeline does and does not do; how to re-run it."})
    out.append({
        "artifact_path": "analysis/RESULTS.md", "type": "documentation",
        "referee_items": "[5]", "manuscript_location": "Response letter / manuscript",
        "source_files": ("analysis/stats/correlation_stats/rho_by_epsilon.csv; "
                         "analysis/stats/rq/correlations.csv"),
        "generated_by": "hand-written from the generated statistics",
        "notes": ("Cluster-aware numerical results, limitations and replacement "
                  "wording for the manuscript.")})
    out.append({
        "artifact_path": "analysis/MANIFEST.csv", "type": "documentation",
        "referee_items": "n/a", "manuscript_location": "n/a",
        "source_files": "n/a", "generated_by": "analysis/src/make_manifest.py",
        "notes": "This file. One row per generated artifact."})
    return out


def main() -> int:
    data = rows()
    missing = [r["artifact_path"] for r in data
               if not (C.REPO / r["artifact_path"]).exists()
               and r["artifact_path"] != "analysis/MANIFEST.csv"
               and r["artifact_path"] != "analysis/README.md"]
    with open(C.ANALYSIS / "MANIFEST.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(data)
    print(f"wrote MANIFEST.csv with {len(data)} rows")
    if missing:
        print(f"WARNING: {len(missing)} declared artifacts are not on disk:")
        for m in missing:
            print("  ", m)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
