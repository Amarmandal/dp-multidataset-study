"""Emit revision/MANIFEST.csv -- one row per generated artifact.

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
     f"<DS>/data/processed_data.pkl; {UTIL}; analysis/d15/d15_majority_class_accuracy.csv",
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
     "DP targets at eps=1.0. Clopper-Pearson exact 95% CI on the LiRA TPR@1%, "
     "n = n_members, k = round(TPR * n_members)."),
    ("rho_by_epsilon", "[19]", "Section 4.5 (Utility-leakage association)",
     "analysis/d15/d15_rho_by_epsilon.csv",
     "Reformat only; nothing recomputed. Significance markers are uncorrected "
     "for multiple comparisons (108 tests). Exact p-values in the CSV."),
    ("within_pair_correlations", "[18]", "Section 4.5 (Utility-leakage association)",
     "analysis/d14/d14_summary.csv; analysis/d14/d14_pair_correlations.csv",
     "Reformat only; nothing recomputed. One Shokri pair is undefined. "
     "Significance counts uncorrected."),
    ("bound_violations", "[3][33]", "Section 5 (Discussion) / [TBD]",
     f"{MIA}; Attack/LiRA/results/<DS>/<DS>_lira_results.csv",
     "41 of 270 DP LiRA configs exceed e^eps * 0.01. NOT evidence of a broken "
     "guarantee -- see the table note and gaps.md 4.2. ci_low_exceeds_bound "
     "marks the rows that survive the exact CI."),
    ("config_inventory", "[9][59][61]", "Appendix A (Reproducibility) / [TBD]",
     f"<DS>/<FAMILY>/**/std_*_report.json; <DS>/<FAMILY>/**/dp_*_report.json; {UTIL}",
     "All software-version fields are [MISSING]: not recorded in any report and "
     "deliberately not read from the current machine. Hyperparameters missing "
     "on 11 non-private and 7 DP configs. See gaps.md 2."),
]

FIGURES = [
    ("gap_vs_leakage", "[51][65]", "Section 4.3 (Memorisation and leakage)",
     "revision/tables/csv/baseline_accuracy_matrix.csv; "
     "revision/tables/csv/baseline_leakage_all_pairs.csv",
     "Two panels: train-test gap vs baseline LiRA TPR@1% and vs baseline LiRA "
     "AUC. Spearman rho, n, p annotated on each axis. No in-image title."),
    ("gap_vs_leakage_vs_N", "[51][65]", "Section 4.3 (Memorisation and leakage)",
     "revision/tables/csv/baseline_accuracy_matrix.csv; "
     "revision/tables/csv/baseline_leakage_all_pairs.csv; " + UTIL,
     "Places the gap-based relationship beside the existing N-based one on a "
     "shared y axis, for direct comparison."),
    ("residual_floor_ci", "[14][15]", "Section 4.4 (Residual leakage under DP)",
     "revision/tables/csv/residual_leakage_eps1.csv; "
     "revision/tables/csv/baseline_leakage_all_pairs.csv",
     "RQ2b common-floor figure with Clopper-Pearson error bars on every point. "
     "Floor estimator (median, with IQR) stated in the axis annotation."),
    ("rq1c_utility_vs_protection", "[72][73][75]", "Section 4.5 (RQ1c)",
     "analysis/d16/d16_figure_input.csv; analysis/d16/d16_correlations.csv",
     "Regenerated from the d16 input; the recorded correlation is asserted, not "
     "recomputed, and a mismatch aborts the build. Utility axis labelled ACL. "
     "All 30 pairs."),
    ("rq1c_utility_vs_protection_exported", "[72][73][75]", "Section 4.5 (RQ1c)",
     "analysis/d16/d16_figure_input.csv; analysis/d16/d16_correlations.csv",
     "As above but on ACL_exported, which matches the artefact the attacks "
     "target. Carries 26 of 30 pairs: four DP-DNN rows have no exported "
     "accuracy. See gaps.md 3.1."),
    ("rq2a_benefit_vs_baseline", "[72][73][75]", "Section 4.5 (RQ2a)",
     "analysis/d16/d16_figure_input.csv; analysis/d16/d16_correlations.csv",
     "Regenerated from the d16 input; recorded correlation asserted."),
    ("l4_avg_vs_worst_case", "[72][73][75]", "Section 4.3 (Attack agreement)",
     "analysis/d16/d16_figure_input.csv; analysis/d16/d16_correlations.csv",
     "Average-case (Yeom advantage) vs worst-case (LiRA AUC) on the non-private "
     "targets. Recorded correlation asserted."),
    ("l6_baseline_leakage_vs_N", "[72][73][75]", "Section 4.3 (Dataset size)",
     "analysis/d16/d16_figure_input.csv; analysis/d16/d16_correlations.csv",
     "Non-private RF leakage against N. Six points. Recorded correlation asserted."),
]


def rows() -> list[dict]:
    out = []
    for stem, items, loc, src, note in TABLES:
        for ext, kind in (("csv", "table (CSV)"), ("tex", "table (LaTeX fragment)")):
            extra = ("" if ext == "csv" else
                     " MDPI booktabs fragment: \\input-able, carries "
                     "\\label{tab:%s}, no document preamble." % stem)
            out.append({
                "artifact_path": f"revision/tables/{ext}/{stem}.{ext}",
                "type": kind, "referee_items": items,
                "manuscript_location": loc, "source_files": src,
                "generated_by": "revision/src/build_tables.py",
                "notes": note + extra})

    for stem, items, loc, src, note in FIGURES:
        for ext, kind in (("pdf", "figure (PDF, vector)"), ("png", "figure (PNG, 300 dpi)")):
            out.append({
                "artifact_path": f"revision/figures/{ext}/{stem}.{ext}",
                "type": kind, "referee_items": items,
                "manuscript_location": loc, "source_files": src,
                "generated_by": "revision/src/build_figures.py",
                "notes": note})

    for ds in L.active_dataset_dirs():
        out.append({
            "artifact_path": f"revision/figures/png/loglog_roc_{ds}.png",
            "type": "figure (PNG, relocated)",
            "referee_items": "[62][78]",
            "manuscript_location": "Section 4.3 (LiRA ROC) / [TBD]",
            "source_files": L.LIRA_ROC_PNG.format(ds=ds),
            "generated_by": "revision/src/build_figures.py",
            "notes": ("COVERS STANDARD (NON-PRIVATE) TARGETS ONLY -- _loglog_roc() "
                      "filters to variant=='standard' and the ROC arrays are "
                      "stripped before the CSV write, so DP curves cannot be "
                      "produced without re-running LiRA. Copied UNCHANGED: the "
                      "title crop was attempted, inspected and found unsafe "
                      "(the title's lower line shares rows 77-79 with the top "
                      "y-tick label), so the two-line in-image title REMAINS. "
                      "See gaps.md 1.2 and 1.3.")})

    src_notes = {
        "config.py": ("Switches: INCLUDE_LUNG_CANCER, UTILITY_LABEL, RANDOM_FPR. "
                      "Also holds paths, the epsilon grids and the [MISSING]/"
                      "[CONFLICT] placeholder strings."),
        "loaders.py": ("Single source of truth for canonical naming (DP-RF vs RF, "
                       "BCP vs Breast Cancer) and for all input loading. Reads "
                       "processed_data.pkl through a restricted unpickler that "
                       "stubs every non-numpy class, so no estimator is ever "
                       "constructed. Run directly for a coverage report."),
        "build_tables.py": "Builds all nine tables as CSV + LaTeX.",
        "build_figures.py": ("Builds the eight generated figures and relocates the "
                             "six LiRA ROC PNGs."),
        "extract_provenance.py": ("Parses the report JSONs (globbed -- they sit at "
                                  "inconsistent depths) into config_inventory."),
        "verify.py": ("Re-derives every numeric table cell independently and writes "
                      "logs/verification.txt. Exits non-zero on any failure."),
        "make_manifest.py": "Emits this file.",
    }
    for name, note in src_notes.items():
        out.append({
            "artifact_path": f"revision/src/{name}", "type": "source",
            "referee_items": "n/a", "manuscript_location": "n/a",
            "source_files": "n/a", "generated_by": "hand-written", "notes": note})

    out.append({
        "artifact_path": "revision/logs/verification.txt", "type": "log",
        "referee_items": "n/a", "manuscript_location": "n/a",
        "source_files": "revision/tables/csv/*.csv and their primary sources",
        "generated_by": "revision/src/verify.py",
        "notes": ("One line per checked value: table | row | column | value | "
                  "source file | source row/filter. Failures listed at the top.")})
    out.append({
        "artifact_path": "revision/logs/gaps.md", "type": "documentation",
        "referee_items": "[3][5][6][8][9][22][48][49][50][59][61][62]",
        "manuscript_location": "Response letter",
        "source_files": "revision/logs/verification.txt; revision/logs/_figure_gaps.json",
        "generated_by": "hand-written from the pipeline's recorded gaps",
        "notes": ("Everything that could not be produced without compute, with "
                  "reasons. Written to be lifted into the response letter. "
                  "CONFLICTS section: none found.")})
    out.append({
        "artifact_path": "revision/README.md", "type": "documentation",
        "referee_items": "n/a", "manuscript_location": "n/a",
        "source_files": "n/a", "generated_by": "hand-written",
        "notes": "What the pipeline does and does not do; how to re-run it."})
    out.append({
        "artifact_path": "revision/MANIFEST.csv", "type": "documentation",
        "referee_items": "n/a", "manuscript_location": "n/a",
        "source_files": "n/a", "generated_by": "revision/src/make_manifest.py",
        "notes": "This file. One row per generated artifact."})
    return out


def main() -> int:
    data = rows()
    missing = [r["artifact_path"] for r in data
               if not (C.REPO / r["artifact_path"]).exists()
               and r["artifact_path"] != "revision/MANIFEST.csv"
               and r["artifact_path"] != "revision/README.md"]
    with open(C.REVISION / "MANIFEST.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
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
