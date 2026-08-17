"""
Comprehensive Model Comparison Script for Kidney Stone Risk Dataset
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

SCRIPT_DIR = Path(__file__).parent
KIDNEY_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "output"

MODELS = {
    'RF':  ('RandomForest', 'dp_results.csv',     'baseline_results.csv'),
    'LR':  ('LR',           'dp_results.csv',     'baseline_results.csv'),
    'GNB': ('GaussianNB',   'dp_results.csv',     'baseline_results.csv'),
    'SVM': ('SVM',          'dp_results.csv',     'baseline_results.csv'),
    'DNN': ('DNN',          'dp_dnn_results.csv', 'baseline_dnn_results.csv'),
}


def _get_acl(row, baseline):
    """Return ACL, trying stored columns first then computing from baseline."""
    for col in ('accuracy_loss_mean', 'ACL'):
        if col in row.index and not pd.isna(row[col]):
            return float(row[col])
    if baseline is not None:
        return float(baseline - row['accuracy_mean'])
    return 0.0


def load_model_results(short_name):
    """Load dp and baseline CSVs for one model. Returns None if DP data is missing."""
    dir_name, dp_file, baseline_file = MODELS[short_name]
    model_dir = KIDNEY_DIR / dir_name / "output"

    dp_path = model_dir / dp_file
    baseline_path = model_dir / baseline_file

    if not dp_path.exists():
        print(f"  [SKIP] {short_name}: DP results not found at {dp_path}")
        return None

    try:
        dp_df = pd.read_csv(dp_path)
    except Exception as e:
        print(f"  [SKIP] {short_name}: failed to read DP results — {e}")
        return None

    baseline_acc = None
    if baseline_path.exists():
        try:
            baseline_df = pd.read_csv(baseline_path)
            if not baseline_df.empty and 'accuracy' in baseline_df.columns:
                baseline_acc = float(baseline_df.iloc[0]['accuracy'])
        except Exception as e:
            print(f"  [WARN] {short_name}: failed to read baseline — {e}")
    else:
        print(f"  [WARN] {short_name}: baseline file not found at {baseline_path}")

    return {'dp_df': dp_df, 'baseline_acc': baseline_acc}


def aggregate_all_results():
    all_results = {}
    baselines = {}

    for short_name in MODELS:
        print(f"Loading {short_name}...")
        result = load_model_results(short_name)
        if result is None:
            continue
        all_results[short_name] = result
        if result['baseline_acc'] is not None:
            baselines[short_name] = result['baseline_acc']
            print(f"  Loaded — baseline accuracy: {result['baseline_acc']:.4f}, "
                  f"{len(result['dp_df'])} DP rows")
        else:
            print(f"  Loaded — no baseline, {len(result['dp_df'])} DP rows")

    return all_results, baselines


def create_comparison_table(all_results, baselines, epsilon_values=None):
    """Build a unified comparison DataFrame. Uses all available epsilons if epsilon_values is None."""
    comparison_data = []

    for model_name, result in all_results.items():
        df = result['dp_df']
        baseline = baselines.get(model_name)

        eps_to_use = epsilon_values if epsilon_values else sorted(df['epsilon'].unique())

        for eps in eps_to_use:
            rows = df[df['epsilon'] == eps]
            if rows.empty:
                continue

            row = rows.iloc[0]
            acl = _get_acl(row, baseline)

            comparison_data.append({
                'Model':        model_name,
                'Epsilon':      eps,
                'Accuracy':     row['accuracy_mean'],
                'Accuracy_Std': row.get('accuracy_std', 0),
                'F1':           row['f1_score_mean'],
                'F1_Std':       row.get('f1_score_std', 0),
                'Precision':    row['precision_mean'],
                'Recall':       row['recall_mean'],
                'Baseline_Acc': baseline,
                'ACL':          acl,
            })

    return pd.DataFrame(comparison_data)


def plot_model_comparison(comparison_df, baselines):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax1 = axes[0, 0]
    for model in comparison_df['Model'].unique():
        d = comparison_df[comparison_df['Model'] == model]
        ax1.errorbar(d['Epsilon'], d['Accuracy'], yerr=d['Accuracy_Std'],
                     marker='o', label=model, capsize=3, linewidth=2)
    for model, bline in baselines.items():
        ax1.axhline(y=bline, linestyle='--', alpha=0.3, linewidth=1)
    ax1.set_xlabel('Privacy Budget (ε)', fontsize=11)
    ax1.set_ylabel('Accuracy', fontsize=11)
    ax1.set_title('Accuracy vs Privacy Budget\n(Kidney Stone Dataset)', fontsize=12, fontweight='bold')
    ax1.set_xscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    for model in comparison_df['Model'].unique():
        d = comparison_df[comparison_df['Model'] == model]
        ax2.errorbar(d['Epsilon'], d['F1'], yerr=d['F1_Std'],
                     marker='s', label=model, capsize=3, linewidth=2)
    ax2.set_xlabel('Privacy Budget (ε)', fontsize=11)
    ax2.set_ylabel('F1-Score', fontsize=11)
    ax2.set_title('F1-Score vs Privacy Budget\n(Kidney Stone Dataset)', fontsize=12, fontweight='bold')
    ax2.set_xscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    for model in comparison_df['Model'].unique():
        d = comparison_df[comparison_df['Model'] == model]
        ax3.plot(d['Epsilon'], d['ACL'] * 100, marker='^', label=model, linewidth=2)
    ax3.set_xlabel('Privacy Budget (ε)', fontsize=11)
    ax3.set_ylabel('Accuracy Loss (%)', fontsize=11)
    ax3.set_title('Accuracy Loss vs Privacy Budget\n(Kidney Stone Dataset)', fontsize=12, fontweight='bold')
    ax3.set_xscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='green', linestyle='--', alpha=0.5)

    ax4 = axes[1, 1]
    eps_1_data = comparison_df[comparison_df['Epsilon'] == 1.0]
    if eps_1_data.empty:
        closest_eps = min(comparison_df['Epsilon'].unique(), key=lambda x: abs(x - 1.0))
        eps_1_data = comparison_df[comparison_df['Epsilon'] == closest_eps]
        ax4.set_title(f'Precision-Recall at ε={closest_eps}\n(Kidney Stone Dataset)',
                      fontsize=12, fontweight='bold')
    else:
        ax4.set_title('Precision-Recall at ε=1.0\n(Kidney Stone Dataset)',
                      fontsize=12, fontweight='bold')

    for _, row in eps_1_data.iterrows():
        ax4.scatter(row['Recall'], row['Precision'], s=200, alpha=0.6)
        ax4.annotate(row['Model'], (row['Recall'], row['Precision']),
                     fontsize=10, ha='center', va='bottom')
    ax4.set_xlabel('Recall', fontsize=11)
    ax4.set_ylabel('Precision', fontsize=11)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "comprehensive_model_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


def generate_latex_table(comparison_df, epsilon_values=None):
    if epsilon_values is None:
        epsilon_values = [0.1, 0.4, 0.8, 1.0, 2.0, 10.0]
    epsilon_values = [e for e in epsilon_values if e in comparison_df['Epsilon'].values]

    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Model Performance Comparison on Kidney Stone Risk Dataset}",
        "\\label{tab:kidney_stone_comparison}",
        "\\begin{tabular}{lcccccc}",
        "\\hline",
        "Model & $\\epsilon$ & Accuracy & F1 & Precision & Recall & ACL \\\\",
        "\\hline",
    ]
    for eps in epsilon_values:
        eps_data = comparison_df[comparison_df['Epsilon'] == eps].sort_values('Accuracy', ascending=False)
        for _, row in eps_data.iterrows():
            lines.append(
                f"{row['Model']} & {eps:.1f} & {row['Accuracy']:.4f} & {row['F1']:.4f} & "
                f"{row['Precision']:.4f} & {row['Recall']:.4f} & {row['ACL']:.4f} \\\\"
            )
        lines.append("\\hline")
    lines += ["\\end{tabular}", "\\end{table}"]

    out_path = OUTPUT_DIR / "kidney_stone_comparison_table.tex"
    out_path.write_text("\n".join(lines))
    print(f"Saved: {out_path}")


def find_optimal_epsilon(comparison_df, acl_threshold=0.15):
    print(f"\n{'='*60}")
    print(f"OPTIMAL EPSILON ANALYSIS (ACL threshold: {acl_threshold:.2%})")
    print(f"{'='*60}")

    optimal_eps = {}
    for model in comparison_df['Model'].unique():
        model_data = comparison_df[comparison_df['Model'] == model]
        acceptable = model_data[model_data['ACL'] <= acl_threshold]

        if not acceptable.empty:
            optimal = acceptable.loc[acceptable['Epsilon'].idxmin()]
            optimal_eps[model] = {
                'epsilon':  optimal['Epsilon'],
                'accuracy': optimal['Accuracy'],
                'f1':       optimal['F1'],
                'acl':      optimal['ACL'],
            }
            print(f"\n{model}:")
            print(f"  Optimal epsilon: {optimal['Epsilon']:.1f}")
            print(f"  Accuracy:        {optimal['Accuracy']:.4f}")
            print(f"  F1-Score:        {optimal['F1']:.4f}")
            print(f"  ACL:             {optimal['ACL']:.4f} ({optimal['ACL']*100:.2f}%)")
        else:
            print(f"\n{model}: No epsilon meets ACL ≤ {acl_threshold:.2%} threshold")

    return optimal_eps


def main():
    print("=" * 60)
    print("KIDNEY STONE RISK DATASET - COMPREHENSIVE MODEL COMPARISON")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results, baselines = aggregate_all_results()

    if not all_results:
        print("\nNo model data found. Check that model output directories exist under KIDNEY_STONE/.")
        return

    print(f"\nLoaded {len(all_results)} model(s): {', '.join(all_results.keys())}")
    print(f"Baselines: { {k: f'{v:.4f}' for k, v in baselines.items()} }")

    comparison_df = create_comparison_table(all_results, baselines)
    out_csv = OUTPUT_DIR / "all_models_comparison.csv"
    comparison_df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}  ({len(comparison_df)} rows)")

    if comparison_df.empty:
        print("No comparative results to plot.")
        return

    plot_model_comparison(comparison_df, baselines)
    generate_latex_table(comparison_df)
    find_optimal_epsilon(comparison_df, acl_threshold=0.15)

    print(f"\n{'='*60}")
    print("BEST MODELS AT KEY EPSILON VALUES")
    print(f"{'='*60}")
    for eps in sorted(comparison_df['Epsilon'].unique()):
        eps_data = comparison_df[comparison_df['Epsilon'] == eps]
        best = eps_data.loc[eps_data['Accuracy'].idxmax()]
        print(f"\n  ε={eps}: best={best['Model']}  "
              f"acc={best['Accuracy']:.4f}±{best['Accuracy_Std']:.4f}  "
              f"F1={best['F1']:.4f}  ACL={best['ACL']:.4f}")

    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
