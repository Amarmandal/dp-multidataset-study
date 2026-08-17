"""
Comprehensive Model Comparison Script for Diabetes Dataset

This script aggregates results from all DP models (RF, LR, GNB, SVM, DNN)
and generates comparative visualizations and tables for the research paper.

Usage:
    python compare_all_models.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

def load_model_results(model_name, base_path="../"):
    """Load results from a specific model directory."""
    model_path = Path(base_path) / model_name

    results = {}

    # Load DP results CSV
    dp_csv = model_path / f"dp_results.csv"
    if dp_csv.exists():
        results['dp_df'] = pd.read_csv(dp_csv)

    # Load DP JSON report
    dp_json = model_path / f"dp_{model_name.lower()}_report.json"
    if dp_json.exists():
        with open(dp_json, 'r') as f:
            results['dp_json'] = json.load(f)

    # Load baseline
    baseline_csv = model_path / "baseline_results.csv"
    if baseline_csv.exists():
        results['baseline'] = pd.read_csv(baseline_csv)

    return results


def aggregate_all_results():
    """Aggregate results from all models."""
    models = {
        'RF': 'RandomForest',
        'LR': 'LR',
        'GNB': 'GaussianNB',
        'SVM': 'SVM',
        'DNN': 'DNN'
    }

    all_results = {}
    baselines = {}

    for short_name, dir_name in models.items():
        print(f"Loading {short_name} results...")
        results = load_model_results(dir_name)

        if results:
            all_results[short_name] = results

            # Extract baseline
            if 'baseline' in results and not results['baseline'].empty:
                baselines[short_name] = results['baseline'].iloc[0]['accuracy']
            elif 'dp_json' in results and 'baseline_accuracy' in results['dp_json']:
                baselines[short_name] = results['dp_json']['baseline_accuracy']

    return all_results, baselines


def create_comparison_table(all_results, baselines, epsilon_values=[0.1, 0.8, 1.0, 2.0, 10.0]):
    """Create comprehensive comparison table."""

    comparison_data = []

    for model_name in all_results.keys():
        results = all_results[model_name]

        if 'dp_df' not in results:
            continue

        df = results['dp_df']
        baseline = baselines.get(model_name, None)

        for eps in epsilon_values:
            row_data = df[df['epsilon'] == eps]

            if row_data.empty:
                continue

            row = row_data.iloc[0]

            comparison_data.append({
                'Model': model_name,
                'Epsilon': eps,
                'Accuracy': row['accuracy_mean'],
                'Accuracy_Std': row.get('accuracy_std', 0),
                'F1': row['f1_score_mean'],
                'F1_Std': row.get('f1_score_std', 0),
                'Precision': row['precision_mean'],
                'Recall': row['recall_mean'],
                'Baseline_Acc': baseline,
                'ACL': row.get('accuracy_loss_mean', 0) if 'accuracy_loss_mean' in row else (baseline - row['accuracy_mean'] if baseline else 0)
            })

    return pd.DataFrame(comparison_data)


def plot_model_comparison(comparison_df, baselines):
    """Generate comparison visualizations."""

    # 1. Accuracy vs Epsilon for all models
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Accuracy vs Epsilon
    ax1 = axes[0, 0]
    for model in comparison_df['Model'].unique():
        model_data = comparison_df[comparison_df['Model'] == model]
        ax1.errorbar(model_data['Epsilon'], model_data['Accuracy'],
                    yerr=model_data['Accuracy_Std'],
                    marker='o', label=model, capsize=3, linewidth=2)

    # Add baseline lines
    for model, baseline in baselines.items():
        ax1.axhline(y=baseline, linestyle='--', alpha=0.3, linewidth=1)

    ax1.set_xlabel('Privacy Budget (ε)', fontsize=11)
    ax1.set_ylabel('Accuracy', fontsize=11)
    ax1.set_title('Accuracy vs Privacy Budget', fontsize=12, fontweight='bold')
    ax1.set_xscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: F1-Score vs Epsilon
    ax2 = axes[0, 1]
    for model in comparison_df['Model'].unique():
        model_data = comparison_df[comparison_df['Model'] == model]
        ax2.errorbar(model_data['Epsilon'], model_data['F1'],
                    yerr=model_data['F1_Std'],
                    marker='s', label=model, capsize=3, linewidth=2)

    ax2.set_xlabel('Privacy Budget (ε)', fontsize=11)
    ax2.set_ylabel('F1-Score', fontsize=11)
    ax2.set_title('F1-Score vs Privacy Budget', fontsize=12, fontweight='bold')
    ax2.set_xscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: ACL vs Epsilon
    ax3 = axes[1, 0]
    for model in comparison_df['Model'].unique():
        model_data = comparison_df[comparison_df['Model'] == model]
        ax3.plot(model_data['Epsilon'], model_data['ACL'] * 100,
                marker='^', label=model, linewidth=2)

    ax3.set_xlabel('Privacy Budget (ε)', fontsize=11)
    ax3.set_ylabel('Accuracy Loss (%)', fontsize=11)
    ax3.set_title('Accuracy Loss vs Privacy Budget', fontsize=12, fontweight='bold')
    ax3.set_xscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='green', linestyle='--', alpha=0.5)

    # Plot 4: Precision-Recall Trade-off at ε=1.0
    ax4 = axes[1, 1]
    eps_1_data = comparison_df[comparison_df['Epsilon'] == 1.0]

    for _, row in eps_1_data.iterrows():
        ax4.scatter(row['Recall'], row['Precision'], s=200, alpha=0.6)
        ax4.annotate(row['Model'], (row['Recall'], row['Precision']),
                    fontsize=10, ha='center')

    ax4.set_xlabel('Recall', fontsize=11)
    ax4.set_ylabel('Precision', fontsize=11)
    ax4.set_title('Precision-Recall at ε=1.0', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig('comprehensive_model_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: comprehensive_model_comparison.png")
    plt.show()


def generate_latex_table(comparison_df, epsilon_values=[0.8, 1.0, 2.0]):
    """Generate LaTeX table for paper."""

    latex_lines = []
    latex_lines.append("\\begin{table}[h]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Model Performance Comparison on Diabetes Dataset}")
    latex_lines.append("\\label{tab:diabetes_comparison}")
    latex_lines.append("\\begin{tabular}{lcccccc}")
    latex_lines.append("\\hline")
    latex_lines.append("Model & $\\epsilon$ & Accuracy & F1 & Precision & Recall & ACL \\\\")
    latex_lines.append("\\hline")

    for eps in epsilon_values:
        eps_data = comparison_df[comparison_df['Epsilon'] == eps].sort_values('Accuracy', ascending=False)

        for _, row in eps_data.iterrows():
            line = f"{row['Model']} & {eps:.1f} & {row['Accuracy']:.4f} & {row['F1']:.4f} & "
            line += f"{row['Precision']:.4f} & {row['Recall']:.4f} & {row['ACL']:.4f} \\\\"
            latex_lines.append(line)

        latex_lines.append("\\hline")

    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")

    latex_table = "\n".join(latex_lines)

    with open('diabetes_comparison_table.tex', 'w') as f:
        f.write(latex_table)

    print("✓ Saved: diabetes_comparison_table.tex")
    return latex_table


def find_optimal_epsilon(comparison_df, acl_threshold=0.15):
    """Find optimal epsilon for each model based on ACL threshold."""

    print(f"\n{'='*60}")
    print(f"OPTIMAL EPSILON ANALYSIS (ACL threshold: {acl_threshold:.2%})")
    print(f"{'='*60}")

    optimal_eps = {}

    for model in comparison_df['Model'].unique():
        model_data = comparison_df[comparison_df['Model'] == model]

        # Find minimum epsilon where ACL <= threshold
        acceptable = model_data[model_data['ACL'] <= acl_threshold]

        if not acceptable.empty:
            optimal = acceptable.loc[acceptable['Epsilon'].idxmin()]
            optimal_eps[model] = {
                'epsilon': optimal['Epsilon'],
                'accuracy': optimal['Accuracy'],
                'f1': optimal['F1'],
                'acl': optimal['ACL']
            }

            print(f"\n{model}:")
            print(f"  Optimal ε: {optimal['Epsilon']:.1f}")
            print(f"  Accuracy: {optimal['Accuracy']:.4f}")
            print(f"  F1-Score: {optimal['F1']:.4f}")
            print(f"  ACL: {optimal['ACL']:.4f} ({optimal['ACL']*100:.2f}%)")
        else:
            print(f"\n{model}: No epsilon meets ACL ≤ {acl_threshold:.2%} threshold")

    return optimal_eps


def main():
    """Main execution function."""

    print("="*60)
    print("DIABETES DATASET - COMPREHENSIVE MODEL COMPARISON")
    print("="*60)

    # Load all results
    all_results, baselines = aggregate_all_results()

    print(f"\nLoaded results for {len(all_results)} models")
    print(f"Baselines: {baselines}")

    # Create comparison table
    comparison_df = create_comparison_table(all_results, baselines)

    # Save comparison table
    comparison_df.to_csv('all_models_comparison.csv', index=False)
    print("\n✓ Saved: all_models_comparison.csv")

    # Generate visualizations
    plot_model_comparison(comparison_df, baselines)

    # Generate LaTeX table
    generate_latex_table(comparison_df)

    # Find optimal epsilon values
    optimal_eps = find_optimal_epsilon(comparison_df, acl_threshold=0.15)

    # Best model at key epsilon values
    print(f"\n{'='*60}")
    print("BEST MODELS AT KEY EPSILON VALUES")
    print(f"{'='*60}")

    for eps in [0.1, 0.8, 1.0, 2.0, 10.0]:
        eps_data = comparison_df[comparison_df['Epsilon'] == eps]
        if not eps_data.empty:
            best = eps_data.loc[eps_data['Accuracy'].idxmax()]
            print(f"\nε = {eps}:")
            print(f"  Best Model: {best['Model']}")
            print(f"  Accuracy: {best['Accuracy']:.4f} ± {best['Accuracy_Std']:.4f}")
            print(f"  F1-Score: {best['F1']:.4f}")

    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
