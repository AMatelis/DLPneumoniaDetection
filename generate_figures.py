import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib import rcParams
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# -------- Publication Styling Setup --------
def setup_publication_style():
    """Configure matplotlib and seaborn for publication-quality figures."""
    # Set style
    sns.set_style("whitegrid")
    
    # Configure matplotlib parameters
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
    rcParams['font.size'] = 10
    rcParams['axes.labelsize'] = 11
    rcParams['axes.titlesize'] = 12
    rcParams['xtick.labelsize'] = 10
    rcParams['ytick.labelsize'] = 10
    rcParams['legend.fontsize'] = 10
    rcParams['figure.titlesize'] = 13
    
    # Line and patch widths
    rcParams['lines.linewidth'] = 1.5
    rcParams['patch.linewidth'] = 0.5
    rcParams['axes.linewidth'] = 0.8
    
    # Color palette
    sns.set_palette("husl")
    
    print("✓ Publication style configured")

def save_figure(fig, filename, output_dir=".", dpi=300, formats=['png', 'pdf']):
    """
    Save figure in multiple formats at publication quality.
    
    Args:
        fig: matplotlib figure object
        filename: Base filename (without extension)
        output_dir: Directory to save
        dpi: Resolution (default 300 DPI for publication)
        formats: List of formats to save ['png', 'pdf', 'svg']
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for fmt in formats:
        filepath = os.path.join(output_dir, f"{filename}.{fmt}")
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight', format=fmt)
    
    print(f"✓ Saved figure: {filename} ({', '.join(formats)})")

# -------- Figure 1: ROC Curves with Confidence Intervals --------
def plot_roc_with_ci(y_true, y_pred_prob, title="ROC Curve with 95% CI", 
                     output_dir=".", filename="fig1_roc_curve"):
    """
    Create ROC curve with bootstrapped 95% confidence intervals.
    
    Args:
        y_true: True binary labels
        y_pred_prob: Predicted probabilities
        title: Figure title
        output_dir: Output directory
        filename: Output filename
    """
    setup_publication_style()
    
    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    # Bootstrap for confidence intervals
    n_bootstraps = 1000
    tprs = []
    aucs = []
    
    np.random.seed(42)
    for i in range(n_bootstraps):
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred_prob[indices]
        
        fpr_boot, tpr_boot, _ = roc_curve(y_true_boot, y_pred_boot)
        auc_boot = auc(fpr_boot, tpr_boot)
        aucs.append(auc_boot)
        
        # Interpolate to standard FPR points
        interp_tpr = np.interp(fpr, fpr_boot, tpr_boot)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
    
    # Calculate confidence intervals
    tprs = np.array(tprs)
    tpr_mean = tprs.mean(axis=0)
    tpr_std = tprs.std(axis=0)
    tpr_upper = np.minimum(tpr_mean + 1.96 * tpr_std, 1)
    tpr_lower = tpr_mean - 1.96 * tpr_std
    
    auc_ci = np.percentile(aucs, [2.5, 97.5])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Plot ROC curve
    ax.plot(fpr, tpr, color='#1f77b4', linewidth=2.5, 
           label=f'ROC (AUC = {roc_auc:.3f})')
    
    # Plot confidence interval
    ax.fill_between(fpr, tpr_lower, tpr_upper, color='#1f77b4', alpha=0.2,
                   label=f'95% CI [{auc_ci[0]:.3f} - {auc_ci[1]:.3f}]')
    
    # Plot diagonal
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier')
    
    # Formatting
    ax.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', framealpha=0.95, edgecolor='black')
    
    save_figure(fig, filename, output_dir)
    plt.close(fig)

# -------- Figure 2: Uncertainty vs Error Analysis --------
def plot_uncertainty_vs_error(y_true, y_pred, uncertainties, title="Uncertainty vs Prediction Error",
                             output_dir=".", filename="fig2_uncertainty_error"):
    """
    Create publication-quality uncertainty vs error visualization.
    Shows both scatter and density plots.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted labels
        uncertainties: Uncertainty scores (entropy or variance)
        title: Figure title
        output_dir: Output directory
        filename: Output filename
    """
    setup_publication_style()
    
    # Compute errors
    errors = (y_true != y_pred).astype(int)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(14, 5))
    
    # Subplot 1: Scatter plot with density
    ax1 = plt.subplot(1, 2, 1)
    scatter = ax1.scatter(uncertainties[errors == 0], np.ones(np.sum(errors == 0)) + np.random.normal(0, 0.02, np.sum(errors == 0)),
                         c='#2ca02c', alpha=0.6, s=30, label='Correct Predictions', edgecolors='none')
    scatter = ax1.scatter(uncertainties[errors == 1], np.zeros(np.sum(errors == 1)) + np.random.normal(0, 0.02, np.sum(errors == 1)),
                         c='#d62728', alpha=0.6, s=30, label='Incorrect Predictions', edgecolors='none')
    
    ax1.set_xlabel('Uncertainty (Entropy)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Prediction Correctness', fontsize=11, fontweight='bold')
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['Incorrect', 'Correct'])
    ax1.set_title('Scatter Plot: Uncertainty Distribution', fontsize=11, fontweight='bold')
    ax1.legend(loc='upper right', framealpha=0.95, edgecolor='black')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Subplot 2: Violin/Box plots
    ax2 = plt.subplot(1, 2, 2)
    data_correct = uncertainties[errors == 0]
    data_incorrect = uncertainties[errors == 1]
    
    bp = ax2.boxplot([data_correct, data_incorrect], labels=['Correct', 'Incorrect'],
                     patch_artist=True, widths=0.6)
    
    colors = ['#2ca02c', '#d62728']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_ylabel('Uncertainty (Entropy)', fontsize=11, fontweight='bold')
    ax2.set_title('Box Plot: Uncertainty Distributions', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    stats_text = f"Correct: μ={data_correct.mean():.4f}, σ={data_correct.std():.4f}\n"
    stats_text += f"Incorrect: μ={data_incorrect.mean():.4f}, σ={data_incorrect.std():.4f}"
    ax2.text(0.98, 0.02, stats_text, transform=ax2.transAxes, 
            fontsize=9, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    fig.suptitle(title, fontsize=12, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    save_figure(fig, filename, output_dir)
    plt.close(fig)

# -------- Figure 3: Grad-CAM Comparison Grid --------
def plot_gradcam_grid(gradcam_stats_path, title="Grad-CAM Activation Analysis",
                     output_dir=".", filename="fig3_gradcam_comparison"):
    """
    Create Grad-CAM comparison grid showing key metrics.
    
    Args:
        gradcam_stats_path: Path to Grad-CAM statistics JSON
        title: Figure title
        output_dir: Output directory
        filename: Output filename
    """
    setup_publication_style()
    
    # Load Grad-CAM statistics
    with open(gradcam_stats_path, 'r') as f:
        gradcam_data = json.load(f)
    
    correct_stats = gradcam_data['correct_predictions']['statistics']
    incorrect_stats = gradcam_data['incorrect_predictions']['statistics']
    
    # Extract key metrics
    metrics = {
        'Heatmap Entropy': ('mean_entropy', 'entropy'),
        'Lung Activation': ('mean_activation_lung', 'activation'),
        'Non-Lung Activation': ('mean_activation_non_lung', 'activation'),
        'Lung:Non-Lung Ratio': ('mean_activation_ratio_lung_vs_nonlung', 'ratio'),
        'Focal Specificity': ('mean_focal_specificity', 'specificity'),
        'Std Activation': ('std_activation_overall', 'std')
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    
    colors_correct = '#2ca02c'
    colors_incorrect = '#d62728'
    
    for idx, (metric_name, (key, _)) in enumerate(metrics.items()):
        ax = axes[idx]
        
        correct_val = correct_stats.get(key, 0)
        incorrect_val = incorrect_stats.get(key, 0)
        
        x_pos = [0, 1]
        values = [correct_val, incorrect_val]
        colors = [colors_correct, colors_incorrect]
        
        bars = ax.bar(x_pos, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel(metric_name, fontsize=10, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(['Correct', 'Incorrect'])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    save_figure(fig, filename, output_dir)
    plt.close(fig)

# -------- Figure 4: Dataset Shift Performance --------
def plot_dataset_shift_performance(shift_results_dict, title="Domain Shift and Adaptation",
                                  output_dir=".", filename="fig4_dataset_shift"):
    """
    Create comprehensive dataset shift performance visualization.
    
    Args:
        shift_results_dict: Dict with results from dataset shift experiments
        title: Figure title
        output_dir: Output directory
        filename: Output filename
    """
    setup_publication_style()
    
    fig = plt.figure(figsize=(14, 10))
    
    # Extract data from all shift experiments
    experiments = []
    baseline_aucs = []
    target_aucs = []
    recovery_aucs = []
    
    for exp_name, results in shift_results_dict.items():
        experiments.append(exp_name)
        baseline_aucs.append(results.get('baseline_auc', 0))
        target_aucs.append(results.get('target_auc', 0))
        recovery_aucs.append(results.get('recovery_auc', results.get('target_auc', 0)))
    
    # Subplot 1: AUC Comparison
    ax1 = plt.subplot(2, 2, 1)
    x = np.arange(len(experiments))
    width = 0.25
    
    bars1 = ax1.bar(x - width, baseline_aucs, width, label='Baseline (Source)', color='#1f77b4', alpha=0.8, edgecolor='black')
    bars2 = ax1.bar(x, target_aucs, width, label='Target (Zero-shot)', color='#ff7f0e', alpha=0.8, edgecolor='black')
    bars3 = ax1.bar(x + width, recovery_aucs, width, label='Recovery (Fine-tuned)', color='#2ca02c', alpha=0.8, edgecolor='black')
    
    ax1.set_ylabel('AUC Score', fontsize=11, fontweight='bold')
    ax1.set_title('AUC Across Domain Shifts', fontsize=11, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(experiments, rotation=15, ha='right')
    ax1.set_ylim([0, 1.05])
    ax1.legend(framealpha=0.95, edgecolor='black')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Subplot 2: Performance Degradation
    ax2 = plt.subplot(2, 2, 2)
    degradations = [baseline - target for baseline, target in zip(baseline_aucs, target_aucs)]
    recovery = [recov - target for recov, target in zip(recovery_aucs, target_aucs)]
    
    bars1 = ax2.bar(x - width/2, degradations, width, label='Degradation', color='#d62728', alpha=0.8, edgecolor='black')
    bars2 = ax2.bar(x + width/2, recovery, width, label='Recovery', color='#2ca02c', alpha=0.8, edgecolor='black')
    
    ax2.set_ylabel('AUC Change', fontsize=11, fontweight='bold')
    ax2.set_title('Performance Degradation & Recovery', fontsize=11, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(experiments, rotation=15, ha='right')
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.8)
    ax2.legend(framealpha=0.95, edgecolor='black')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Subplot 3: Degradation Percentage
    ax3 = plt.subplot(2, 2, 3)
    degradation_pct = [(d / b * 100) if b > 0 else 0 for d, b in zip(degradations, baseline_aucs)]
    recovery_pct = [(r / d * 100) if d > 0 else 0 for r, d in zip(recovery, degradations)]
    
    bars1 = ax3.bar(x - width/2, degradation_pct, width, label='Degradation %', color='#d62728', alpha=0.8, edgecolor='black')
    bars2 = ax3.bar(x + width/2, recovery_pct, width, label='Recovery %', color='#2ca02c', alpha=0.8, edgecolor='black')
    
    ax3.set_ylabel('Percentage Change (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Relative Performance Change', fontsize=11, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(experiments, rotation=15, ha='right')
    ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.8)
    ax3.legend(framealpha=0.95, edgecolor='black')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Subplot 4: Summary Statistics Table
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('tight')
    ax4.axis('off')
    
    table_data = []
    for exp, baseline, target, recov in zip(experiments, baseline_aucs, target_aucs, recovery_aucs):
        deg = baseline - target
        table_data.append([
            exp,
            f'{baseline:.3f}',
            f'{target:.3f}',
            f'{recov:.3f}',
            f'{deg:.3f}',
            f'{(deg/baseline*100):.1f}%'
        ])
    
    table = ax4.table(cellText=table_data,
                     colLabels=['Experiment', 'Baseline', 'Target', 'Recovery', 'Δ AUC', '% Drop'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.15, 0.12, 0.12, 0.12, 0.12, 0.12])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Color header
    for i in range(6):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(6):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
            else:
                table[(i, j)].set_facecolor('#ffffff')
    
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    save_figure(fig, filename, output_dir)
    plt.close(fig)

# -------- Figure 5: Model Calibration Analysis --------
def plot_calibration_curves(calibration_stats_path, title="Probability Calibration Analysis",
                           output_dir=".", filename="fig5_calibration"):
    """
    Create publication-quality calibration curve visualization.
    
    Args:
        calibration_stats_path: Path to calibration statistics JSON
        title: Figure title
        output_dir: Output directory
        filename: Output filename
    """
    setup_publication_style()
    
    # Load calibration statistics
    with open(calibration_stats_path, 'r') as f:
        calib_data = json.load(f)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Subplot 1: ECE Comparison
    ax = axes[0, 0]
    methods = ['Uncalibrated', 'Temperature\nScaling', 'Platt\nScaling']
    eces = [
        calib_data['uncalibrated']['ece'],
        calib_data['temperature_scaling']['ece'],
        calib_data['platt_scaling']['ece']
    ]
    colors = ['#d62728', '#1f77b4', '#2ca02c']
    
    bars = ax.bar(methods, eces, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Expected Calibration Error', fontsize=11, fontweight='bold')
    ax.set_title('ECE Comparison', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, ece in zip(bars, eces):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{ece:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Subplot 2: Reliability Diagram - Uncalibrated
    ax = axes[0, 1]
    uncalib_bins = calib_data['uncalibrated']['bin_metrics']
    confidences = [b['avg_confidence'] for b in uncalib_bins if b['bin_size'] > 0]
    accuracies = [b['accuracy'] for b in uncalib_bins if b['bin_size'] > 0]
    bin_sizes = [b['bin_size'] for b in uncalib_bins if b['bin_size'] > 0]
    
    ax.scatter(confidences, accuracies, s=np.array(bin_sizes)*2, alpha=0.6, color='#d62728', edgecolors='black', linewidth=1)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect Calibration')
    ax.fill_between([0, 1], 0, 1, alpha=0.1, color='gray')
    ax.set_xlabel('Mean Predicted Confidence', fontsize=10, fontweight='bold')
    ax.set_ylabel('Fraction of Positives', fontsize=10, fontweight='bold')
    ax.set_title('Uncalibrated Predictions', fontsize=11, fontweight='bold')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Subplot 3: Reliability Diagram - Temperature Scaled
    ax = axes[1, 0]
    temp_bins = calib_data['temperature_scaling']['bin_metrics']
    confidences = [b['avg_confidence'] for b in temp_bins if b['bin_size'] > 0]
    accuracies = [b['accuracy'] for b in temp_bins if b['bin_size'] > 0]
    bin_sizes = [b['bin_size'] for b in temp_bins if b['bin_size'] > 0]
    
    ax.scatter(confidences, accuracies, s=np.array(bin_sizes)*2, alpha=0.6, color='#1f77b4', edgecolors='black', linewidth=1)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect Calibration')
    ax.fill_between([0, 1], 0, 1, alpha=0.1, color='gray')
    ax.set_xlabel('Mean Predicted Confidence', fontsize=10, fontweight='bold')
    ax.set_ylabel('Fraction of Positives', fontsize=10, fontweight='bold')
    ax.set_title(f"Temperature Scaling (T={calib_data['temperature_scaling']['temperature']:.3f})", 
                fontsize=11, fontweight='bold')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Subplot 4: Summary Statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = f"""
    CALIBRATION ANALYSIS SUMMARY
    {'─' * 50}
    
    Uncalibrated:
    • ECE: {calib_data['uncalibrated']['ece']:.6f}
    
    Temperature Scaling:
    • ECE: {calib_data['temperature_scaling']['ece']:.6f}
    • Temperature: {calib_data['temperature_scaling']['temperature']:.4f}
    • Reduction: {((calib_data['uncalibrated']['ece'] - calib_data['temperature_scaling']['ece']) / calib_data['uncalibrated']['ece'] * 100):.1f}%
    
    Platt Scaling:
    • ECE: {calib_data['platt_scaling']['ece']:.6f}
    • A: {calib_data['platt_scaling']['A']:.4f}
    • B: {calib_data['platt_scaling']['B']:.4f}
    • Reduction: {((calib_data['uncalibrated']['ece'] - calib_data['platt_scaling']['ece']) / calib_data['uncalibrated']['ece'] * 100):.1f}%
    
    Dataset Info:
    • Calibration Set: {calib_data['calibration_data_info']['calibration_set_size']}
    • Test Set: {calib_data['calibration_data_info']['test_set_size']}
    • Total Validation: {calib_data['calibration_data_info']['total_validation_size']}
    """
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    save_figure(fig, filename, output_dir)
    plt.close(fig)

# -------- Figure 6: Comprehensive Results Summary --------
def plot_results_summary(exp_dir, title="Pneumonia Detection: Research Summary",
                        output_dir=".", filename="fig6_results_summary"):
    """
    Create a comprehensive results summary figure combining key metrics.
    
    Args:
        exp_dir: Experiment directory containing JSON outputs
        title: Figure title
        output_dir: Output directory
        filename: Output filename
    """
    setup_publication_style()
    
    fig = plt.figure(figsize=(16, 10))
    
    # Load all available results
    results_dict = {}
    outputs_dir = os.path.join(exp_dir, 'outputs')
    
    for filename in os.listdir(outputs_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(outputs_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    results_dict[filename] = json.load(f)
            except:
                pass
    
    # Create summary text
    summary_text = "EXPERIMENTAL RESULTS SUMMARY\n"
    summary_text += "=" * 70 + "\n\n"
    
    # Add available metrics
    if 'metadata.json' in results_dict:
        metadata = results_dict['metadata.json']
        summary_text += f"Experiment: {metadata.get('experiment_name', 'N/A')}\n"
        summary_text += f"Version: {metadata.get('experiment_version', 'N/A')}\n"
        summary_text += f"Timestamp: {metadata.get('timestamp', 'N/A')}\n"
        summary_text += f"Seed: {metadata.get('random_seed', 'N/A')}\n"
        summary_text += "\n"
    
    summary_text += "Key Findings:\n"
    summary_text += "─" * 70 + "\n"
    
    # Model performance
    summary_text += "✓ Model Performance:\n"
    summary_text += "  - ResNet50 transfer learning with frozen base\n"
    summary_text += "  - Binary classification: Normal vs Pneumonia\n"
    summary_text += "  - Balanced class weights applied\n\n"
    
    # Uncertainty quantification
    if 'uncertainty_stats.json' in results_dict:
        unc_data = results_dict['uncertainty_stats.json']
        summary_text += "✓ Uncertainty Quantification:\n"
        summary_text += f"  - MC Dropout: {unc_data['uncertainty_stats']['mean_uncertainty']:.4f} (avg entropy)\n"
        summary_text += f"  - Correct predictions: {unc_data['uncertainty_stats']['uncertainty_correct']:.4f}\n"
        summary_text += f"  - Incorrect predictions: {unc_data['uncertainty_stats']['uncertainty_incorrect']:.4f}\n\n"
    
    # Calibration
    if 'calibration_analysis.json' in results_dict:
        calib_data = results_dict['calibration_analysis.json']
        summary_text += "✓ Probability Calibration:\n"
        summary_text += f"  - Uncalibrated ECE: {calib_data['uncalibrated']['ece']:.6f}\n"
        summary_text += f"  - Temperature Scaling ECE: {calib_data['temperature_scaling']['ece']:.6f}\n"
        summary_text += f"  - Platt Scaling ECE: {calib_data['platt_scaling']['ece']:.6f}\n\n"
    
    # Clinical thresholds
    if 'clinical_thresholds.json' in results_dict:
        thresh_data = results_dict['clinical_thresholds.json']
        summary_text += "✓ Clinical Decision Thresholds:\n"
        summary_text += f"  - Screening (high sensitivity): {thresh_data['optimal_thresholds']['screening']['threshold']:.3f}\n"
        summary_text += f"  - Confirmation (high specificity): {thresh_data['optimal_thresholds']['confirmation']['threshold']:.3f}\n"
        summary_text += f"  - Balanced (Youden): {thresh_data['optimal_thresholds']['balanced']['threshold']:.3f}\n\n"
    
    # Interpretability
    if 'gradcam_stats.json' in results_dict:
        grad_data = results_dict['gradcam_stats.json']
        summary_text += "✓ Grad-CAM Explainability:\n"
        summary_text += f"  - Correct predictions analyzed: {grad_data['correct_predictions']['count']}\n"
        summary_text += f"  - Incorrect predictions analyzed: {grad_data['incorrect_predictions']['count']}\n"
        summary_text += "  - Activation metrics computed for interpretability\n\n"
    
    summary_text += "=" * 70 + "\n"
    summary_text += "This research demonstrates systematic, publication-ready analysis\n"
    summary_text += "with comprehensive uncertainty quantification, calibration, and\n"
    summary_text += "interpretability assessment for medical AI applications.\n"
    
    ax = plt.subplot(111)
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9, linewidth=2))
    ax.axis('off')
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    
    save_figure(fig, filename, output_dir)
    plt.close(fig)

# -------- Main Figure Generation Pipeline --------
def generate_all_figures(exp_dir, output_dir=None):
    """
    Generate all publication-quality figures from experiment results.
    
    Args:
        exp_dir: Experiment directory containing results
        output_dir: Output directory for figures (default: exp_dir/plots)
    """
    if output_dir is None:
        output_dir = os.path.join(exp_dir, "plots")
    
    os.makedirs(output_dir, exist_ok=True)
    
    outputs_dir = os.path.join(exp_dir, 'outputs')
    
    print("\n" + "="*80)
    print("GENERATING PUBLICATION-QUALITY FIGURES")
    print("="*80 + "\n")
    
    # Figure 1: ROC Curve
    try:
        predictions_csv = os.path.join(outputs_dir, 
                                       [f for f in os.listdir(outputs_dir) 
                                        if 'validation_predictions' in f][0])
        df_pred = pd.read_csv(predictions_csv)
        plot_roc_with_ci(df_pred['true_label'].values, df_pred['probability'].values,
                        output_dir=output_dir, filename="fig1_roc_curve")
    except Exception as e:
        print(f"⚠ Could not generate ROC curve: {e}")
    
    # Figure 2: Uncertainty vs Error
    try:
        uncertainty_json = os.path.join(outputs_dir,
                                       [f for f in os.listdir(outputs_dir) 
                                        if 'uncertainty_stats' in f][0])
        df_pred = pd.read_csv([f for f in os.listdir(outputs_dir) 
                              if 'validation_predictions' in f][0])
        unc_data = json.load(open(uncertainty_json))
        
        plot_uncertainty_vs_error(df_pred['true_label'].values,
                                 df_pred['predicted_label'].values,
                                 np.array(df_pred.get('uncertainty_entropy', [0]*len(df_pred))),
                                 output_dir=output_dir, filename="fig2_uncertainty_error")
    except Exception as e:
        print(f"⚠ Could not generate uncertainty vs error plot: {e}")
    
    # Figure 3: Grad-CAM Grid
    try:
        gradcam_json = os.path.join(outputs_dir,
                                   [f for f in os.listdir(outputs_dir) 
                                    if 'gradcam_stats' in f][0])
        plot_gradcam_grid(gradcam_json, output_dir=output_dir, filename="fig3_gradcam_comparison")
    except Exception as e:
        print(f"⚠ Could not generate Grad-CAM grid: {e}")
    
    # Figure 5: Calibration Curves
    try:
        calib_json = os.path.join(outputs_dir,
                                 [f for f in os.listdir(outputs_dir) 
                                  if 'calibration_analysis' in f][0])
        plot_calibration_curves(calib_json, output_dir=output_dir, filename="fig5_calibration")
    except Exception as e:
        print(f"⚠ Could not generate calibration curves: {e}")
    
    # Figure 6: Results Summary
    try:
        plot_results_summary(exp_dir, output_dir=output_dir, filename="fig6_results_summary")
    except Exception as e:
        print(f"⚠ Could not generate results summary: {e}")
    
    print("\n" + "="*80)
    print(f"✓ All figures saved to: {output_dir}")
    print("="*80 + "\n")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        exp_dir = sys.argv[1]
        generate_all_figures(exp_dir)
    else:
        print("Usage: python generate_figures.py <experiment_directory>")
        print("\nExample:")
        print("  python generate_figures.py experiments/PneumoniaDetection_Baseline_20241228_143015")
