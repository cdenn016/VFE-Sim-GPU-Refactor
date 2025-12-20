"""
Publication Metrics for Gauge-Theoretic VFE Transformer
========================================================

Generates publication-quality figures for the JMLR submission.

Key Figures:
1. Training Curves - Loss, PPL, BPC over training steps
2. Attention Heatmaps - KL-divergence based attention patterns
3. Model Comparison - Bar charts comparing architectures
4. Scaling Study - Performance vs embedding dimension K

Author: Robert C. Dennis
Date: December 2025
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
import json
import csv
import math
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker


# =============================================================================
# Publication Style Settings
# =============================================================================

PUBLICATION_STYLE = {
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'lines.linewidth': 1.5,
    'axes.linewidth': 1.0,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
}


def apply_publication_style():
    """Apply publication-quality matplotlib styling."""
    plt.rcParams.update(PUBLICATION_STYLE)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TrainingSnapshot:
    """Single snapshot of training metrics."""
    step: int
    epoch: float
    train_loss: float
    train_ce: float
    train_ppl: float
    train_bpc: float
    val_loss: Optional[float] = None
    val_ppl: Optional[float] = None
    val_bpc: Optional[float] = None
    grad_norm_total: float = 0.0
    grad_norm_mu: float = 0.0
    grad_norm_sigma: float = 0.0
    grad_norm_phi: float = 0.0
    lr_current: float = 0.0
    tokens_per_sec: float = 0.0
    step_time: float = 0.0
    beta_loss: float = 0.0  # Belief alignment term

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentResult:
    """Results from a single experiment configuration."""
    name: str
    config: Dict[str, Any]
    final_val_ppl: float
    final_val_bpc: float
    best_val_ppl: float
    total_params: int
    training_time: float
    tokens_per_sec: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# Training Tracker
# =============================================================================

class TrainingTracker:
    """
    Track training dynamics for publication figures.

    Records loss, PPL, BPC, gradients over training steps.
    """

    def __init__(self, save_dir: Optional[Path] = None):
        self.save_dir = Path(save_dir) if save_dir else Path("./outputs/figures")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.history: List[TrainingSnapshot] = []

    def record(
        self,
        step: int,
        epoch: float,
        train_metrics: Dict[str, float],
        grad_norms: Optional[Dict[str, float]] = None,
        lr: float = 0.0,
        step_time: float = 0.0,
        batch_size: int = 1,
        seq_len: int = 1,
    ):
        """Record a training step."""
        tokens_per_sec = (batch_size * seq_len) / step_time if step_time > 0 else 0

        train_ce = train_metrics.get('ce_loss', train_metrics.get('ce', 0))
        train_bpc = train_ce / math.log(2) if train_ce > 0 else 0
        train_ppl = math.exp(min(train_ce, 20)) if train_ce > 0 else float('inf')

        snapshot = TrainingSnapshot(
            step=step,
            epoch=epoch,
            train_loss=train_metrics.get('loss', 0),
            train_ce=train_ce,
            train_ppl=train_ppl,
            train_bpc=train_bpc,
            beta_loss=train_metrics.get('beta_loss', train_metrics.get('beta', 0)),
            grad_norm_total=grad_norms.get('total', 0) if grad_norms else 0,
            grad_norm_mu=grad_norms.get('mu', 0) if grad_norms else 0,
            grad_norm_sigma=grad_norms.get('sigma', 0) if grad_norms else 0,
            grad_norm_phi=grad_norms.get('phi', 0) if grad_norms else 0,
            lr_current=lr,
            tokens_per_sec=tokens_per_sec,
            step_time=step_time,
        )

        self.history.append(snapshot)

    def record_validation(self, step: int, val_metrics: Dict[str, float]):
        """Record validation metrics for a given step."""
        for snapshot in reversed(self.history):
            if snapshot.step == step:
                val_ce = val_metrics.get('ce_loss', val_metrics.get('loss', 0))
                snapshot.val_loss = val_metrics.get('loss', val_ce)
                snapshot.val_ppl = val_metrics.get('perplexity',
                    math.exp(min(val_ce, 20)) if val_ce > 0 else float('inf'))
                snapshot.val_bpc = val_ce / math.log(2) if val_ce > 0 else None
                break

    def get_summary(self) -> Dict[str, Any]:
        """Get training summary statistics."""
        if not self.history:
            return {}

        val_ppls = [s.val_ppl for s in self.history if s.val_ppl is not None]
        val_bpcs = [s.val_bpc for s in self.history if s.val_bpc is not None]

        return {
            'total_steps': len(self.history),
            'final_train_loss': self.history[-1].train_loss,
            'final_train_ppl': self.history[-1].train_ppl,
            'final_train_bpc': self.history[-1].train_bpc,
            'final_val_ppl': val_ppls[-1] if val_ppls else None,
            'final_val_bpc': val_bpcs[-1] if val_bpcs else None,
            'best_val_ppl': min(val_ppls) if val_ppls else None,
            'best_val_bpc': min(val_bpcs) if val_bpcs else None,
            'avg_tokens_per_sec': np.mean([s.tokens_per_sec for s in self.history]),
            'total_time_sec': sum([s.step_time for s in self.history]),
        }

    def save_json(self, filename: str = "training_history.json"):
        """Save history to JSON."""
        data = {
            'summary': self.get_summary(),
            'history': [s.to_dict() for s in self.history],
        }
        with open(self.save_dir / filename, 'w') as f:
            json.dump(data, f, indent=2)

    def save_csv(self, filename: str = "training_history.csv"):
        """Save history to CSV."""
        if not self.history:
            return
        with open(self.save_dir / filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(self.history[0].to_dict().keys()))
            writer.writeheader()
            for snapshot in self.history:
                writer.writerow(snapshot.to_dict())


# =============================================================================
# Publication Figure Generator
# =============================================================================

class PublicationFigures:
    """
    Generate publication-quality figures for gauge VFE transformer paper.
    """

    def __init__(self, save_dir: Optional[Path] = None):
        self.save_dir = Path(save_dir) if save_dir else Path("./outputs/figures")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        apply_publication_style()

    def plot_training_curves(
        self,
        tracker: TrainingTracker,
        save_name: str = "training_curves",
        show_components: bool = True,
    ) -> plt.Figure:
        """
        Plot training curves: Loss, PPL, BPC.

        Args:
            tracker: TrainingTracker with recorded history
            save_name: Filename for saved figure
            show_components: If True, show loss decomposition

        Returns:
            matplotlib Figure
        """
        if not tracker.history:
            raise ValueError("No training history to plot")

        history = tracker.history
        steps = [s.step for s in history]

        # Validation data
        val_steps = [s.step for s in history if s.val_ppl is not None]
        val_ppls = [s.val_ppl for s in history if s.val_ppl is not None]
        val_bpcs = [s.val_bpc for s in history if s.val_bpc is not None]

        if show_components:
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        else:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes = axes.reshape(1, -1)

        # (a) Loss curves
        ax = axes[0, 0] if show_components else axes[0, 0]
        ax.plot(steps, [s.train_loss for s in history], 'b-',
                label='Train', alpha=0.7, linewidth=1)
        if val_steps:
            ax.plot(val_steps, [s.val_loss for s in history if s.val_loss],
                    'r-', label='Val', linewidth=2)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss')
        ax.set_title('(a) Total Loss')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # (b) Perplexity
        ax = axes[0, 1] if show_components else axes[0, 1]
        ax.semilogy(steps, [s.train_ppl for s in history], 'b-',
                    label='Train', alpha=0.7, linewidth=1)
        if val_ppls:
            ax.semilogy(val_steps, val_ppls, 'r-', label='Val', linewidth=2)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Perplexity')
        ax.set_title('(b) Perplexity')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # (c) BPC
        ax = axes[1, 0] if show_components else axes[0, 2]
        ax.plot(steps, [s.train_bpc for s in history], 'b-',
                label='Train', alpha=0.7, linewidth=1)
        if val_bpcs:
            ax.plot(val_steps, val_bpcs, 'r-', label='Val', linewidth=2)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Bits per Token')
        ax.set_title('(c) Bits per Token')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        if show_components:
            # (d) Gradient norms
            ax = axes[1, 1]
            ax.semilogy(steps, [s.grad_norm_total for s in history],
                        'k-', label='Total', alpha=0.8, linewidth=1.5)
            ax.semilogy(steps, [s.grad_norm_mu for s in history],
                        'b--', label='μ', alpha=0.6, linewidth=1)
            ax.semilogy(steps, [s.grad_norm_sigma for s in history],
                        'g--', label='Σ', alpha=0.6, linewidth=1)
            ax.semilogy(steps, [s.grad_norm_phi for s in history],
                        'r--', label='φ', alpha=0.6, linewidth=1)
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Gradient Norm')
            ax.set_title('(d) Gradient Norms')
            ax.legend(loc='upper right', ncol=2)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save
        fig.savefig(self.save_dir / f"{save_name}.pdf")
        fig.savefig(self.save_dir / f"{save_name}.png", dpi=300)

        return fig

    def plot_attention_heatmap(
        self,
        beta: torch.Tensor,
        tokens: Optional[List[str]] = None,
        save_name: str = "attention_heatmap",
        title: str = "KL-Divergence Attention Weights",
        head_idx: Optional[int] = None,
    ) -> plt.Figure:
        """
        Plot attention heatmap from KL-divergence based attention.

        Args:
            beta: Attention weights (B, N, N) or (B, H, N, N)
            tokens: Optional token labels
            save_name: Filename for saved figure
            title: Figure title
            head_idx: If multi-head, which head to plot (None = average)

        Returns:
            matplotlib Figure
        """
        # Handle multi-head attention
        if beta.dim() == 4:  # (B, H, N, N)
            if head_idx is not None:
                attn = beta[0, head_idx].detach().cpu().numpy()
                title = f"{title} (Head {head_idx})"
            else:
                attn = beta[0].mean(dim=0).detach().cpu().numpy()
                title = f"{title} (Averaged)"
        else:  # (B, N, N)
            attn = beta[0].detach().cpu().numpy()

        N = attn.shape[0]

        fig, ax = plt.subplots(figsize=(8, 7))

        # Use log scale for better visualization of attention distribution
        im = ax.imshow(attn, cmap='Blues', aspect='auto')

        # Labels
        if tokens is not None and len(tokens) <= 32:
            tokens_display = [t[:10] for t in tokens]  # Truncate long tokens
            ax.set_xticks(range(len(tokens_display)))
            ax.set_yticks(range(len(tokens_display)))
            ax.set_xticklabels(tokens_display, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(tokens_display, fontsize=8)
        else:
            # Show every 10th tick for long sequences
            tick_spacing = max(1, N // 10)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

        ax.set_xlabel('Key Position (j)')
        ax.set_ylabel('Query Position (i)')
        ax.set_title(title)

        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('β_ij')

        plt.tight_layout()

        fig.savefig(self.save_dir / f"{save_name}.pdf")
        fig.savefig(self.save_dir / f"{save_name}.png", dpi=300)

        return fig

    def plot_model_comparison(
        self,
        results: List[ExperimentResult],
        save_name: str = "model_comparison",
        metric: str = "ppl",
    ) -> plt.Figure:
        """
        Plot bar chart comparing different model architectures.

        Args:
            results: List of ExperimentResult from different configs
            save_name: Filename for saved figure
            metric: 'ppl' or 'bpc'

        Returns:
            matplotlib Figure
        """
        if not results:
            raise ValueError("No results to plot")

        names = [r.name for r in results]

        if metric == "ppl":
            values = [r.final_val_ppl for r in results]
            ylabel = "Validation Perplexity"
            title = "Model Comparison: Perplexity"
        else:
            values = [r.final_val_bpc for r in results]
            ylabel = "Validation BPC"
            title = "Model Comparison: Bits per Token"

        # Color scheme
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(results)]

        fig, ax = plt.subplots(figsize=(8, 5))

        bars = ax.bar(range(len(names)), values, color=colors, edgecolor='black', linewidth=1)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)

        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=15, ha='right')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')

        # Add parameter count annotation
        param_text = '\n'.join([f"{r.name}: {r.total_params/1000:.1f}K params" for r in results])
        ax.text(0.98, 0.98, param_text, transform=ax.transAxes,
                fontsize=8, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        fig.savefig(self.save_dir / f"{save_name}.pdf")
        fig.savefig(self.save_dir / f"{save_name}.png", dpi=300)

        return fig

    def plot_scaling_study(
        self,
        k_values: List[int],
        ppl_values: List[float],
        param_counts: Optional[List[int]] = None,
        save_name: str = "scaling_study",
    ) -> plt.Figure:
        """
        Plot scaling study: PPL vs embedding dimension K.

        Args:
            k_values: List of embedding dimensions tested
            ppl_values: Corresponding perplexity values
            param_counts: Optional parameter counts
            save_name: Filename for saved figure

        Returns:
            matplotlib Figure
        """
        fig, ax1 = plt.subplots(figsize=(8, 5))

        # PPL vs K
        color = 'tab:blue'
        ax1.set_xlabel('Embedding Dimension (K)')
        ax1.set_ylabel('Validation Perplexity', color=color)
        line1 = ax1.plot(k_values, ppl_values, 'o-', color=color,
                         linewidth=2, markersize=8, label='PPL')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, alpha=0.3)

        # Mark optimal K
        min_idx = np.argmin(ppl_values)
        ax1.axvline(x=k_values[min_idx], color='red', linestyle='--',
                   alpha=0.5, label=f'Optimal K={k_values[min_idx]}')
        ax1.scatter([k_values[min_idx]], [ppl_values[min_idx]],
                   color='red', s=150, zorder=5, marker='*')

        # Add parameter counts on secondary axis
        if param_counts:
            ax2 = ax1.twinx()
            color = 'tab:gray'
            ax2.set_ylabel('Parameters (K)', color=color)
            line2 = ax2.plot(k_values, [p/1000 for p in param_counts],
                            's--', color=color, alpha=0.6, linewidth=1,
                            markersize=5, label='Params')
            ax2.tick_params(axis='y', labelcolor=color)

        ax1.set_title('Scaling Study: Performance vs Embedding Dimension')
        ax1.legend(loc='upper right')

        plt.tight_layout()

        fig.savefig(self.save_dir / f"{save_name}.pdf")
        fig.savefig(self.save_dir / f"{save_name}.png", dpi=300)

        return fig

    def plot_train_val_gap(
        self,
        tracker: TrainingTracker,
        save_name: str = "train_val_gap",
    ) -> plt.Figure:
        """
        Plot train-validation gap over training.

        Shows generalization behavior - gap should stay small.

        Args:
            tracker: TrainingTracker with recorded history
            save_name: Filename for saved figure

        Returns:
            matplotlib Figure
        """
        history = tracker.history

        # Get steps where we have both train and val
        val_steps = []
        gaps_bpc = []

        for s in history:
            if s.val_bpc is not None:
                val_steps.append(s.step)
                gaps_bpc.append(s.val_bpc - s.train_bpc)

        if not val_steps:
            raise ValueError("No validation data to plot")

        fig, ax = plt.subplots(figsize=(8, 4))

        ax.plot(val_steps, gaps_bpc, 'b-o', linewidth=1.5, markersize=4)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.fill_between(val_steps, gaps_bpc, 0, alpha=0.3,
                        color='green' if np.mean(gaps_bpc) < 0 else 'red')

        ax.set_xlabel('Training Step')
        ax.set_ylabel('Val BPC - Train BPC')
        ax.set_title('Generalization Gap')
        ax.grid(True, alpha=0.3)

        # Add annotation
        final_gap = gaps_bpc[-1]
        ax.annotate(f'Final gap: {final_gap:+.3f}',
                   xy=(val_steps[-1], final_gap),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=10, ha='left')

        plt.tight_layout()

        fig.savefig(self.save_dir / f"{save_name}.pdf")
        fig.savefig(self.save_dir / f"{save_name}.png", dpi=300)

        return fig


# =============================================================================
# Main Publication Metrics Coordinator
# =============================================================================

class PublicationMetrics:
    """
    Main coordinator for publication metrics and figures.

    Usage:
        metrics = PublicationMetrics("wikitext103_experiment")

        # During training
        for step in range(max_steps):
            metrics.record_step(step, epoch, train_metrics, grad_norms)
            if step % eval_interval == 0:
                metrics.record_validation(step, val_metrics)

        # After training
        metrics.save_all()
        metrics.generate_figures()
    """

    def __init__(self, experiment_name: str, base_dir: Optional[Path] = None):
        self.experiment_name = experiment_name
        self.base_dir = Path(base_dir) if base_dir else Path("./outputs")
        self.experiment_dir = self.base_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self.tracker = TrainingTracker(self.experiment_dir)
        self.figures = PublicationFigures(self.experiment_dir / "figures")

        self.comparison_results: List[ExperimentResult] = []
        self.scaling_data: Dict[str, List] = {'k': [], 'ppl': [], 'params': []}

        print(f"[PublicationMetrics] Initialized: {self.experiment_dir}")

    def record_step(
        self,
        step: int,
        epoch: float,
        train_metrics: Dict[str, float],
        grad_norms: Optional[Dict[str, float]] = None,
        lr: float = 0.0,
        step_time: float = 0.0,
        batch_size: int = 1,
        seq_len: int = 1,
    ):
        """Record training step metrics."""
        self.tracker.record(step, epoch, train_metrics, grad_norms,
                           lr, step_time, batch_size, seq_len)

    def record_validation(self, step: int, val_metrics: Dict[str, float]):
        """Record validation metrics."""
        self.tracker.record_validation(step, val_metrics)

    def add_comparison_result(self, result: ExperimentResult):
        """Add a result for model comparison."""
        self.comparison_results.append(result)

    def add_scaling_point(self, k: int, ppl: float, params: int):
        """Add a point for scaling study."""
        self.scaling_data['k'].append(k)
        self.scaling_data['ppl'].append(ppl)
        self.scaling_data['params'].append(params)

    def save_all(self):
        """Save all metrics to files."""
        self.tracker.save_json()
        self.tracker.save_csv()

        # Save comparison results
        if self.comparison_results:
            with open(self.experiment_dir / "comparison_results.json", 'w') as f:
                json.dump([r.to_dict() for r in self.comparison_results], f, indent=2)

        # Save scaling data
        if self.scaling_data['k']:
            with open(self.experiment_dir / "scaling_data.json", 'w') as f:
                json.dump(self.scaling_data, f, indent=2)

        print(f"[PublicationMetrics] Saved to {self.experiment_dir}")

    def generate_figures(self, attention_weights: Optional[torch.Tensor] = None):
        """Generate all publication figures."""
        figures_generated = []

        # Training curves
        try:
            self.figures.plot_training_curves(self.tracker)
            figures_generated.append("training_curves")
            plt.close()
        except Exception as e:
            print(f"[WARN] Could not generate training curves: {e}")

        # Train-val gap
        try:
            self.figures.plot_train_val_gap(self.tracker)
            figures_generated.append("train_val_gap")
            plt.close()
        except Exception as e:
            print(f"[WARN] Could not generate train-val gap: {e}")

        # Model comparison
        if self.comparison_results:
            try:
                self.figures.plot_model_comparison(self.comparison_results)
                figures_generated.append("model_comparison")
                plt.close()
            except Exception as e:
                print(f"[WARN] Could not generate model comparison: {e}")

        # Scaling study
        if self.scaling_data['k']:
            try:
                self.figures.plot_scaling_study(
                    self.scaling_data['k'],
                    self.scaling_data['ppl'],
                    self.scaling_data['params']
                )
                figures_generated.append("scaling_study")
                plt.close()
            except Exception as e:
                print(f"[WARN] Could not generate scaling study: {e}")

        # Attention heatmap
        if attention_weights is not None:
            try:
                self.figures.plot_attention_heatmap(attention_weights)
                figures_generated.append("attention_heatmap")
                plt.close()
            except Exception as e:
                print(f"[WARN] Could not generate attention heatmap: {e}")

        print(f"[PublicationMetrics] Generated figures: {', '.join(figures_generated)}")

    def print_summary(self):
        """Print experiment summary."""
        summary = self.tracker.get_summary()

        print("\n" + "=" * 60)
        print(f"EXPERIMENT: {self.experiment_name}")
        print("=" * 60)

        if summary:
            print(f"Total Steps:     {summary.get('total_steps', 0)}")
            print(f"Final Train PPL: {summary.get('final_train_ppl', 0):.2f}")
            print(f"Final Val PPL:   {summary.get('final_val_ppl', 'N/A')}")
            print(f"Best Val PPL:    {summary.get('best_val_ppl', 'N/A')}")
            print(f"Final Train BPC: {summary.get('final_train_bpc', 0):.4f}")
            print(f"Final Val BPC:   {summary.get('final_val_bpc', 'N/A')}")
            print(f"Throughput:      {summary.get('avg_tokens_per_sec', 0):.0f} tok/s")
            print(f"Total Time:      {summary.get('total_time_sec', 0)/60:.1f} min")

        print("=" * 60)


# =============================================================================
# Convenience Functions
# =============================================================================

def create_results_table(results: List[ExperimentResult]) -> str:
    """Create LaTeX table from experiment results."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Experimental Results on WikiText-103}",
        r"\label{tab:results}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{PPL} $\downarrow$ & \textbf{BPC} $\downarrow$ & \textbf{Params} \\",
        r"\midrule",
    ]

    for r in results:
        lines.append(
            f"{r.name} & {r.final_val_ppl:.1f} & {r.final_val_bpc:.3f} & {r.total_params/1000:.1f}K \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Publication Metrics - Demo")
    print("=" * 60)

    # Create metrics tracker
    metrics = PublicationMetrics("demo_experiment")

    # Simulate training
    np.random.seed(42)
    for step in range(500):
        loss = 8.0 * np.exp(-step / 200) + 5.5 + np.random.randn() * 0.1
        train_metrics = {
            'loss': loss,
            'ce_loss': loss - 0.1,
            'beta_loss': 0.1,
        }
        grad_norms = {
            'total': 2.0 / (step + 1) + 0.1,
            'mu': 1.0 / (step + 1) + 0.05,
            'sigma': 0.5 / (step + 1) + 0.02,
            'phi': 0.3 / (step + 1) + 0.01,
        }

        metrics.record_step(
            step=step,
            epoch=step / 100,
            train_metrics=train_metrics,
            grad_norms=grad_norms,
            lr=0.001,
            step_time=0.5,
            batch_size=32,
            seq_len=128,
        )

        if step % 50 == 0:
            val_loss = loss + 0.2 + np.random.randn() * 0.05
            metrics.record_validation(step, {'loss': val_loss, 'ce_loss': val_loss})

    # Add comparison results
    metrics.add_comparison_result(ExperimentResult(
        name="Standard Transformer",
        config={},
        final_val_ppl=350.0,
        final_val_bpc=8.45,
        best_val_ppl=340.0,
        total_params=500000,
        training_time=3600,
        tokens_per_sec=8000,
    ))
    metrics.add_comparison_result(ExperimentResult(
        name="Gauge VFE",
        config={},
        final_val_ppl=300.0,
        final_val_bpc=8.23,
        best_val_ppl=290.0,
        total_params=450000,
        training_time=5400,
        tokens_per_sec=5000,
    ))

    # Add scaling data
    for k in [7, 11, 25, 37, 63]:
        ppl = 250 + 50 * abs(k - 11) / 10 + np.random.randn() * 10
        metrics.add_scaling_point(k, ppl, k * 50257 + k * k * 100)

    # Save and generate
    metrics.save_all()
    metrics.generate_figures()
    metrics.print_summary()

    print("\n Demo complete! Check ./outputs/demo_experiment/")
