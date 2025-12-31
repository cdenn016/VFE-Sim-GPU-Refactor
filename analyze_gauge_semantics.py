#!/usr/bin/env python3
"""
Analyze whether gauge frames φ encode semantic relationships.

SET THESE PATHS, then run: python analyze_gauge_semantics.py
"""

import torch
import numpy as np
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

# =============================================================================
# CONFIGURATION - SET THESE PATHS
# =============================================================================

EXPERIMENT_CONFIG_PATH = "runs/your_experiment/experiment_config.json"  # <-- CHANGE THIS
CHECKPOINT_PATH = "runs/your_experiment/best_model.pt"                   # <-- CHANGE THIS

# =============================================================================
# LOAD TOKENIZER
# =============================================================================

try:
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
    print("Loaded GPT-2 tokenizer")
except ImportError:
    print("ERROR: Install tiktoken: pip install tiktoken")
    sys.exit(1)

# =============================================================================
# LOAD MODEL
# =============================================================================

print(f"\nLoading config: {EXPERIMENT_CONFIG_PATH}")
config_path = Path(EXPERIMENT_CONFIG_PATH)
if config_path.exists():
    with open(config_path) as f:
        config = json.load(f)
    print(f"  embed_dim: {config.get('embed_dim', 'N/A')}")
    print(f"  lambda_beta: {config.get('lambda_beta', 'N/A')}")
else:
    print(f"  WARNING: Config not found")
    config = {}

print(f"\nLoading checkpoint: {CHECKPOINT_PATH}")
ckpt_path = Path(CHECKPOINT_PATH)
if not ckpt_path.exists():
    print(f"ERROR: Checkpoint not found: {ckpt_path}")
    sys.exit(1)

checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
elif 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint

mu_embed = None
phi_embed = None

for key, value in state_dict.items():
    if 'mu_embed' in key and 'weight' in key:
        mu_embed = value
        print(f"  Found mu_embed: {value.shape}")
    if 'phi_embed' in key and 'weight' in key:
        phi_embed = value
        print(f"  Found phi_embed: {value.shape}")

if mu_embed is None:
    print("ERROR: No mu_embed found!")
    sys.exit(1)

if phi_embed is None:
    print("WARNING: No phi_embed found (model may use fixed gauge frames)")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def dist(t1, t2, embed):
    """Euclidean distance between embeddings."""
    if embed is None or t1 >= len(embed) or t2 >= len(embed):
        return float('nan')
    return torch.norm(embed[t1] - embed[t2]).item()


def get_token_id(word):
    """Get token ID if word is a single BPE token."""
    tokens = tokenizer.encode(word)
    if len(tokens) == 1:
        return tokens[0]
    return None

# =============================================================================
# ANALYSIS 1: BPE VOCABULARY CHECK
# =============================================================================

print("\n" + "=" * 60)
print("BPE VOCABULARY CHECK")
print("=" * 60)

test_words = ["cat", "dog", "the", "and", "run", "big", "kitten", "airplane", "happy"]
for word in test_words:
    tokens = tokenizer.encode(word)
    if len(tokens) == 1:
        print(f"  '{word}' -> token {tokens[0]} (single)")
    else:
        decoded = [tokenizer.decode([t]) for t in tokens]
        print(f"  '{word}' -> {decoded} (multi-token)")

# =============================================================================
# ANALYSIS 2: TOKEN CLASS DISTANCES
# =============================================================================

print("\n" + "=" * 60)
print("TOKEN CLASS ANALYSIS")
print("=" * 60)
print("Comparing: letters vs digits vs punctuation")

letter_ids = []
digit_ids = []
punct_ids = []

for tid in range(256):
    try:
        s = tokenizer.decode([tid])
        if len(s) == 1:
            if s.isalpha():
                letter_ids.append(tid)
            elif s.isdigit():
                digit_ids.append(tid)
            elif not s.isalnum() and not s.isspace():
                punct_ids.append(tid)
    except:
        pass

print(f"\nFound: {len(letter_ids)} letters, {len(digit_ids)} digits, {len(punct_ids)} punct")

# Intra-class (letter-letter)
intra_mu, intra_phi = [], []
for i, t1 in enumerate(letter_ids[:10]):
    for t2 in letter_ids[i+1:10]:
        intra_mu.append(dist(t1, t2, mu_embed))
        intra_phi.append(dist(t1, t2, phi_embed))

# Inter-class (letter-digit, letter-punct)
inter_mu, inter_phi = [], []
for t1 in letter_ids[:10]:
    for t2 in digit_ids[:5] + punct_ids[:5]:
        inter_mu.append(dist(t1, t2, mu_embed))
        inter_phi.append(dist(t1, t2, phi_embed))

# Clean NaNs
intra_mu = [x for x in intra_mu if not np.isnan(x)]
intra_phi = [x for x in intra_phi if not np.isnan(x)]
inter_mu = [x for x in inter_mu if not np.isnan(x)]
inter_phi = [x for x in inter_phi if not np.isnan(x)]

print(f"\nmu embeddings:")
print(f"  Intra-class (letter-letter): {np.mean(intra_mu):.4f}")
print(f"  Inter-class (letter-other):  {np.mean(inter_mu):.4f}")
if intra_mu and inter_mu:
    print(f"  Ratio: {np.mean(inter_mu) / np.mean(intra_mu):.2f}x")

if intra_phi and inter_phi:
    print(f"\nphi embeddings (gauge frames):")
    print(f"  Intra-class (letter-letter): {np.mean(intra_phi):.4f}")
    print(f"  Inter-class (letter-other):  {np.mean(inter_phi):.4f}")
    ratio = np.mean(inter_phi) / np.mean(intra_phi)
    print(f"  Ratio: {ratio:.2f}x")

    if ratio > 1.2:
        print(f"\n  --> phi DOES show class structure!")
    else:
        print(f"\n  --> phi does NOT show clear class structure.")

# =============================================================================
# ANALYSIS 3: WORD PAIR DISTANCES
# =============================================================================

print("\n" + "=" * 60)
print("WORD PAIR ANALYSIS (single-token words only)")
print("=" * 60)

pairs = [
    ("cat", "dog", "related"),
    ("cat", "the", "unrelated"),
    ("man", "day", "unrelated"),
    ("big", "new", "unrelated"),
    ("run", "see", "related (verbs)"),
    ("has", "had", "related"),
]

related_phi = []
unrelated_phi = []

for w1, w2, rel in pairs:
    t1 = get_token_id(w1)
    t2 = get_token_id(w2)

    if t1 is None or t2 is None:
        print(f"  {w1:8} - {w2:8}: SKIP (multi-token)")
        continue

    phi_d = dist(t1, t2, phi_embed) if phi_embed is not None else float('nan')
    mu_d = dist(t1, t2, mu_embed)

    print(f"  {w1:8} - {w2:8} [{rel:15}]: phi={phi_d:.4f}, mu={mu_d:.4f}")

    if not np.isnan(phi_d):
        if "related" in rel:
            related_phi.append(phi_d)
        else:
            unrelated_phi.append(phi_d)

if related_phi and unrelated_phi:
    print(f"\nRelated mean:   {np.mean(related_phi):.4f}")
    print(f"Unrelated mean: {np.mean(unrelated_phi):.4f}")
    ratio = np.mean(unrelated_phi) / np.mean(related_phi)
    print(f"Ratio: {ratio:.2f}x")

# =============================================================================
# VISUALIZATION
# =============================================================================

print("\n" + "=" * 60)
print("VISUALIZATION")
print("=" * 60)

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

def categorize_token(tid):
    """Categorize a token by type."""
    try:
        s = tokenizer.decode([tid])
        if len(s) == 1:
            if s.isalpha():
                return 'letter'
            elif s.isdigit():
                return 'digit'
            elif not s.isalnum() and not s.isspace():
                return 'punct'
        if s.strip() in {'the', 'a', 'an', 'is', 'are', 'was', 'of', 'to', 'in', 'for', 'and', 'or'}:
            return 'function'
        return 'content'
    except:
        return 'other'

# Get tokens to visualize (first 500 for speed)
n_viz = min(500, len(phi_embed) if phi_embed is not None else 0)

if phi_embed is not None and n_viz > 0:
    phi_np = phi_embed[:n_viz].numpy()
    phi_dim = phi_np.shape[1]

    # Categorize tokens
    categories = [categorize_token(tid) for tid in range(n_viz)]
    category_colors = {
        'letter': '#E74C3C',    # red
        'digit': '#3498DB',     # blue
        'punct': '#2ECC71',     # green
        'function': '#9B59B6',  # purple
        'content': '#F39C12',   # orange
        'other': '#95A5A6',     # gray
    }
    colors = [category_colors.get(c, '#95A5A6') for c in categories]

    # Identify gauge group from dimension
    # SO(2): 1 generator, SO(3): 3 generators, SO(N): N(N-1)/2 generators
    if phi_dim == 1:
        gauge_str = "SO(2)"
    elif phi_dim == 3:
        gauge_str = "SO(3)"
    else:
        # Solve N(N-1)/2 = phi_dim for N
        n_approx = int((1 + np.sqrt(1 + 8 * phi_dim)) / 2)
        gauge_str = f"SO({n_approx})"
    print(f"phi_dim = {phi_dim} ({gauge_str})")

    # Normalize to unit sphere for visualization
    phi_norms = np.linalg.norm(phi_np, axis=1, keepdims=True)
    phi_norms = np.clip(phi_norms, 1e-8, None)  # avoid div by zero
    phi_unit = phi_np / phi_norms

    if phi_dim == 1:
        # SO(2): 1D gauge frames - show as histogram and jittered scatter
        fig = plt.figure(figsize=(14, 6))

        # Histogram of phi values by category
        ax1 = fig.add_subplot(121)
        for cat in category_colors:
            mask = [c == cat for c in categories]
            if any(mask):
                idx = [i for i, m in enumerate(mask) if m]
                vals = phi_np[idx, 0]
                ax1.hist(vals, bins=30, alpha=0.5, label=cat, color=category_colors[cat])

        ax1.set_xlabel('φ (SO(2) angle)')
        ax1.set_ylabel('Count')
        ax1.set_title('SO(2) Gauge Frame Distribution')
        ax1.legend(loc='upper right', fontsize=8)

        # Jittered scatter plot (add random y for visibility)
        ax2 = fig.add_subplot(122)
        np.random.seed(42)
        for cat in category_colors:
            mask = [c == cat for c in categories]
            if any(mask):
                idx = [i for i, m in enumerate(mask) if m]
                x_vals = phi_np[idx, 0]
                y_jitter = np.random.uniform(-0.4, 0.4, len(idx))
                ax2.scatter(x_vals, y_jitter, c=category_colors[cat], label=cat, alpha=0.6, s=20)

        ax2.set_xlabel('φ (SO(2) angle)')
        ax2.set_ylabel('(jittered for visibility)')
        ax2.set_title('SO(2) Gauge Frames by Token Type')
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.set_ylim(-0.6, 0.6)

    elif phi_dim == 3:
        # SO(3): Direct 3D visualization on sphere
        fig = plt.figure(figsize=(14, 6))

        # 3D sphere plot
        ax1 = fig.add_subplot(121, projection='3d')

        # Draw unit sphere wireframe
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
        ax1.plot_wireframe(x_sphere, y_sphere, z_sphere, alpha=0.1, color='gray')

        # Plot points
        for cat in category_colors:
            mask = [c == cat for c in categories]
            if any(mask):
                idx = [i for i, m in enumerate(mask) if m]
                ax1.scatter(phi_unit[idx, 0], phi_unit[idx, 1], phi_unit[idx, 2],
                           c=category_colors[cat], label=cat, alpha=0.6, s=20)

        ax1.set_xlabel('φ₁')
        ax1.set_ylabel('φ₂')
        ax1.set_zlabel('φ₃')
        ax1.set_title('SO(3) Gauge Frames on Unit Sphere')
        ax1.legend(loc='upper left', fontsize=8)

        # 2D projection (first two components)
        ax2 = fig.add_subplot(122)
        for cat in category_colors:
            mask = [c == cat for c in categories]
            if any(mask):
                idx = [i for i, m in enumerate(mask) if m]
                ax2.scatter(phi_np[idx, 0], phi_np[idx, 1],
                           c=category_colors[cat], label=cat, alpha=0.6, s=20)

        ax2.set_xlabel('φ₁')
        ax2.set_ylabel('φ₂')
        ax2.set_title('SO(3) Gauge Frames (φ₁ vs φ₂)')
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')

    else:
        # SO(N) with N > 3: Use PCA to reduce to 2D/3D
        n_components = min(3, phi_dim)
        print(f"Using PCA to reduce {phi_dim}D -> {n_components}D for visualization")

        pca = PCA(n_components=n_components)
        phi_pca = pca.fit_transform(phi_np)

        var_explained = pca.explained_variance_ratio_
        var_str = " + ".join([f"{v:.1%}" for v in var_explained])
        print(f"PCA variance explained: {var_str}")

        fig = plt.figure(figsize=(14, 6))

        if n_components >= 3:
            # 3D PCA plot
            ax1 = fig.add_subplot(121, projection='3d')
            for cat in category_colors:
                mask = [c == cat for c in categories]
                if any(mask):
                    idx = [i for i, m in enumerate(mask) if m]
                    ax1.scatter(phi_pca[idx, 0], phi_pca[idx, 1], phi_pca[idx, 2],
                               c=category_colors[cat], label=cat, alpha=0.6, s=20)

            ax1.set_xlabel(f'PC1 ({var_explained[0]:.1%})')
            ax1.set_ylabel(f'PC2 ({var_explained[1]:.1%})')
            ax1.set_zlabel(f'PC3 ({var_explained[2]:.1%})')
            ax1.set_title(f'{gauge_str} Gauge Frames (PCA from {phi_dim}D)')
            ax1.legend(loc='upper left', fontsize=8)

            # 2D PCA plot
            ax2 = fig.add_subplot(122)
        elif n_components == 2:
            # Only 2D available
            ax2 = fig.add_subplot(111)
        else:
            # Only 1D - shouldn't happen here but handle gracefully
            ax2 = fig.add_subplot(111)

        if n_components >= 2:
            for cat in category_colors:
                mask = [c == cat for c in categories]
                if any(mask):
                    idx = [i for i, m in enumerate(mask) if m]
                    ax2.scatter(phi_pca[idx, 0], phi_pca[idx, 1],
                               c=category_colors[cat], label=cat, alpha=0.6, s=20)

            ax2.set_xlabel(f'PC1 ({var_explained[0]:.1%})')
            ax2.set_ylabel(f'PC2 ({var_explained[1]:.1%})')
            ax2.set_title(f'{gauge_str} Gauge Frames (PCA from {phi_dim}D)')
            ax2.legend(loc='upper left', fontsize=8)
            ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    save_path = Path(CHECKPOINT_PATH).parent / 'gauge_frame_clustering.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {save_path}")

    plt.show()

else:
    print("No phi embeddings to visualize")

# =============================================================================
# DONE
# =============================================================================

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
