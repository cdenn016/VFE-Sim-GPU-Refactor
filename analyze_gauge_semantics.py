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
# DONE
# =============================================================================

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
