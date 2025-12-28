#!/usr/bin/env python3
"""
Analyze whether gauge frames φ encode semantic relationships.

Hypothesis: Semantically related tokens have similar φ (gauge frames),
so transport Ω_ij ≈ I between them.

Tests:
1. Do φ embeddings cluster semantically (like μ does)?
2. Is ||φ_cat - φ_cute|| < ||φ_cat - φ_airplane||?
3. Compare φ distances for related vs unrelated word pairs.
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False
    print("tiktoken not available, using manual token IDs")


def load_model_embeddings(checkpoint_path: str):
    """Load μ and φ embeddings from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Try different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Extract embeddings
    mu_embed = None
    phi_embed = None

    for key, value in state_dict.items():
        if 'mu_embed' in key and 'weight' in key:
            mu_embed = value
            print(f"Found μ embeddings: {key}, shape {value.shape}")
        if 'phi_embed' in key and 'weight' in key:
            phi_embed = value
            print(f"Found φ embeddings: {key}, shape {value.shape}")

    return mu_embed, phi_embed


def get_token_id(word: str, tokenizer=None):
    """Get BPE token ID for a word."""
    if tokenizer is not None:
        tokens = tokenizer.encode(word)
        return tokens[0] if tokens else None
    return None


def analyze_semantic_distances(mu_embed, phi_embed, tokenizer=None):
    """Compare distances for semantically related vs unrelated pairs."""

    # Define test pairs
    # Format: (word1, word2, relationship)
    related_pairs = [
        ("cat", "kitten", "same concept"),
        ("cat", "dog", "both animals"),
        ("cat", "furry", "attribute"),
        ("happy", "joy", "synonyms"),
        ("run", "running", "same verb"),
        ("big", "large", "synonyms"),
        ("hot", "cold", "antonyms but same domain"),
    ]

    unrelated_pairs = [
        ("cat", "airplane", "unrelated"),
        ("cat", "democracy", "unrelated"),
        ("happy", "concrete", "unrelated"),
        ("run", "purple", "unrelated"),
        ("big", "Wednesday", "unrelated"),
    ]

    print("\n" + "="*70)
    print("SEMANTIC DISTANCE ANALYSIS")
    print("="*70)

    if tokenizer is None:
        print("\nNo tokenizer available. Using GPT-2 encoding...")
        if HAS_TIKTOKEN:
            tokenizer = tiktoken.get_encoding("gpt2")
        else:
            print("Cannot proceed without tiktoken. Install with: pip install tiktoken")
            return

    def get_embeddings(word):
        """Get μ and φ for a word."""
        tokens = tokenizer.encode(word)
        if not tokens:
            return None, None
        token_id = tokens[0]
        if token_id >= len(mu_embed):
            return None, None
        return mu_embed[token_id], phi_embed[token_id] if phi_embed is not None else None

    def compute_distance(embed1, embed2):
        """Euclidean distance between embeddings."""
        if embed1 is None or embed2 is None:
            return float('nan')
        return torch.norm(embed1 - embed2).item()

    print("\n--- Related Pairs ---")
    related_mu_dists = []
    related_phi_dists = []

    for word1, word2, relation in related_pairs:
        mu1, phi1 = get_embeddings(word1)
        mu2, phi2 = get_embeddings(word2)

        mu_dist = compute_distance(mu1, mu2)
        phi_dist = compute_distance(phi1, phi2)

        if not np.isnan(mu_dist):
            related_mu_dists.append(mu_dist)
        if not np.isnan(phi_dist):
            related_phi_dists.append(phi_dist)

        print(f"  {word1:12} - {word2:12} ({relation:20}): μ_dist={mu_dist:.4f}, φ_dist={phi_dist:.4f}")

    print("\n--- Unrelated Pairs ---")
    unrelated_mu_dists = []
    unrelated_phi_dists = []

    for word1, word2, relation in unrelated_pairs:
        mu1, phi1 = get_embeddings(word1)
        mu2, phi2 = get_embeddings(word2)

        mu_dist = compute_distance(mu1, mu2)
        phi_dist = compute_distance(phi1, phi2)

        if not np.isnan(mu_dist):
            unrelated_mu_dists.append(mu_dist)
        if not np.isnan(phi_dist):
            unrelated_phi_dists.append(phi_dist)

        print(f"  {word1:12} - {word2:12} ({relation:20}): μ_dist={mu_dist:.4f}, φ_dist={phi_dist:.4f}")

    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    if related_mu_dists and unrelated_mu_dists:
        print(f"\nμ embeddings:")
        print(f"  Related pairs mean distance:   {np.mean(related_mu_dists):.4f}")
        print(f"  Unrelated pairs mean distance: {np.mean(unrelated_mu_dists):.4f}")
        print(f"  Ratio (unrelated/related):     {np.mean(unrelated_mu_dists)/np.mean(related_mu_dists):.2f}x")

    if related_phi_dists and unrelated_phi_dists:
        print(f"\nφ embeddings (gauge frames):")
        print(f"  Related pairs mean distance:   {np.mean(related_phi_dists):.4f}")
        print(f"  Unrelated pairs mean distance: {np.mean(unrelated_phi_dists):.4f}")
        print(f"  Ratio (unrelated/related):     {np.mean(unrelated_phi_dists)/np.mean(related_phi_dists):.2f}x")

        if np.mean(unrelated_phi_dists) > np.mean(related_phi_dists) * 1.2:
            print("\n  ✓ φ encodes semantic relationships!")
            print("    Related tokens have similar gauge frames.")
        else:
            print("\n  ✗ φ does NOT clearly encode semantic relationships.")
            print("    Gauge frames may serve a different purpose.")

    return {
        'related_mu': related_mu_dists,
        'unrelated_mu': unrelated_mu_dists,
        'related_phi': related_phi_dists,
        'unrelated_phi': unrelated_phi_dists,
    }


def analyze_phi_clustering(phi_embed, tokenizer=None):
    """Check if φ embeddings cluster by token type (letters, digits, punctuation)."""

    print("\n" + "="*70)
    print("φ CLUSTERING BY TOKEN TYPE")
    print("="*70)

    if tokenizer is None and HAS_TIKTOKEN:
        tokenizer = tiktoken.get_encoding("gpt2")

    if tokenizer is None:
        print("No tokenizer available")
        return

    # Categorize tokens
    categories = {
        'lowercase': [],
        'uppercase': [],
        'digits': [],
        'punctuation': [],
        'space_tokens': [],
    }

    for token_id in range(min(1000, len(phi_embed))):  # Check first 1000 tokens
        try:
            token_str = tokenizer.decode([token_id])

            if token_str.strip().isalpha() and token_str.islower():
                categories['lowercase'].append(token_id)
            elif token_str.strip().isalpha() and token_str.isupper():
                categories['uppercase'].append(token_id)
            elif token_str.strip().isdigit():
                categories['digits'].append(token_id)
            elif token_str.strip() and not token_str.strip().isalnum():
                categories['punctuation'].append(token_id)
            elif token_str.startswith(' '):
                categories['space_tokens'].append(token_id)
        except:
            continue

    # Compute mean φ for each category
    print("\nCategory statistics:")
    category_means = {}

    for cat_name, token_ids in categories.items():
        if len(token_ids) > 5:
            cat_phis = phi_embed[token_ids]
            cat_mean = cat_phis.mean(dim=0)
            cat_std = cat_phis.std(dim=0).mean()
            category_means[cat_name] = cat_mean

            print(f"  {cat_name:15}: n={len(token_ids):4}, "
                  f"mean_φ={cat_mean.numpy()}, "
                  f"intra_std={cat_std:.4f}")

    # Compute inter-category distances
    print("\nInter-category φ distances:")
    cat_names = list(category_means.keys())

    for i, cat1 in enumerate(cat_names):
        for cat2 in cat_names[i+1:]:
            dist = torch.norm(category_means[cat1] - category_means[cat2]).item()
            print(f"  {cat1:15} - {cat2:15}: {dist:.4f}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze gauge frame semantics")
    parser.add_argument("checkpoint", nargs="?", help="Path to model checkpoint")
    parser.add_argument("--random", action="store_true", help="Test with random embeddings")
    args = parser.parse_args()

    if args.random:
        print("Testing with RANDOM embeddings (baseline)...")
        print("If φ encodes semantics, trained model should show different pattern.\n")

        vocab_size = 50257
        embed_dim = 32
        phi_dim = 3

        mu_embed = torch.randn(vocab_size, embed_dim)
        phi_embed = torch.randn(vocab_size, phi_dim) * 0.1

    elif args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        mu_embed, phi_embed = load_model_embeddings(args.checkpoint)

        if mu_embed is None:
            print("ERROR: Could not find μ embeddings in checkpoint")
            return
        if phi_embed is None:
            print("WARNING: No φ embeddings found (model may not use learnable_phi)")

    else:
        print("Usage:")
        print("  python analyze_gauge_semantics.py <checkpoint.pt>")
        print("  python analyze_gauge_semantics.py --random  # baseline test")
        return

    # Run analyses
    if phi_embed is not None:
        analyze_semantic_distances(mu_embed, phi_embed)
        analyze_phi_clustering(phi_embed)
    else:
        print("\nNo φ embeddings to analyze. Model may use fixed gauge frames.")
        print("Analyzing μ embeddings only...")
        analyze_semantic_distances(mu_embed, None)


if __name__ == "__main__":
    main()
