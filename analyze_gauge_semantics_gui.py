#!/usr/bin/env python3
"""
GUI tool to analyze whether gauge frames φ encode semantic relationships.

Click to run, input paths to experiment_config.json and best_model.pt
"""

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import torch
import numpy as np
import json
from pathlib import Path
import sys
import threading

sys.path.insert(0, str(Path(__file__).parent))

try:
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
except ImportError:
    tokenizer = None


class GaugeAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Gauge Frame Semantic Analysis")
        self.root.geometry("800x700")

        # Variables
        self.config_path = tk.StringVar()
        self.checkpoint_path = tk.StringVar()

        self.mu_embed = None
        self.phi_embed = None

        self._build_ui()

    def _build_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Config path
        ttk.Label(main_frame, text="Experiment Config:").grid(row=0, column=0, sticky="w", pady=5)
        config_frame = ttk.Frame(main_frame)
        config_frame.grid(row=0, column=1, sticky="ew", pady=5)
        config_frame.columnconfigure(0, weight=1)

        ttk.Entry(config_frame, textvariable=self.config_path, width=60).grid(row=0, column=0, sticky="ew")
        ttk.Button(config_frame, text="Browse...", command=self._browse_config).grid(row=0, column=1, padx=5)

        # Checkpoint path
        ttk.Label(main_frame, text="Model Checkpoint:").grid(row=1, column=0, sticky="w", pady=5)
        ckpt_frame = ttk.Frame(main_frame)
        ckpt_frame.grid(row=1, column=1, sticky="ew", pady=5)
        ckpt_frame.columnconfigure(0, weight=1)

        ttk.Entry(ckpt_frame, textvariable=self.checkpoint_path, width=60).grid(row=0, column=0, sticky="ew")
        ttk.Button(ckpt_frame, text="Browse...", command=self._browse_checkpoint).grid(row=0, column=1, padx=5)

        main_frame.columnconfigure(1, weight=1)

        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=2, column=0, columnspan=2, pady=15)

        ttk.Button(btn_frame, text="Load Model", command=self._load_model).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Run Analysis", command=self._run_analysis).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Clear", command=self._clear_output).pack(side="left", padx=5)

        # Output
        ttk.Label(main_frame, text="Results:").grid(row=3, column=0, sticky="nw", pady=5)

        self.output = scrolledtext.ScrolledText(main_frame, width=90, height=30, font=("Courier", 10))
        self.output.grid(row=4, column=0, columnspan=2, sticky="nsew", pady=5)
        main_frame.rowconfigure(4, weight=1)

    def _browse_config(self):
        path = filedialog.askopenfilename(
            title="Select experiment_config.json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if path:
            self.config_path.set(path)

    def _browse_checkpoint(self):
        path = filedialog.askopenfilename(
            title="Select best_model.pt",
            filetypes=[("PyTorch files", "*.pt"), ("All files", "*.*")]
        )
        if path:
            self.checkpoint_path.set(path)

    def _log(self, msg):
        self.output.insert(tk.END, msg + "\n")
        self.output.see(tk.END)
        self.root.update()

    def _clear_output(self):
        self.output.delete(1.0, tk.END)

    def _load_model(self):
        self._clear_output()

        # Load config
        config_path = Path(self.config_path.get())
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            self._log(f"Loaded config: {config_path}")
            self._log(f"  embed_dim: {config.get('embed_dim', 'N/A')}")
            self._log(f"  lambda_beta: {config.get('lambda_beta', 'N/A')}")
        else:
            self._log(f"Config not found: {config_path}")

        # Load checkpoint
        ckpt_path = Path(self.checkpoint_path.get())
        if ckpt_path.exists():
            self._log(f"\nLoading checkpoint: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location='cpu')

            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            self.mu_embed = None
            self.phi_embed = None

            for key, value in state_dict.items():
                if 'mu_embed' in key and 'weight' in key:
                    self.mu_embed = value
                    self._log(f"  Found mu_embed: {value.shape}")
                if 'phi_embed' in key and 'weight' in key:
                    self.phi_embed = value
                    self._log(f"  Found phi_embed: {value.shape}")

            if self.mu_embed is None:
                self._log("  WARNING: No mu_embed found!")
            if self.phi_embed is None:
                self._log("  WARNING: No phi_embed found!")

            self._log("\nModel loaded! Click 'Run Analysis' to analyze.")
        else:
            self._log(f"Checkpoint not found: {ckpt_path}")

    def _run_analysis(self):
        if self.mu_embed is None:
            self._log("ERROR: Load model first!")
            return

        if tokenizer is None:
            self._log("ERROR: tiktoken not installed. Run: pip install tiktoken")
            return

        self._log("\n" + "=" * 60)
        self._log("GAUGE FRAME SEMANTIC ANALYSIS")
        self._log("=" * 60)

        # Run in thread to keep UI responsive
        thread = threading.Thread(target=self._do_analysis)
        thread.start()

    def _do_analysis(self):
        self._analyze_bpe_vocab()
        self._analyze_token_classes()
        self._analyze_word_pairs()

    def _analyze_bpe_vocab(self):
        self._log("\n--- BPE VOCABULARY CHECK ---")

        test_words = ["cat", "dog", "the", "and", "run", "big",
                      "kitten", "airplane", "happy", "running"]

        for word in test_words:
            tokens = tokenizer.encode(word)
            if len(tokens) == 1:
                self._log(f"  '{word}' -> token {tokens[0]} (single)")
            else:
                decoded = [tokenizer.decode([t]) for t in tokens]
                self._log(f"  '{word}' -> {tokens} = {decoded} (multi)")

    def _analyze_token_classes(self):
        self._log("\n--- TOKEN CLASS ANALYSIS ---")
        self._log("Comparing: letters vs digits vs punctuation")

        # Find single-char tokens
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

        self._log(f"\nFound: {len(letter_ids)} letters, {len(digit_ids)} digits, {len(punct_ids)} punct")

        def dist(tid1, tid2, embed):
            if embed is None or tid1 >= len(embed) or tid2 >= len(embed):
                return float('nan')
            return torch.norm(embed[tid1] - embed[tid2]).item()

        # Intra-class (letter-letter)
        intra_mu, intra_phi = [], []
        for i, t1 in enumerate(letter_ids[:10]):
            for t2 in letter_ids[i+1:10]:
                intra_mu.append(dist(t1, t2, self.mu_embed))
                intra_phi.append(dist(t1, t2, self.phi_embed))

        # Inter-class (letter-digit, letter-punct)
        inter_mu, inter_phi = [], []
        for t1 in letter_ids[:10]:
            for t2 in digit_ids[:5] + punct_ids[:5]:
                inter_mu.append(dist(t1, t2, self.mu_embed))
                inter_phi.append(dist(t1, t2, self.phi_embed))

        intra_mu = [x for x in intra_mu if not np.isnan(x)]
        intra_phi = [x for x in intra_phi if not np.isnan(x)]
        inter_mu = [x for x in inter_mu if not np.isnan(x)]
        inter_phi = [x for x in inter_phi if not np.isnan(x)]

        self._log(f"\nmu embeddings:")
        self._log(f"  Intra-class (letter-letter): {np.mean(intra_mu):.4f}")
        self._log(f"  Inter-class (letter-other):  {np.mean(inter_mu):.4f}")
        if intra_mu and inter_mu:
            ratio = np.mean(inter_mu) / np.mean(intra_mu)
            self._log(f"  Ratio: {ratio:.2f}x")

        if intra_phi and inter_phi:
            self._log(f"\nphi embeddings (gauge frames):")
            self._log(f"  Intra-class (letter-letter): {np.mean(intra_phi):.4f}")
            self._log(f"  Inter-class (letter-other):  {np.mean(inter_phi):.4f}")
            ratio = np.mean(inter_phi) / np.mean(intra_phi)
            self._log(f"  Ratio: {ratio:.2f}x")

            if ratio > 1.2:
                self._log(f"\n  RESULT: phi DOES show class structure!")
            else:
                self._log(f"\n  RESULT: phi does NOT show clear class structure.")

    def _analyze_word_pairs(self):
        self._log("\n--- WORD PAIR ANALYSIS ---")
        self._log("(Only valid for single-token words)")

        def get_embed(word):
            tokens = tokenizer.encode(word)
            if len(tokens) != 1:
                return None, None
            tid = tokens[0]
            mu = self.mu_embed[tid] if self.mu_embed is not None and tid < len(self.mu_embed) else None
            phi = self.phi_embed[tid] if self.phi_embed is not None and tid < len(self.phi_embed) else None
            return mu, phi

        def dist(a, b):
            if a is None or b is None:
                return float('nan')
            return torch.norm(a - b).item()

        # Test some pairs
        pairs = [
            ("cat", "dog", "related"),
            ("cat", "the", "unrelated"),
            ("man", "woman", "related"),
            ("man", "and", "unrelated"),
            ("big", "small", "related"),
            ("big", "for", "unrelated"),
        ]

        related_phi = []
        unrelated_phi = []

        for w1, w2, rel in pairs:
            mu1, phi1 = get_embed(w1)
            mu2, phi2 = get_embed(w2)

            phi_d = dist(phi1, phi2)
            mu_d = dist(mu1, mu2)

            status = "OK" if not np.isnan(phi_d) else "SKIP (multi-token)"
            self._log(f"  {w1:8} - {w2:8} [{rel:10}]: phi={phi_d:.4f}, mu={mu_d:.4f} {status}")

            if not np.isnan(phi_d):
                if rel == "related":
                    related_phi.append(phi_d)
                else:
                    unrelated_phi.append(phi_d)

        if related_phi and unrelated_phi:
            self._log(f"\nRelated mean:   {np.mean(related_phi):.4f}")
            self._log(f"Unrelated mean: {np.mean(unrelated_phi):.4f}")
            ratio = np.mean(unrelated_phi) / np.mean(related_phi)
            self._log(f"Ratio: {ratio:.2f}x")

            if ratio > 1.2:
                self._log("\nCONCLUSION: phi encodes word-level semantics!")
            else:
                self._log("\nCONCLUSION: No clear word-level semantic encoding in phi.")

        self._log("\n" + "=" * 60)
        self._log("ANALYSIS COMPLETE")
        self._log("=" * 60)


def main():
    root = tk.Tk()
    app = GaugeAnalyzerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
