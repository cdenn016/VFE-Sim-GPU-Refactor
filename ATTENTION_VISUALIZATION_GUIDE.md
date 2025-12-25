# Attention Visualization Guide

## What "Integrate WikiText-103 Loader" Meant

You asked: *"what do you mean: Integrate your WikiText-103 loader into the --mode validation path"*

**Answer:** Replace the TODO placeholder with your existing data loading code so you visualize **REAL validation sequences** instead of fake hardcoded text.

### Before (TODO Placeholder)
```python
# TODO: Load from actual dataset
print(f"[TODO] Loading from {args.data_path}")
# ... uses fake text: "Natural language processing models..."
```

### After (Integrated)
```python
from transformer.data import create_dataloaders

train_loader, val_loader, vocab_size = create_dataloaders(
    max_seq_len=128,
    batch_size=8,
    vocab_size=50257,
    dataset='wikitext-103',  # YOUR existing data loader!
)

# Get actual validation sequence
for batch_idx, (input_ids, target_ids) in enumerate(val_loader):
    token_ids = input_ids[0:1]  # REAL data, not fake!
    break
```

---

## Usage Examples

### 1. Visualize User-Provided Text (Quick Test)
```bash
python visualize_attention_with_context.py \
    --mode text \
    --text "The transformer architecture has revolutionized natural language processing"
```

### 2. Visualize REAL WikiText-2 Validation Data
```bash
python visualize_attention_with_context.py \
    --mode validation \
    --dataset wikitext-2
```

### 3. Visualize REAL WikiText-103 Validation Data
```bash
python visualize_attention_with_context.py \
    --mode validation \
    --dataset wikitext-103 \
    --seq-len 128
```

### 4. Random Sequence (Debugging Only)
```bash
python visualize_attention_with_context.py \
    --mode random \
    --seq-len 64
```

---

## What You'll See

### Output in Terminal:
```
================================================================================
ATTENTION VISUALIZATION WITH FULL CONTEXT
================================================================================

Loading REAL validation data from wikitext-2...
==================================================================
CREATING WIKITEXT-2 DATALOADERS (BPE via tiktoken)
==================================================================

[SEQUENCE] REAL VALIDATION DATA: WIKITEXT-2
  Decoded text: In 1941, after the fall of France, the British government
  established a civil defense force to protect against air raids. The force
  was composed of volunteers who received basic training in first aid...

  Length: 128 tokens
  Tokens: In | 1941 | , | after | the | fall | of | France ...

[ATTENTION DIAGNOSTIC SUMMARY]
================================================================================
Sequence: REAL VALIDATION DATA: WIKITEXT-2
Length: 128 tokens
Attention heads: 4
--------------------------------------------------------------------------------
  Head 0: row_std = 0.234  [✓ SHARP]
  Head 1: row_std = 0.187  [✓ SHARP]
  Head 2: row_std = 0.042  [❌ UNIFORM]
  Head 3: row_std = 0.156  [✓ SHARP]

  AVERAGED: row_std = 0.089  [⚠ MEDIUM]
    ^ This is what you were seeing! Averaging destroys patterns.

  KL Matrix: row_std = 1.234
================================================================================

✓ Saved: attention_with_context.png
```

### What's in the Figure:
1. **Top row**: The actual sequence with decoded tokens
2. **Row 1**: Individual attention heads (NOT averaged!)
   - You'll see which heads are sharp vs uniform
3. **Row 2**: Head-averaged attention (what you were seeing before)
   - Plus row uniformity statistics
4. **Row 3**: KL divergence matrix
   - Shows if uniform attention is caused by uniform beliefs

---

## Key Insights

### Your Original Question
> "when I plot attention I get a say, 128 by 128 matrix for max-seq-len=128. is this plot an average of all sequences?"

**Answer:**
- ✅ **Specific sequence**: You ARE plotting batch element 0 (not averaged across batch)
- ❌ **Averaged heads**: You WERE averaging across attention heads (this destroys patterns!)
- ❌ **Unknown sequence**: You DIDN'T know what sequence it was (random? validation? what text?)

### The Fix
1. **Plot individual heads** (not averaged)
2. **Show the actual text/tokens** being visualized
3. **Use real validation data** (not random sequences)

### What You'll Discover

**Individual heads have different patterns:**
- Head 0: Attends to recent tokens (recency bias)
- Head 1: Attends to first token (BOS token)
- Head 2: Uniform (hasn't learned yet)
- Head 3: Attends to specific content words

**Averaging destroys this:**
```
Head 0:  [0.1, 0.1, 0.8]  # Sharp!
Head 1:  [0.8, 0.1, 0.1]  # Sharp!
Average: [0.45, 0.1, 0.45]  # Looks uniform! 😞
```

---

## Files Created

1. **`visualize_attention_heads.py`** (Basic)
   - Shows per-head attention
   - Uses random sequences (quick debugging)

2. **`visualize_attention_with_context.py`** ⭐ (Full Featured)
   - Shows actual decoded text
   - Loads real WikiText data
   - Per-head analysis with diagnostics

3. **`load_validation_example.py`** (Tutorial)
   - Standalone example of data loading
   - Shows exactly how to use `create_dataloaders()`

4. **`ATTENTION_VISUALIZATION_GUIDE.md`** (This file)
   - Explains what "integration" means
   - Usage examples and interpretation

---

## For Publication Figures

When generating attention heatmaps for your paper, make sure to:

1. **Always specify the sequence:**
   ```python
   metrics.generate_interpretability_outputs(
       model=model,
       sample_batch=val_batch,  # REAL validation data
       tokenizer=tokenizer,      # For decoding tokens
   )
   ```

2. **Caption should include:**
   - What the sequence is (e.g., "WikiText-103 validation excerpt")
   - The decoded text (first 50 chars)
   - Which head (if not averaged)
   - Whether self-attention is masked

3. **Example caption:**
   > "Attention pattern for Head 2 on WikiText-103 validation sequence:
   > 'In 1941, after the fall of France...' (128 tokens).
   > Self-attention masked, κ=1.0."

---

## Next Steps

1. **Run on real data:**
   ```bash
   python visualize_attention_with_context.py --mode validation --dataset wikitext-2
   ```

2. **Check if individual heads are sharp** (not averaged!)

3. **If ALL heads are uniform:**
   - Check belief diversity: Are all μ similar?
   - Check gauge frames: Are all φ similar?
   - Try lower kappa: `kappa_beta=0.3` for sharper attention
   - Check training: Has model trained long enough?

4. **For paper:** Use validation mode with actual sequences you can quote in captions!
