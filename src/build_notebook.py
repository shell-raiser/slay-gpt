"""Script to generate the main Jupyter notebook for training a GPT model on SlayyPoint transcripts."""

import json
import os

def md(source):
    """Create a markdown cell."""
    return {"cell_type": "markdown", "id": "n/a", "metadata": {}, "source": source.split("\n")}

def code(source):
    """Create a code cell."""
    return {"cell_type": "code", "execution_count": None, "id": "n/a", "metadata": {}, "outputs": [], "source": source.split("\n")}

cells = []

# ── Title ──────────────────────────────────────────────────────────────
cells.append(md(
"""# Building an LLM from Scratch — SlayyPoint Transcript Edition

This notebook follows **"Build a Large Language Model (From Scratch)"** by Sebastian Raschka,
covering **Chapters 2 through 5**:

| Chapter | Topic | What we code |
|---------|-------|--------------|
| Ch 2 | Working with Text Data | Tokenization, Dataset/DataLoader, Embeddings, Positional encodings |
| Ch 3 | Coding Attention Mechanisms | Self-attention, Multi-head attention, Causal masking |
| Ch 4 | Implementing a GPT Model | Transformer blocks, LayerNorm, GELU, full GPT model |
| Ch 5 | Pretraining on Unlabeled Data | Training loop, Loss/perplexity, Text generation, Checkpointing |

**Dataset:** Video transcripts from [SlayyPoint Official](https://www.youtube.com/@SlayyPointOfficial)

---"""
))

# ── Ch2 Header ─────────────────────────────────────────────────────────
cells.append(md(
"""# Chapter 2 — Working with Text Data

Before we can train an LLM, we need to:
1. **Load** raw text (our YouTube transcripts)
2. **Tokenize** it (convert text → integer token IDs)
3. Create **sliding-window batches** for training

We use **Byte Pair Encoding (BPE)** — the same tokenization scheme used by GPT-2/3/4."""
))

cells.append(md("## 2.1 Loading and cleaning the transcript data"))

cells.append(code(
"""import os
import re
import requests

# ---------- configuration ----------
DATA_DIR    = os.path.join(os.path.dirname(os.getcwd()), "data")
RAW_SRT_DIR = DATA_DIR               # where the .srt files live
CLEAN_FILE  = os.path.join(DATA_DIR, "cleaned_transcripts.txt")

# ---------- helper: strip SRT formatting ----------
def clean_srt(path):
    \"\"\"Read an SRT subtitle file and return plain text (no timestamps/numbers).\"\"\"
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    # Remove sequence numbers (lines that are just digits)
    text = re.sub(r"^\\d+\\s*$", "", text, flags=re.MULTILINE)
    # Remove timestamps  (00:01:23,456 --> 00:01:25,789)
    text = re.sub(r"\\d{2}:\\d{2}:\\d{2}[,.]\\d{3}\\s*-->\\s*\\d{2}:\\d{2}:\\d{2}[,.]\\d{3}", "", text)
    # Remove SRT tags like <i>, </i>
    text = re.sub(r"<[^>]+>", "", text)
    # Collapse whitespace
    text = re.sub(r"\\s+", " ", text).strip()
    return text

# ---------- load every .srt file ----------
all_texts = []
for fname in sorted(os.listdir(RAW_SRT_DIR)):
    if fname.endswith(".srt"):
        cleaned = clean_srt(os.path.join(RAW_SRT_DIR, fname))
        if cleaned:
            all_texts.append(cleaned)

# If a pre-cleaned file already exists, just use that
if os.path.exists(CLEAN_FILE):
    with open(CLEAN_FILE, "r", encoding="utf-8") as f:
        raw_text = f.read()
    print(f"Loaded existing cleaned file: {CLEAN_FILE}")
else:
    raw_text = "\\n\\n".join(all_texts)
    with open(CLEAN_FILE, "w", encoding="utf-8") as f:
        f.write(raw_text)
    print(f"Created cleaned file from {len(all_texts)} transcripts")

print(f"Characters : {len(raw_text):,}")
print(f"Words      : {len(raw_text.split()):,}")
print(f"\\nPreview (first 300 chars):\\n{raw_text[:300]}")"""
))

# ── Ch 2.2 — Tokenization ─────────────────────────────────────────────
cells.append(md(
"""## 2.2 Tokenization with Byte Pair Encoding (BPE)

LLMs don't read raw text — they read **integer token IDs**.
BPE is a subword tokenizer: it splits rare words into smaller meaningful pieces
while keeping common words as single tokens.

GPT-2's BPE tokenizer has a vocabulary of 50,257 tokens.

**Key idea from the book:** We could build our own BPE tokenizer, but for this
project we reuse the pretrained GPT-2 tokenizer so our model's token
embeddings are compatible with pretrained GPT-2 weights (Chapter 6)."""
))

cells.append(code(
"""import tiktoken

# tiktoken is OpenAI's fast BPE implementation (pip install tiktoken)
tokenizer = tiktoken.get_encoding("gpt2")

# Quick demo
example = "SlayyPoint is the best channel"
token_ids = tokenizer.encode(example)
print(f"Text   : {example}")
print(f"Tokens : {token_ids}")
print(f"Decoded: {[tokenizer.decode([t]) for t in token_ids]}")"""
))

cells.append(code(
"""# Tokenize the entire corpus
all_token_ids = tokenizer.encode(raw_text)
total_tokens = len(all_token_ids)
print(f"Total tokens in corpus: {total_tokens:,}")"""
))

# ── Ch 2.3 — Train / Val split ────────────────────────────────────────
cells.append(md(
"""## 2.3 Train / Validation Split

We split the token array 90 / 10. We split at the **token** level
(not character level) because BPE tokens are the unit the model sees."""
))

cells.append(code(
"""train_ratio = 0.90
split_idx   = int(train_ratio * total_tokens)
train_ids   = all_token_ids[:split_idx]
val_ids     = all_token_ids[split_idx:]

print(f"Train tokens: {len(train_ids):,}")
print(f"Val   tokens: {len(val_ids):,}")"""
))

# ── Ch 2.4 — Dataset with sliding window ──────────────────────────────
cells.append(md(
"""## 2.4 Dataset with Sliding Window

The book's `GPTDatasetV1` creates input/target pairs using a sliding window.
Given a `max_length` (context window) and a `stride`:
- **input** = tokens[i : i + max_length]
- **target** = tokens[i+1 : i + max_length + 1]  (shifted by 1)

This is the standard "next-token prediction" objective."""
))

cells.append(code(
"""import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    \"\"\"A simple sliding-window dataset for next-token prediction.

    Each sample is a chunk of `max_length` tokens.
    The target is the same chunk shifted right by 1 position.
    \"\"\"
    def __init__(self, token_ids, tokenizer, max_length, stride):
        self.input_ids  = []
        self.target_ids = []
        # Slide a window of size max_length across the token list
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk  = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]"""
))

cells.append(code(
"""def create_dataloader_v1(token_ids, batch_size=4, max_length=256,
                         stride=256, drop_last=True, shuffle=True, num_workers=0):
    \"\"\"Build a DataLoader of GPTDatasetV1 chunks.

    Args:
        token_ids   : list of integer token IDs (the full corpus)
        batch_size  : number of sequences per batch
        max_length  : context window size (tokens per sample)
        stride      : how far to slide the window between samples
        drop_last   : drop the last incomplete batch (keeps shapes uniform)
        shuffle     : shuffle between epochs
        num_workers : DataLoader workers (0 = main process)
    \"\"\"
    dataset = GPTDatasetV1(token_ids, tokenizer, max_length, stride)
    return DataLoader(dataset, batch_size=batch_size,
                      drop_last=drop_last, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True)"""
))

# ── Ch 2.5 — Embeddings ───────────────────────────────────────────────
cells.append(md(
"""## 2.5 Token Embeddings + Positional Embeddings

Two embedding layers convert integer token IDs into dense vectors:

1. **Token Embedding** — maps each token ID to a learnable `d_model`-dim vector.
2. **Positional Embedding** — adds position information so the model knows
   token order (transformers have no built-in notion of sequence position).

The final input to the transformer = token embedding + positional embedding."""
))

cells.append(code(
"""import torch.nn as nn

class TokenEmbedding(nn.Module):
    \"\"\"Maps token IDs to dense vectors of size `emb_dim`.\"\"\"
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)

    def forward(self, token_ids):
        # token_ids: (batch, seq_len)  →  (batch, seq_len, emb_dim)
        return self.embedding(token_ids)


class PositionalEmbedding(nn.Module):
    \"\"\"Learnable positional embeddings (one vector per position).

    Unlike sinusoidal encodings, these are trained along with the rest
    of the model. They work well when the training sequence length is
    reasonable (the book uses this approach for GPT-2).
    \"\"\"
    def __init__(self, max_length, emb_dim):
        super().__init__()
        self.embedding = nn.Embedding(max_length, emb_dim)

    def forward(self, token_ids):
        batch_size, seq_len = token_ids.shape
        #.arange creates position indices [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=token_ids.device)
        return self.embedding(positions).unsqueeze(0)  # broadcast over batch


class InputEmbedding(nn.Module):
    \"\"\"Token embedding + positional embedding + dropout.

    This is the complete input layer of the GPT model.
    \"\"\"
    def __init__(self, vocab_size, max_length, emb_dim, drop_rate=0.1):
        super().__init__()
        self.token_emb = TokenEmbedding(vocab_size, emb_dim)
        self.pos_emb   = PositionalEmbedding(max_length, emb_dim)
        self.dropout   = nn.Dropout(drop_rate)

    def forward(self, token_ids):
        token_embeds = self.token_emb(token_ids)   # (B, T, D)
        pos_embeds   = self.pos_emb(token_ids)     # (1, T, D)
        return self.dropout(token_embeds + pos_embeds)"""
))

# ── Ch2 Summary ────────────────────────────────────────────────────────
cells.append(md(
"""---
### Chapter 2 — Key Takeaways
- Text must be **tokenized** before the model can process it. We use GPT-2's BPE tokenizer.
- The **sliding window** approach creates overlapping input/target pairs for next-token prediction.
- **Token + Positional embeddings** convert integer IDs into dense vectors the transformer can work with.

---"""
))

# ════════════════════════════════════════════════════════════════════════
# CHAPTER 3 — Attention Mechanisms
# ════════════════════════════════════════════════════════════════════════
cells.append(md(
"""# Chapter 3 — Coding Attention Mechanisms

The **self-attention** mechanism lets every token "look at" every other token
in the sequence and compute a weighted sum of their value vectors.

Key concepts:
- **Scaled dot-product attention**: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
- **Causal mask**: prevents tokens from attending to future positions
- **Multi-head attention**: runs several attention functions in parallel"""
))

cells.append(md("## 3.1 Causal Self-Attention (Single Head)"))

cells.append(code(
"""class CausalSelfAttention(nn.Module):
    \"\"\"Single-head causal self-attention.

    This computes Q, K, V from the same input, applies scaled dot-product
    attention, and masks future positions so the model is autoregressive
    (it can only look left, not right).

    Book reference: Section 3.4 — 'Implementing efficient multi-head attention'
    \"\"\"
    def __init__(self, emb_dim, context_length, drop_rate=0.1, qkv_bias=False):
        super().__init__()
        self.d_k = emb_dim  # we use single-head here for simplicity
        # Linear projections for Query, Key, Value
        self.W_q = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.W_k = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.W_v = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(emb_dim, emb_dim)  # output projection
        self.dropout  = nn.Dropout(drop_rate)
        # Pre-compute the causal mask (upper triangular = masked out)
        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        self.register_buffer("mask", mask.bool())  # saved with model but not a parameter

    def forward(self, x):
        B, T, D = x.shape
        # Project input into Q, K, V
        Q = self.W_q(x)  # (B, T, D)
        K = self.W_k(x)  # (B, T, D)
        V = self.W_v(x)  # (B, T, D)
        # Attention scores: QK^T / sqrt(d_k)
        scores = (Q @ K.transpose(1, 2)) / (D ** 0.5)  # (B, T, T)
        # Apply causal mask: set future positions to -inf so softmax → 0
        scores.masked_fill_(self.mask[:T, :T], float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)     # (B, T, T)
        attn_weights = self.dropout(attn_weights)
        # Weighted sum of values
        context = attn_weights @ V                        # (B, T, D)
        return self.out_proj(context)"""
))

cells.append(md("## 3.2 Multi-Head Attention"))

cells.append(code(
"""class MultiHeadAttention(nn.Module):
    \"\"\"Multi-head attention: runs `n_heads` attention heads in parallel.

    The input is split into `n_heads` chunks, each of dimension `d_k = emb_dim // n_heads`.
    Each head computes attention independently, then outputs are concatenated
    and projected back to `emb_dim`.

    Book reference: Section 3.5
    \"\"\"
    def __init__(self, emb_dim, n_heads, context_length, drop_rate=0.1, qkv_bias=False):
        super().__init__()
        assert emb_dim % n_heads == 0, "emb_dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_k     = emb_dim // n_heads

        self.W_q = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.W_k = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.W_v = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(emb_dim, emb_dim)
        self.dropout  = nn.Dropout(drop_rate)
        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        self.register_buffer("mask", mask.bool())

    def forward(self, x):
        B, T, D = x.shape
        H = self.n_heads
        d_k = self.d_k
        # Project and reshape to (B, T, n_heads, d_k) then transpose to (B, n_heads, T, d_k)
        Q = self.W_q(x).view(B, T, H, d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, H, d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, H, d_k).transpose(1, 2)
        # Attention scores for all heads at once
        scores = (Q @ K.transpose(-2, -1)) / (d_k ** 0.5)  # (B, H, T, T)
        scores.masked_fill_(self.mask[:T, :T], float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        # Weighted sum and reshape back
        context = (attn_weights @ V)                         # (B, H, T, d_k)
        context = context.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(context)"""
))

cells.append(md(
"""---
### Chapter 3 — Key Takeaways
- **Self-attention** computes a weighted representation of the sequence where each token
  can attend to all previous tokens (thanks to the causal mask).
- **Multi-head** attention lets the model capture different types of relationships
  simultaneously (e.g., syntax vs. semantics).
- The **causal mask** ensures autoregressive generation — the model never peeks at the future.

---"""
))

# ════════════════════════════════════════════════════════════════════════
# CHAPTER 4 — Implementing a GPT Model from Scratch
# ════════════════════════════════════════════════════════════════════════
cells.append(md(
"""# Chapter 4 — Implementing a GPT Model from Scratch

Now we assemble all the pieces into a full GPT-like transformer:

```
Input Embeddings
      │
      ▼
┌─────────────────────┐
│  Transformer Block  │  × n_layers
│  ├─ LayerNorm       │
│  ├─ Multi-Head Attn │
│  ├─ Residual Add    │
│  ├─ LayerNorm       │
│  ├─ FeedForward     │
│  └─ Residual Add    │
└─────────────────────┘
      │
      ▼
  Final LayerNorm
      │
      ▼
  Linear → logits (vocab_size)"""
))

cells.append(md("## 4.1 Layer Normalization"))

cells.append(code(
"""class LayerNorm(nn.Module):
    \"\"\"Custom LayerNorm (no PyTorch built-in, so we implement it).

    Normalizes activations to zero mean and unit variance, then applies
    a learnable scale and shift. This stabilizes training.

    Book reference: Section 4.2
    \"\"\"
    def __init__(self, emb_dim):
        super().__init__()
        self.eps   = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift"""
))

cells.append(md("## 4.2 Feed-Forward Network with GELU Activation"))

cells.append(code(
"""class GELU(nn.Module):
    \"\"\"Gaussian Error Linear Unit — a smooth approximation to ReLU.

    GELU(x) ≈ 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))

    Used in GPT-2 and most modern LLMs instead of ReLU.
    Book reference: Section 4.3
    \"\"\"
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    \"\"\"Position-wise feed-forward network: Linear → GELU → Linear.

    The hidden dimension is typically 4× the embedding dimension.
    This is where most of the model's 'knowledge' is stored.
    Book reference: Section 4.3
    \"\"\"
    def __init__(self, emb_dim, drop_rate=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
            nn.Dropout(drop_rate),
        )

    def forward(self, x):
        return self.net(x)"""
))

cells.append(md("## 4.3 Transformer Block"))

cells.append(code(
"""class TransformerBlock(nn.Module):
    \"\"\"One transformer block = LayerNorm → Multi-Head Attention → Add
                                → LayerNorm → FeedForward → Add

    The residual connections (the '+ x' parts) allow gradients to flow
    directly through the network, making deep models trainable.
    Book reference: Section 4.4
    \"\"\"
    def __init__(self, emb_dim, n_heads, context_length, drop_rate=0.1, qkv_bias=False):
        super().__init__()
        self.norm1        = LayerNorm(emb_dim)
        self.attention     = MultiHeadAttention(emb_dim, n_heads, context_length,
                                                drop_rate, qkv_bias)
        self.norm2        = LayerNorm(emb_dim)
        self.feed_forward = FeedForward(emb_dim, drop_rate)
        self.drop_shortcut = nn.Dropout(drop_rate)

    def forward(self, x):
        # First sub-layer: multi-head attention with residual connection
        shortcut = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # residual connection

        # Second sub-layer: feed-forward with residual connection
        shortcut = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # residual connection
        return x"""
))

cells.append(md("## 4.4 The Full GPT Model"))

cells.append(code(
"""class GPTModel(nn.Module):
    \"\"\"A GPT-like transformer language model.

    Architecture:
      1. Token + Positional Embeddings
      2. Stack of Transformer Blocks
      3. Final LayerNorm
      4. Linear head → vocabulary logits

    This matches the 124M-parameter GPT-2 architecture described in
    Radford et al. (2019), as implemented in the book (Section 4.5).
    \"\"\"
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb    = nn.Embedding(cfg["vocab_size"],  cfg["emb_dim"])
        self.pos_emb    = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb   = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(
                cfg["emb_dim"], cfg["n_heads"], cfg["context_length"],
                cfg["drop_rate"], cfg["qkv_bias"]
            ) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head   = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        \"\"\"Forward pass: returns logits for next-token prediction.\"\"\"
        B, T = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)                        # (B, T, D)
        pos_embeds = self.pos_emb(torch.arange(T, device=in_idx.device))  # (T, D)
        x = self.drop_emb(tok_embeds + pos_embeds)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)                                 # (B, T, vocab_size)
        return logits"""
))

cells.append(md("## 4.5 Text Generation (Greedy Decoding)"))

cells.append(code(
"""def generate_text_simple(model, idx, max_new_tokens, context_size):
    \"\"\"Autoregressive text generation — one token at a time.

    At each step:
      1. Crop the input to the last `context_size` tokens (context window)
      2. Forward pass → logits
      3. Pick the token with the highest logit (greedy decoding)
      4. Append it to the sequence and repeat

    Book reference: Section 4.6
    \"\"\"
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop to context size if the sequence is too long
            idx_cond = idx[:, -context_size:]
            logits = model(idx_cond)
            # Focus only on the last time step
            logits = logits[:, -1, :]     # (B, vocab_size)
            probs  = torch.softmax(logits, dim=-1)
            idx_next = torch.argmax(probs, dim=-1, keepdim=True)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)
    model.train()
    return idx"""
))

cells.append(md(
"""---
### Chapter 4 — Key Takeaways
- **LayerNorm** stabilizes training by normalizing activations.
- **GELU** is a smooth activation preferred in transformers over ReLU.
- **Residual connections** enable training of deep networks.
- The GPT model is simply embeddings → transformer blocks → final norm → linear head.
- **Greedy decoding** generates text one token at a time by picking the highest-probability next token.

---"""
))

# ════════════════════════════════════════════════════════════════════════
# CHAPTER 5 — Pretraining on Unlabeled Data
# ════════════════════════════════════════════════════════════════════════
cells.append(md(
"""# Chapter 5 — Pretraining on Unlabeled Data

This is where we actually **train** the model. The core loop is:
1. Sample a batch of (input, target) sequences
2. Forward pass → logits
3. Compute cross-entropy loss against the targets
4. Backpropagate and update weights
5. Repeat

We also track **perplexity** (exponential of the loss) — a more interpretable metric."""
))

cells.append(md("## 5.1 Model Configuration"))

cells.append(code(
"""# We use a SMALL model so training is fast on a laptop.
# The original GPT-2 124M uses context_length=1024, emb_dim=768, n_layers=12.
# We shrink everything for quick iteration.

GPT_CONFIG_SMALL = {
    "vocab_size"    : 50257,   # GPT-2 BPE vocabulary
    "context_length": 256,     # reduced from 1024 for speed
    "emb_dim"       : 256,     # reduced from 768
    "n_heads"       : 8,       # reduced from 12  (emb_dim must be divisible)
    "n_layers"      : 6,       # reduced from 12
    "drop_rate"     : 0.1,
    "qkv_bias"      : False,
}

# Uncomment below for the full 124M config (needs a GPU!)
# GPT_CONFIG_SMALL = {
#     "vocab_size": 50257, "context_length": 1024,
#     "emb_dim": 768, "n_heads": 12, "n_layers": 12,
#     "drop_rate": 0.1, "qkv_bias": False,
# }

print(f"Config: {GPT_CONFIG_SMALL}")"""
))

cells.append(md("## 5.2 Build DataLoaders"))

cells.append(code(
"""# Re-tokenize with the right context length
MAX_LENGTH = GPT_CONFIG_SMALL["context_length"]
STRIDE     = MAX_LENGTH  # non-overlapping for simplicity

torch.manual_seed(42)
train_loader = create_dataloader_v1(
    train_ids, batch_size=8, max_length=MAX_LENGTH,
    stride=STRIDE, drop_last=True, shuffle=True
)
val_loader = create_dataloader_v1(
    val_ids, batch_size=8, max_length=MAX_LENGTH,
    stride=STRIDE, drop_last=False, shuffle=False
)

# Sanity check
for inputs, targets in train_loader:
    print(f"Input  shape: {inputs.shape}")   # (batch, context_length)
    print(f"Target shape: {targets.shape}")
    break"""
))

cells.append(md("## 5.3 Loss Functions"))

cells.append(code(
"""def calc_loss_batch(input_batch, target_batch, model, device):
    \"\"\"Compute the cross-entropy loss for a single batch.\"\"\"
    input_batch  = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    # Flatten both to (batch*seq_len, vocab_size) and (batch*seq_len,)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    \"\"\"Average the loss over several batches (for faster evaluation).\"\"\"
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    if num_batches is None:
        num_batches = len(data_loader)
    num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            total_loss += calc_loss_batch(input_batch, target_batch, model, device).item()
        else:
            break
    return total_loss / num_batches"""
))

cells.append(md(
"""## 5.4 Perplexity

**Perplexity** = exp(cross-entropy loss).

It represents the "effective number of choices" the model is uncertain about
at each step. A perplexity of 1 means perfect prediction; lower is better."""
))

cells.append(code(
"""def calc_perplexity(loss):
    \"\"\"Perplexity is the exponential of the cross-entropy loss.\"\"\"
    return torch.exp(torch.tensor(loss))"""
))

cells.append(md("## 5.5 The Training Loop"))

cells.append(code(
"""def train_model_simple(model, train_loader, val_loader, optimizer, device,
                       num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    \"\"\"The core training loop from Chapter 5.

    For each epoch:
      - Iterate over training batches (forward, loss, backward, optimizer step)
      - Every `eval_freq` steps, evaluate on train/val and print a text sample
      - After each epoch, generate a text sample to qualitatively inspect progress

    Returns:
        train_losses, val_losses, tokens_seen  (for plotting)
    \"\"\"
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen  = 0
    global_step  = 0

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step  += 1

            # Periodic evaluation
            if global_step % eval_freq == 0:
                train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
                val_loss   = calc_loss_loader(val_loader,   model, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}, "
                      f"Perplexity {calc_perplexity(val_loss):.1f}")

        # Generate a sample after each epoch
        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
        val_loss   = calc_loss_loader(val_loader,   model, device, eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    \"\"\"Print a short generated text to qualitatively check model progress.\"\"\"
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = tokenizer.encode(start_context)
    token_ids = torch.tensor(encoded, device=device).unsqueeze(0)
    generated = generate_text_simple(model, token_ids, 50, context_size)
    decoded = tokenizer.decode(generated[0].tolist())
    print(f"  → {decoded}\\n")
    model.train()"""
))

cells.append(md("## 5.6 Initialize and Train"))

cells.append(code(
"""import torch

# Device selection: GPU → MPS (Mac) → CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Create model
torch.manual_seed(42)
model = GPTModel(GPT_CONFIG_SMALL)
model.to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,}  ({n_params/1e6:.1f}M)")

# Optimizer — AdamW is standard for transformer training
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)"""
))

cells.append(code(
"""# ── TRAIN ──
# This will take ~5-15 min on CPU, much faster on GPU.
# Increase num_epochs for better results.

TRAINING_CONFIG = {
    "num_epochs"  : 10,
    "eval_freq"   : 50,    # evaluate every N optimizer steps
    "eval_iter"   : 10,    # batches for evaluation
    "start_context": "hello everyone and welcome back to",  # SlayyPoint-style prompt
}

train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    **TRAINING_CONFIG, tokenizer=tokenizer
)"""
))

cells.append(md("## 5.7 Plot Training Curves"))

cells.append(code(
"""import matplotlib.pyplot as plt

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(tokens_seen, train_losses, label="Train loss")
ax1.plot(tokens_seen, val_losses,   label="Val loss")
ax1.set_xlabel("Tokens seen")
ax1.set_ylabel("Cross-entropy loss")
ax1.set_title("Training Progress")
ax1.legend()
plt.tight_layout()
plt.show()"""
))

cells.append(md("## 5.8 Generate Text with the Trained Model"))

cells.append(code(
"""# Try your own prompt!
prompt = "so basically the problem is"

prompt_ids = tokenizer.encode(prompt)
prompt_tensor = torch.tensor(prompt_ids, device=device).unsqueeze(0)

output_ids = generate_text_simple(model, prompt_tensor, 200, GPT_CONFIG_SMALL["context_length"])
output_text = tokenizer.decode(output_ids[0].tolist())
print(output_text)"""
))

cells.append(md("## 5.9 Save & Load Checkpoints"))

cells.append(code(
"""def save_checkpoint(model, optimizer, epoch, loss, filepath):
    \"\"\"Save a training checkpoint so you can resume later.\"\"\"
    torch.save({
        "epoch": epoch,
        "model_state_dict":    model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, filepath)
    print(f"Checkpoint saved → {filepath}")


def load_checkpoint(filepath, model, optimizer):
    \"\"\"Load a training checkpoint.\"\"\"
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch   = checkpoint["epoch"]
    loss    = checkpoint["loss"]
    print(f"Loaded checkpoint from epoch {epoch}, loss {loss:.4f}")
    return epoch, loss


# Save the trained model
save_checkpoint(model, optimizer, 0, val_losses[-1] if val_losses else 0.0,
                "../models/slayy_gpt.pt")"""
))

cells.append(md("## 5.10 Optional: Load Pretrained GPT-2 Weights"))

cells.append(code(
"""# This section shows how to load OpenAI's pretrained GPT-2 weights
# into our architecture.  (Requires: pip install tensorflow)
#
# Uncomment to run:

# import numpy as np
# from tensorflow.python.training.py_utils import read_checkpoint as tf_read_checkpoint
#
# def load_weights_into_gpt(model, params):
#     \"\"\"Copy pretrained GPT-2 weights into our GPTModel.\"\"\"
#     model.tok_emb.weight  = torch.nn.Parameter(torch.tensor(params["wte"], dtype=torch.float32))
#     model.pos_emb.weight  = torch.nn.Parameter(torch.tensor(params["wpe"], dtype=torch.float32))
#     for b in range(len(params["blocks"])):
#         q_w, k_w, v_w = np.split(params["blocks"][b]["attn"]["c_attn"]["w"], 3, axis=-1)
#         model.trf_blocks[b].attention.W_q.weight = torch.nn.Parameter(torch.tensor(q_w.T, dtype=torch.float32))
#         model.trf_blocks[b].attention.W_k.weight = torch.nn.Parameter(torch.tensor(k_w.T, dtype=torch.float32))
#         model.trf_blocks[b].attention.W_v.weight = torch.nn.Parameter(torch.tensor(v_w.T, dtype=torch.float32))
#         q_b, k_b, v_b = np.split(params["blocks"][b]["attn"]["c_attn"]["b"], 3, axis=-1)
#         model.trf_blocks[b].attention.W_q.bias = torch.nn.Parameter(torch.tensor(q_b, dtype=torch.float32))
#         model.trf_blocks[b].attention.W_k.bias = torch.nn.Parameter(torch.tensor(k_b, dtype=torch.float32))
#         model.trf_blocks[b].attention.W_v.bias = torch.nn.Parameter(torch.tensor(v_b, dtype=torch.float32))
#         model.trf_blocks[b].attention.out_proj.weight = torch.nn.Parameter(
#             torch.tensor(params["blocks"][b]["attn"]["c_proj"]["w"].T, dtype=torch.float32))
#         model.trf_blocks[b].attention.out_proj.bias = torch.nn.Parameter(
#             torch.tensor(params["blocks"][b]["attn"]["c_proj"]["b"], dtype=torch.float32))
#         model.trf_blocks[b].feed_forward.net[0].weight = torch.nn.Parameter(
#             torch.tensor(params["blocks"][b]["mlp"]["c_fc"]["w"].T, dtype=torch.float32))
#         model.trf_blocks[b].feed_forward.net[0].bias = torch.nn.Parameter(
#             torch.tensor(params["blocks"][b]["mlp"]["c_fc"]["b"], dtype=torch.float32))
#         model.trf_blocks[b].feed_forward.net[2].weight = torch.nn.Parameter(
#             torch.tensor(params["blocks"][b]["mlp"]["c_proj"]["w"].T, dtype=torch.float32))
#         model.trf_blocks[b].feed_forward.net[2].bias = torch.nn.Parameter(
#             torch.tensor(params["blocks"][b]["mlp"]["c_proj"]["b"], dtype=torch.float32))
#         model.trf_blocks[b].norm1.scale = torch.nn.Parameter(torch.tensor(params["blocks"][b]["ln_1"]["g"], dtype=torch.float32))
#         model.trf_blocks[b].norm1.shift = torch.nn.Parameter(torch.tensor(params["blocks"][b]["ln_1"]["b"], dtype=torch.float32))
#         model.trf_blocks[b].norm2.scale = torch.nn.Parameter(torch.tensor(params["blocks"][b]["ln_2"]["g"], dtype=torch.float32))
#         model.trf_blocks[b].norm2.shift = torch.nn.Parameter(torch.tensor(params["blocks"][b]["ln_2"]["b"], dtype=torch.float32))
#     model.final_norm.scale = torch.nn.Parameter(torch.tensor(params["g"], dtype=torch.float32))
#     model.final_norm.shift = torch.nn.Parameter(torch.tensor(params["b"], dtype=torch.float32))
#     model.out_head.weight  = torch.nn.Parameter(torch.tensor(params["wte"].T, dtype=torch.float32))
#
# print("See the commented code above to load pretrained GPT-2 weights.")"""
))

# ── Final summary ──────────────────────────────────────────────────────
cells.append(md(
"""---
# Summary — What We Built

| Component | File / Cell | What it does |
|-----------|-------------|--------------|
| **Tokenizer** | §2.2 | BPE tokenizer converts text ↔ token IDs |
| **Dataset** | §2.4 | Sliding window creates (input, target) pairs |
| **Embeddings** | §2.5 | Token + positional embeddings → dense vectors |
| **Attention** | §3.1–3.2 | Causal self-attention + multi-head attention |
| **Transformer Block** | §4.1–4.3 | LayerNorm → Attention → Add → LayerNorm → FFN → Add |
| **GPT Model** | §4.4 | Full model: embeddings → blocks → linear head |
| **Generation** | §4.5 | Autoregressive text generation (greedy decoding) |
| **Training** | §5.1–5.6 | Cross-entropy loss, training loop, evaluation |
| **Checkpointing** | §5.9 | Save/load training progress |

## Next Steps (after Ch 5 in the book)
- **Ch 6:** Finetune for text classification (e.g., sentiment analysis)
- **Ch 7:** Finetune to follow instructions (chat-style)
- **Ch 7+:** Load pretrained GPT-2 weights for much better results

## Tips
- For better results, use the full 124M config (needs a GPU)
- Train for more epochs — the small dataset overfits quickly
- Try different prompts to see how the model's "SlayyPoint style" evolves
---"""
))

# ── Assemble and write ─────────────────────────────────────────────────
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "notebooks", "slayy_point_llm.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook written → {out_path}")
print(f"Total cells: {len(cells)} ({sum(1 for c in cells if c['cell_type']=='code')} code, "
      f"{sum(1 for c in cells if c['cell_type']=='markdown')} markdown)")
