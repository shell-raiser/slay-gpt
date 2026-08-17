"""
Train a small GPT on SlayyPoint transcripts.
Covers concepts from Chapters 2-5 of 'Build a Large Language Model (From Scratch)'.

Usage:
    python src/train.py
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tiktoken


# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CLEAN_FILE = os.path.join(DATA_DIR, "cleaned_transcripts.txt")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

CFG = {
    "vocab_size":     50257,   # GPT-2 BPE vocabulary size
    "context_length": 256,     # max sequence length
    "emb_dim":        256,     # embedding dimension
    "n_heads":        8,       # attention heads (emb_dim must be divisible)
    "n_layers":       6,       # transformer blocks
    "drop_rate":      0.1,
    "qkv_bias":       False,
}

TRAIN = {
    "batch_size":   8,
    "num_epochs":   10,
    "eval_freq":    50,
    "eval_iter":    10,
    "lr":           5e-4,
    "weight_decay": 0.1,
    "train_ratio":  0.90,
}


# ══════════════════════════════════════════════════════════════════════
# CHAPTER 2 — Data Loading & Tokenization
# ══════════════════════════════════════════════════════════════════════
def load_transcripts():
    """Load the cleaned transcript text."""
    with open(CLEAN_FILE, "r", encoding="utf-8") as f:
        return f.read()


class GPTDataset(Dataset):
    """Sliding-window dataset for next-token prediction (Ch 2.4)."""
    def __init__(self, token_ids, max_length, stride):
        self.inputs, self.targets = [], []
        for i in range(0, len(token_ids) - max_length, stride):
            self.inputs.append(torch.tensor(token_ids[i:i + max_length], dtype=torch.long))
            self.targets.append(torch.tensor(token_ids[i + 1:i + max_length + 1], dtype=torch.long))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


# ══════════════════════════════════════════════════════════════════════
# CHAPTER 3 — Attention Mechanisms
# ══════════════════════════════════════════════════════════════════════
class MultiHeadAttention(nn.Module):
    """Multi-head causal self-attention (Ch 3.5)."""
    def __init__(self, d_in, d_out, context_length, n_heads, drop_rate=0.1, qkv_bias=False):
        super().__init__()
        assert d_out % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_out // n_heads
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(drop_rate)
        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        self.register_buffer("mask", mask.bool())

    def forward(self, x):
        B, T, _ = x.shape
        H, dk = self.n_heads, self.d_k
        Q = self.W_q(x).view(B, T, H, dk).transpose(1, 2)
        K = self.W_k(x).view(B, T, H, dk).transpose(1, 2)
        V = self.W_v(x).view(B, T, H, dk).transpose(1, 2)
        scores = (Q @ K.transpose(-2, -1)) / dk**0.5
        scores.masked_fill_(self.mask[:T, :T], float("-inf"))
        weights = self.dropout(torch.softmax(scores, dim=-1))
        context = (weights @ V).transpose(1, 2).contiguous().view(B, T, H * dk)
        return self.out_proj(context)


# ══════════════════════════════════════════════════════════════════════
# CHAPTER 4 — GPT Model Components
# ══════════════════════════════════════════════════════════════════════
class LayerNorm(nn.Module):
    """Layer normalization (Ch 4.2)."""
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean, var = x.mean(dim=-1, keepdim=True), x.var(dim=-1, keepdim=True, unbiased=False)
        return self.scale * (x - mean) / torch.sqrt(var + self.eps) + self.shift


class GELU(nn.Module):
    """GELU activation (Ch 4.3)."""
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * x**3)))


class FeedForward(nn.Module):
    """Feed-forward network: Linear → GELU → Linear (Ch 4.3)."""
    def __init__(self, emb_dim, drop_rate=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim), GELU(),
            nn.Linear(4 * emb_dim, emb_dim), nn.Dropout(drop_rate),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """One transformer block: Norm → Attn → Add → Norm → FFN → Add (Ch 4.4)."""
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.attn  = MultiHeadAttention(cfg["emb_dim"], cfg["emb_dim"], cfg["context_length"],
                                        cfg["n_heads"], cfg["drop_rate"], cfg["qkv_bias"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.ffn   = FeedForward(cfg["emb_dim"], cfg["drop_rate"])
        self.drop  = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Attention with residual
        shortcut = x
        x = self.drop(self.attn(self.norm1(x)))
        x = x + shortcut
        # FFN with residual
        shortcut = x
        x = self.drop(self.ffn(self.norm2(x)))
        return x + shortcut


class GPTModel(nn.Module):
    """Full GPT model (Ch 4.5): embeddings → transformer blocks → linear head."""
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop    = nn.Dropout(cfg["drop_rate"])
        self.blocks  = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.norm    = LayerNorm(cfg["emb_dim"])
        self.head    = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, idx):
        B, T = idx.shape
        x = self.drop(self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device)))
        x = self.norm(self.blocks(x))
        return self.head(x)


# ══════════════════════════════════════════════════════════════════════
# CHAPTER 5 — Training Loop, Loss, Generation
# ══════════════════════════════════════════════════════════════════════
def calc_loss_batch(input_batch, target_batch, model, device):
    """Cross-entropy loss for one batch."""
    logits = model(input_batch.to(device))
    return F.cross_entropy(logits.flatten(0, 1), target_batch.to(device).flatten())


def calc_loss_loader(loader, model, device, num_batches=None):
    """Average loss over `num_batches` from a DataLoader."""
    total, n = 0.0, min(num_batches or len(loader), len(loader))
    for i, (x, y) in enumerate(loader):
        if i >= n:
            break
        total += calc_loss_batch(x, y, model, device).item()
    return total / n


@torch.no_grad()
def generate(model, idx, max_new_tokens, context_size):
    """Greedy autoregressive generation (Ch 4.6)."""
    model.eval()
    for _ in range(max_new_tokens):
        logits = model(idx[:, -context_size:])[:, -1, :]
        idx = torch.cat([idx, torch.argmax(torch.softmax(logits, dim=-1), dim=-1, keepdim=True)], dim=1)
    model.train()
    return idx


def train(model, train_loader, val_loader, optimizer, device, tokenizer, cfg, train_cfg):
    """Main training loop (Ch 5.2). Returns loss history."""
    train_losses, val_losses, tokens_seen = [], [], []
    seen, step = 0, 0
    context_size = cfg["context_length"]
    prompt = tokenizer.encode("hello everyone and welcome back to")

    for epoch in range(train_cfg["num_epochs"]):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(xb, yb, model, device)
            loss.backward()
            optimizer.step()
            seen += xb.numel()
            step += 1

            if step % train_cfg["eval_freq"] == 0:
                tl = calc_loss_loader(train_loader, model, device, train_cfg["eval_iter"])
                vl = calc_loss_loader(val_loader, model, device, train_cfg["eval_iter"])
                train_losses.append(tl)
                val_losses.append(vl)
                tokens_seen.append(seen)
                print(f"  Ep {epoch+1} Step {step:05d} | train {tl:.3f} | val {vl:.3f}")

        # Sample text after each epoch
        out = generate(model, torch.tensor([prompt], device=device), 50, context_size)
        print(f"  Sample: {tokenizer.decode(out[0].tolist())}\n")

    return train_losses, val_losses, tokens_seen


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    # --- Device ---
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    print(f"Device: {device}")

    # --- Data ---
    print("Loading transcripts...")
    raw_text = load_transcripts()
    print(f"  Characters: {len(raw_text):,}")

    tokenizer = tiktoken.get_encoding("gpt2")
    all_ids = tokenizer.encode(raw_text)
    print(f"  Tokens: {len(all_ids):,}")

    split = int(len(all_ids) * TRAIN["train_ratio"])
    train_ids, val_ids = all_ids[:split], all_ids[split:]

    ctx = CFG["context_length"]
    train_loader = DataLoader(GPTDataset(train_ids, ctx, ctx), batch_size=TRAIN["batch_size"],
                              shuffle=True, drop_last=True, pin_memory=True)
    val_loader   = DataLoader(GPTDataset(val_ids, ctx, ctx), batch_size=TRAIN["batch_size"],
                              shuffle=False, drop_last=False, pin_memory=True)
    print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # --- Model ---
    torch.manual_seed(42)
    model = GPTModel(CFG).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params ({n_params/1e6:.1f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN["lr"], weight_decay=TRAIN["weight_decay"])

    # --- Train ---
    print("\nTraining...")
    train(model, train_loader, val_loader, optimizer, device, tokenizer, CFG, TRAIN)

    # --- Save ---
    save_path = os.path.join(MODEL_DIR, "slayy_gpt.pt")
    torch.save({"config": CFG, "model_state_dict": model.state_dict()}, save_path)
    print(f"Model saved → {save_path}")


if __name__ == "__main__":
    main()
