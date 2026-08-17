import os
import sys
import torch
import tiktoken

# Make the src/ directory importable so we can find train.py
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from train import GPTModel


MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "/home/kss/Downloads", "slayy_gpt_Collab.pt")
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.8   # >1.0 = more random, <1.0 = more greedy
TOP_K = 50          # only sample from top-k tokens (0 = disabled)


def load_model(path, device):
    """Load the saved model checkpoint."""
    if not os.path.exists(path):
        print(f"Error: No model found at {path}")
        print("Run `python src/train.py` first.")
        sys.exit(1)

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    cfg = checkpoint["config"]

    model = GPTModel(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded model: {n_params/1e6:.1f}M params")
    return model, cfg


def generate_with_temperature(model, idx, max_new_tokens, context_size, temperature=1.0, top_k=None):
    """Generate text with temperature scaling and optional top-k sampling.

    - temperature: scales logits before softmax. Higher = more random.
    - top_k: if set, zero out all logits except the top-k highest before sampling.
    """
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(idx[:, -context_size:])[:, -1, :]

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = torch.softmax(logits / temperature, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
    model.train()
    return idx


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "hello everyone and welcome back to"

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")

    tokenizer = tiktoken.get_encoding("gpt2")
    model, cfg = load_model(MODEL_PATH, device)

    context_size = cfg["context_length"]
    prompt_ids = tokenizer.encode(prompt)
    prompt_tensor = torch.tensor([prompt_ids], device=device)

    print(f"Prompt: \"{prompt}\"")
    print(f"Generating {MAX_NEW_TOKENS} tokens (temp={TEMPERATURE}, top_k={TOP_K})...\n")

    output = generate_with_temperature(
        model, prompt_tensor, MAX_NEW_TOKENS, context_size,
        temperature=TEMPERATURE, top_k=TOP_K
    )
    print(tokenizer.decode(output[0].tolist()))


if __name__ == "__main__":
    main()
