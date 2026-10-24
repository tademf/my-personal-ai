"""
Load the fine-tuned model and generate a reply to a user prompt.

Usage:
    python src/generate.py "hi"
    python src/generate.py "what is python"
    python src/generate.py "tell me a joke"
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from tokenizer import CharTokenizer
from model import ModelConfig, TinyLLM


# ── Constants ────────────────────────────────────────────────

CHECKPOINT = "models/checkpoints/finetuned.pt"
TOKENIZER = "models/checkpoints/tokenizer_ft.json"
USER_MARKER = "<|user|>"
ASSISTANT_MARKER = "<|assistant|>"
END_MARKER = "<|end|>"


# ── Model loading ────────────────────────────────────────────


def load_model():
    """Load the fine-tuned model and its tokenizer."""
    ckpt_path = Path(CHECKPOINT)
    tok_path = Path(TOKENIZER)

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Model not found: {ckpt_path}\n"
            f"Run 'python src/finetune.py' first."
        )
    if not tok_path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found: {tok_path}\n"
            f"Run 'python src/finetune.py' first."
        )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config: ModelConfig = ckpt["config"]
    model = TinyLLM(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tokenizer = CharTokenizer.load(tok_path)
    return model, tokenizer


# ── Generation ───────────────────────────────────────────────


def generate_reply(
    model: TinyLLM,
    tokenizer: CharTokenizer,
    user_message: str,
    max_new_tokens: int = 80,
    temperature: float = 0.4,
    top_k: int = 40,
) -> str:
    """
    Generate an assistant reply to a user message.

    Returns the extracted reply text with markers cleaned up.
    """
    prompt = f"{USER_MARKER}{user_message}{ASSISTANT_MARKER}"
    ids = tokenizer.encode(prompt, add_bos=True)
    x = torch.tensor([ids], dtype=torch.long)

    with torch.no_grad():
        out = model.generate(
            x,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

    raw = tokenizer.decode(out[0].tolist(), skip_special=True)
    return clean_reply(raw)


def clean_reply(raw: str) -> str:
    """Extract the assistant reply from generated text."""
    # Take everything after the assistant marker
    if ASSISTANT_MARKER in raw:
        reply = raw.split(ASSISTANT_MARKER, 1)[1]
    else:
        reply = raw

    # Stop at end marker or user marker (model may generate extra turns)
    for stop in (END_MARKER, USER_MARKER):
        if stop in reply:
            reply = reply.split(stop, 1)[0]

    # Remove any marker fragments
    for frag in ("<|", "|>", "|", "<", ">", "="):
        reply = reply.replace(frag, "")

    return reply.strip()


# ── CLI entry point ──────────────────────────────────────────


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python src/generate.py \"your prompt here\"")
        print("Example: python src/generate.py \"hi\"")
        sys.exit(1)

    user_message = " ".join(sys.argv[1:])

    print(f"Loading model…")
    model, tokenizer = load_model()

    print(f"\nUser: {user_message}")
    reply = generate_reply(model, tokenizer, user_message)
    print(f"Assistant: {reply}")


if __name__ == "__main__":
    main()