"""
=============================================================================
STAGE SIX: TEXT GENERATION (generate_tiny_llm.py)
=============================================================================
Loads the trained checkpoint from models/tiny_llm.pt and generates text
autoregressively from a user-supplied prompt.

Usage:
  python generate_tiny_llm.py --prompt "Thou art" --max-tokens 80 --temp 0.8 --top-k 5
=============================================================================
"""

import sys
import argparse
from pathlib import Path
from typing import Optional
import torch

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from lessons.stage6.model import TransformerLMConfig, TinyTransformerLM, CharTokenizer

CHECKPOINT_PATH = BASE_DIR / "models" / "tiny_llm.pt"


def load_model_and_tokenizer(checkpoint_path: Path = CHECKPOINT_PATH):
    """Loads model weights, configuration, and tokenizer from checkpoint file."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Run 'python train_tiny_llm.py' first!"
        )

    checkpoint = torch.load(checkpoint_path, weights_only=False)

    # 1. Rebuild tokenizer
    tokenizer = CharTokenizer.from_dict(checkpoint["tokenizer"])

    # 2. Rebuild model from saved configuration
    config_dict = checkpoint["config"]
    config = TransformerLMConfig(**config_dict)
    model = TinyTransformerLM(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, tokenizer, checkpoint


def generate_from_prompt(
    prompt: str,
    max_tokens: int = 80,
    temperature: float = 0.8,
    top_k: Optional[int] = 5,
    checkpoint_path: Path = CHECKPOINT_PATH
) -> str:
    """Encodes user prompt, executes generation, and decodes back to text."""
    model, tokenizer, checkpoint = load_model_and_tokenizer(checkpoint_path)

    # Encode prompt safely
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not prompt_ids:
        prompt_ids = [tokenizer.bos_id]

    generated_ids = model.generate(
        prompt_ids=prompt_ids,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        stop_id=tokenizer.eos_id
    )

    completed_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return completed_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text with trained tiny LLM.")
    parser.add_argument("--prompt", type=str, default="When forty winters ", help="Seed text prompt")
    parser.add_argument("--max-tokens", type=int, default=80, help="Max new tokens to generate")
    parser.add_argument("--temp", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K token candidate limit")
    args = parser.parse_args()

    print("=" * 75)
    print("PERSONAL AI - TINY LLM TEXT GENERATION")
    print("=" * 75)
    print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
    print(f"Prompt:      \"{args.prompt}\"")
    print(f"Temperature: {args.temp} | Top-K: {args.top_k} | Max Tokens: {args.max_tokens}\n")

    output = generate_from_prompt(
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temp,
        top_k=args.top_k
    )

    print("--- Generated Text Output ---")
    print(output)
    print("-----------------------------")
    print("\nHonest Evaluation:")
    print("This is a tiny educational model (~50,000 parameters) trained on Shakespeare sonnets.")
    print("While it mimics vocabulary, rhythm, and letter sequences, it does NOT have semantic")
    print("factual reasoning. In Stage Seven, we will run full-scale models (like Llama) via Ollama!")
