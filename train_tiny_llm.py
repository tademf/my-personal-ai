"""
=============================================================================
STAGE SIX: TRAIN MY FIRST TINY LANGUAGE MODEL (train_tiny_llm.py)
=============================================================================
This script trains a complete decoder-only Transformer language model from scratch
on personal CPU hardware using PyTorch.

DATASET:
A curated public-domain educational corpus from Shakespeare's Sonnets (data/tiny_corpus.txt).
License: Public Domain (CC0 / Unrestricted).

DISCLAIMER:
This is an educational model trained on a few thousand characters with ~50,000
parameters. It is designed to demonstrate real Transformer next-token training
mechanics and autoregressive text generation. It does NOT have the capabilities,
general knowledge, or coherence of ChatGPT.
=============================================================================
"""

import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any
import torch
import torch.optim as optim

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from lessons.stage6.model import TransformerLMConfig, TinyTransformerLM, CharTokenizer
from lessons.stage6.dataset import load_corpus, prepare_datasets, CORPUS_PATH

MODELS_DIR = BASE_DIR / "models"
CHECKPOINT_PATH = MODELS_DIR / "tiny_llm.pt"
LOG_PATH = BASE_DIR / "training_log.json"


def train_tiny_llm(
    context_length: int = 48,
    d_model: int = 64,
    num_heads: int = 4,
    num_layers: int = 2,
    d_ff: int = 128,
    batch_size: int = 16,
    epochs: int = 120,
    lr: float = 3e-3,
    eval_interval: int = 20,
    eval_batches: int = 5,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Executes reproducible, CPU-friendly language model training.
    """
    print("=" * 75)
    print("PERSONAL AI PROJECT - STAGE SIX: TRAIN TINY LLM FROM SCRATCH")
    print("=" * 75)

    # 1. Set seed for complete reproducibility
    torch.manual_seed(random_seed)

    # 2. Load Corpus and prepare datasets
    print(f"Loading corpus from: {CORPUS_PATH}")
    corpus = load_corpus(CORPUS_PATH)
    print(f"Total Corpus Length: {len(corpus):,} characters")

    tokenizer, train_data, val_data = prepare_datasets(
        corpus, context_length=context_length, train_ratio=0.85
    )

    vocab_size = len(tokenizer)
    print(f"Vocabulary Size:     {vocab_size} unique characters + special tokens")
    print(f"Training Split:      {len(train_data):,} characters (85%)")
    print(f"Validation Split:    {len(val_data):,} characters (15%)")
    print(f"Context Window:      {context_length} tokens")

    # 3. Model Configuration & Instantiation
    config = TransformerLMConfig(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=0.1
    )
    model = TinyTransformerLM(config)
    param_count = model.count_parameters()
    print(f"Total Parameters:    {param_count:,} (CPU-friendly Transformer)")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    # 4. Training Loop
    print("\n--- Training Starting on CPU ---")
    start_time = time.time()
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "epochs": []}

    print(f"{'Epoch':<8} | {'Train Loss':<12} | {'Val Loss':<12} | {'Status'}")
    print("-" * 50)

    for epoch in range(1, epochs + 1):
        model.train()
        X_batch, Y_batch = train_data.get_batch(batch_size=batch_size, random_seed=random_seed + epoch)

        optimizer.zero_grad()
        logits, loss = model(X_batch, Y_batch)
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Periodic Validation
        if epoch % eval_interval == 0 or epoch == 1 or epoch == epochs:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for v_step in range(eval_batches):
                    X_val, Y_val = val_data.get_batch(batch_size=batch_size, random_seed=999 + v_step)
                    _, val_loss = model(X_val, Y_val)
                    val_losses.append(val_loss.item())

            avg_val_loss = sum(val_losses) / len(val_losses)
            history["train_loss"].append(loss.item())
            history["val_loss"].append(avg_val_loss)
            history["epochs"].append(epoch)

            status = "Best Checkpoint" if avg_val_loss < best_val_loss else "Training"
            print(f"{epoch:<8d} | {loss.item():<12.4f} | {avg_val_loss:<12.4f} | {status}")

            # Save best checkpoint
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                MODELS_DIR.mkdir(parents=True, exist_ok=True)
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "config": config.__dict__,
                    "tokenizer": tokenizer.to_dict(),
                    "best_val_loss": best_val_loss,
                    "epoch": epoch,
                    "parameter_count": param_count
                }
                torch.save(checkpoint, CHECKPOINT_PATH)

    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.2f} seconds ({elapsed_time/epochs:.3f} s/epoch).")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Model saved to: {CHECKPOINT_PATH} ({CHECKPOINT_PATH.stat().st_size:,} bytes)")

    # 5. Save training log
    log_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "corpus_path": str(CORPUS_PATH),
        "vocab_size": vocab_size,
        "context_length": context_length,
        "d_model": d_model,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "param_count": param_count,
        "epochs": epochs,
        "best_val_loss": best_val_loss,
        "history": history,
        "elapsed_seconds": round(elapsed_time, 2)
    }
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)
    print(f"Training log saved to: {LOG_PATH}")

    # 6. Sample Autoregressive Generation
    print("\n" + "=" * 75)
    print("DEMO: AUTOREGRESSIVE TEXT GENERATION FROM PROMPT")
    print("=" * 75)

    sample_prompt = "Shall I "
    prompt_ids = tokenizer.encode(sample_prompt)
    generated_ids = model.generate(
        prompt_ids,
        max_new_tokens=80,
        temperature=0.8,
        top_k=5
    )
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print(f"Input Prompt:    \"{sample_prompt}\"")
    print(f"Model Output:    \"{generated_text}\"")

    print("\nUnderstanding the Output:")
    print("Notice the model has learned English letter patterns, common syllables, and spaces.")
    print("Because it is trained on ~2,400 characters, it produces plausible poetic sounds,")
    print("but lacks global reasoning or grammar. Scaling up to millions of words is what creates coherence.")

    return log_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a tiny Transformer LLM from scratch.")
    parser.add_argument("--epochs", type=int, default=120, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--context-length", type=int, default=48, help="Context length")
    parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate")
    args = parser.parse_args()

    train_tiny_llm(
        epochs=args.epochs,
        batch_size=args.batch_size,
        context_length=args.context_length,
        lr=args.lr
    )
    print("\n" + "=" * 75)
    print("STAGE SIX TRAINING FINISHED SUCCESSFULLY!")
    print("=" * 75)
