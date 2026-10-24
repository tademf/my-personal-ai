"""
Training loop for the tiny LLM.

Trains a TinyLLM on a text corpus using next-token prediction.
Saves checkpoints periodically and the final model at the end.

Usage:
    python src/train.py                      # default config
    python src/train.py --steps 2000         # override max steps
    python src/train.py --resume models/checkpoints/step_500.pt

The default config is sized for a CPU:
    - ~830K parameters
    - batch_size=16, seq_len=128
    - 2000 training steps (~15-25 min on a modern i5)
    - valid loss should drop from ~4.16 to ~1.8-2.2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from tokenizer import CharTokenizer
from dataset import TextDataset
from model import ModelConfig, TinyLLM


# ── Config ────────────────────────────────────────────────────


def make_default_config(vocab_size: int) -> ModelConfig:
    """
    Model config sized for CPU training.

    ~830K parameters. Increase d_model / n_layers if you have a GPU
    or are willing to wait longer.
    """
    return ModelConfig(
        vocab_size=vocab_size,
        max_seq_len=128,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        dropout=0.1,
    )


# ── Helpers ───────────────────────────────────────────────────


def format_time(seconds: float) -> str:
    """Format seconds as 'Xm Ys' or 'Xs'."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.0f}s"


@torch.no_grad()
def estimate_loss(
    model: TinyLLM,
    dataset: TextDataset,
    eval_batches: int = 20,
    batch_size: int = 16,
    seq_len: int = 128,
) -> dict[str, float]:
    """
    Estimate train and val loss by averaging over a few fixed batches.

    Uses the same batches every time so loss values are comparable
    across evaluation points (much less noisy than single-batch checks).
    """
    model.eval()
    losses = {}
    for split in ("train", "val"):
        total = 0.0
        for _ in range(eval_batches):
            x, y = dataset.get_batch(split, batch_size=batch_size, seq_len=seq_len)
            _, loss = model(x, y)
            total += loss.item()
        losses[split] = total / eval_batches
    model.train()
    return losses


# ── Training ──────────────────────────────────────────────────


def train(args: argparse.Namespace) -> None:
    # ── Setup paths ──────────────────────────────────────────
    ckpt_dir = Path("models/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Seed (reproducibility) ───────────────────────────────
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── Load dataset ─────────────────────────────────────────
    print(f"Loading dataset from {args.data_file}...")
    dataset = TextDataset(args.data_file)
    vocab_size = dataset.tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}")
    print()

    # ── Build or load model ──────────────────────────────────
    config = make_default_config(vocab_size)
    model = TinyLLM(config)
    start_step = 0
    best_val_loss = float("inf")

    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        start_step = checkpoint.get("step", 0)
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        print(f"  resumed at step {start_step}, best val loss {best_val_loss:.4f}")
        print()

    # ── Optimizer (AdamW with weight decay) ──────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # Cosine learning-rate schedule with warmup
    def lr_at(step: int) -> float:
        if step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        # cosine decay from lr to lr * min_lr_ratio
        return args.lr * (
            args.min_lr_ratio
            + (1 - args.min_lr_ratio) * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
        )

    # ── Training loop ────────────────────────────────────────
    print(f"Training for {args.max_steps} steps")
    print(f"Batch size: {args.batch_size}, seq_len: {args.seq_len}")
    print(f"Learning rate: {args.lr}, warmup: {args.warmup_steps} steps")
    print(f"Evaluating every {args.eval_every} steps")
    print(f"Checkpointing every {args.save_every} steps to {ckpt_dir}/")
    print()

    model.train()
    t0 = time.time()

    for step in range(start_step, args.max_steps):
        # Set LR for this step
        lr = lr_at(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Forward + backward
        x, y = dataset.get_batch("train", batch_size=args.batch_size, seq_len=args.seq_len)
        logits, loss = model(x, y)

        # Guard: skip step if loss is NaN/Inf (rare, but possible)
        if not torch.isfinite(loss):
            print(f"  ⚠ step {step}: non-finite loss {loss.item()}, skipping")
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Log training loss
        if (step + 1) % args.log_every == 0:
            elapsed = time.time() - t0
            steps_done = step + 1 - start_step
            steps_per_sec = steps_done / elapsed if elapsed > 0 else 0
            eta = (args.max_steps - step - 1) / steps_per_sec if steps_per_sec > 0 else 0
            print(
                f"step {step + 1:>5}/{args.max_steps} "
                f"| train loss {loss.item():.4f} "
                f"| lr {lr:.2e} "
                f"| {steps_per_sec:.2f} it/s "
                f"| eta {format_time(eta)}"
            )

        # Evaluate on train + val
        if (step + 1) % args.eval_every == 0:
            losses = estimate_loss(model, dataset, eval_batches=20, batch_size=args.batch_size, seq_len=args.seq_len)
            print(
                f"  ▼ eval @ step {step + 1}: "
                f"train {losses['train']:.4f}, val {losses['val']:.4f}"
            )
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                torch.save(
                    {
                        "step": step + 1,
                        "model_state_dict": model.state_dict(),
                        "config": config,
                        "best_val_loss": best_val_loss,
                    },
                    ckpt_dir / "best.pt",
                )
                print(f"  ✓ new best val loss {best_val_loss:.4f} saved to {ckpt_dir / 'best.pt'}")

        # Periodic checkpoint
        if (step + 1) % args.save_every == 0:
            ckpt_path = ckpt_dir / f"step_{step + 1}.pt"
            torch.save(
                {
                    "step": step + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "best_val_loss": best_val_loss,
                },
                ckpt_path,
            )
            print(f"  💾 checkpoint saved: {ckpt_path}")

    # ── Save final model ─────────────────────────────────────
    final_path = ckpt_dir / "final.pt"
    torch.save(
        {
            "step": args.max_steps,
            "model_state_dict": model.state_dict(),
            "config": config,
            "best_val_loss": best_val_loss,
        },
        final_path,
    )
    print()
    print(f"Training complete in {format_time(time.time() - t0)}")
    print(f"Final model saved to {final_path}")
    print(f"Best val loss: {best_val_loss:.4f}")

    # Save the tokenizer next to the checkpoint (convenient for inference)
    tokenizer_path = ckpt_dir / "tokenizer.json"
    dataset.tokenizer.save(tokenizer_path)
    print(f"Tokenizer saved to {tokenizer_path}")

    # ── Generate a short sample so you can hear the model ────
    print()
    print("Sample generation (temperature=0.8, 200 chars):")
    print("─" * 60)
    model.eval()
    prompt = torch.tensor([[dataset.tokenizer.BOS_ID]], dtype=torch.long)
    out = model.generate(prompt, max_new_tokens=200, temperature=0.8, top_k=40)
    text = dataset.tokenizer.decode(out[0].tolist(), skip_special=True)
    print(text)
    print("─" * 60)


# ── CLI ───────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the tiny LLM on a text corpus.")
    p.add_argument("--data-file", default="data/raw/shakespeare.txt",
                   help="Path to the training text file")
    p.add_argument("--max-steps", type=int, default=2000,
                   help="Number of gradient steps (default 2000)")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Batch size (default 16)")
    p.add_argument("--seq-len", type=int, default=128,
                   help="Sequence length per sample (default 128)")
    p.add_argument("--lr", type=float, default=3e-4,
                   help="Peak learning rate (default 3e-4)")
    p.add_argument("--weight-decay", type=float, default=0.1,
                   help="AdamW weight decay (default 0.1)")
    p.add_argument("--warmup-steps", type=int, default=100,
                   help="Linear warmup steps (default 100)")
    p.add_argument("--min-lr-ratio", type=float, default=0.1,
                   help="LR at end of cosine decay, as fraction of peak (default 0.1)")
    p.add_argument("--log-every", type=int, default=50,
                   help="Print training loss every N steps (default 50)")
    p.add_argument("--eval-every", type=int, default=250,
                   help="Run eval on train+val every N steps (default 250)")
    p.add_argument("--save-every", type=int, default=500,
                   help="Save a checkpoint every N steps (default 500)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default 42)")
    p.add_argument("--resume", type=str, default=None,
                   help="Resume from a checkpoint file")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())