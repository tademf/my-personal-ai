"""
Conversation fine-tuning.

Takes the Shakespeare-pretrained model and teaches it to respond
in a User:/Assistant: format.

Usage:
    python src/finetune.py
    python src/finetune.py --steps 300 --lr 1e-4

Result: a fine-tuned model saved to models/checkpoints/finetuned.pt
that can be prompted with:
    <|user|>hi<|assistant|>
and will produce a short reply.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from tokenizer import CharTokenizer
from model import ModelConfig, TinyLLM


# ── Formatting ────────────────────────────────────────────────

USER_MARKER = "<|user|>"
ASSISTANT_MARKER = "<|assistant|>"
END_MARKER = "<|end|>"


def format_conversation(user: str, assistant: str) -> str:
    """Format one (user, assistant) pair as a single training string."""
    return f"{USER_MARKER}{user}{ASSISTANT_MARKER}{assistant}{END_MARKER}"


def format_prompt_for_inference(user: str) -> str:
    """Format a user message as a prompt — the model should continue with the answer."""
    return f"{USER_MARKER}{user}{ASSISTANT_MARKER}"


# ── Build fine-tuning dataset ─────────────────────────────────


def load_conversations(jsonl_path: str | Path) -> list[dict]:
    """Load (user, assistant) pairs from a JSONL file."""
    pairs = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def build_dataset(
    jsonl_path: str | Path,
    tokenizer: CharTokenizer,
    max_seq_len: int,
) -> list[torch.Tensor]:
    """
    Convert the JSONL file into a list of tokenized sequences.

    Each sequence ends with END_MARKER, so the model learns to stop
    after the assistant's response.
    """
    pairs = load_conversations(jsonl_path)
    print(f"Loaded {len(pairs)} conversation pairs")

    # Extend tokenizer vocab with any new characters
    all_text = "".join(format_conversation(p["user"], p["assistant"]) for p in pairs)
    tokenizer.fit(all_text)  # adds new characters (like <, |, >) if missing

    seqs = []
    for p in pairs:
        text = format_conversation(p["user"], p["assistant"])
        ids = tokenizer.encode(text)
        ids = ids[: max_seq_len + 1]  # +1 because targets shift by 1
        if len(ids) < 2:
            continue
        seqs.append(torch.tensor(ids, dtype=torch.long))
    return seqs


def get_batch(
    seqs: list[torch.Tensor],
    batch_size: int,
    pad_token_id: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a batch of sequences, pad to the longest in the batch.

    Targets are inputs shifted by one position. Padding positions in
    targets are set to -100 so cross-entropy ignores them.
    """
    batch = random.sample(seqs, min(batch_size, len(seqs)))
    max_len = max(len(s) for s in batch) - 1  # -1 because we need shift

    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    targets = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, seq in enumerate(batch):
        L = len(seq) - 1
        input_ids[i, :L] = seq[:-1]
        targets[i, :L] = seq[1:]

    return input_ids, targets


# ── Training ──────────────────────────────────────────────────


def train(args: argparse.Namespace) -> None:
    ckpt_dir = Path("models/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Load base model from best.pt or final.pt
    base_ckpt = Path(args.base_checkpoint)
    if not base_ckpt.exists():
        raise FileNotFoundError(
            f"Base checkpoint not found: {base_ckpt}\n"
            f"Run 'python src/train.py' first."
        )

    print(f"Loading base checkpoint: {base_ckpt}")
    ckpt = torch.load(base_ckpt, map_location="cpu", weights_only=False)
    config: ModelConfig = ckpt["config"]

    # Load tokenizer from pretrained checkpoint, then extend it
    print("Loading conversations and extending tokenizer…")
    tok_path = base_ckpt.parent / "tokenizer.json"
    tokenizer = CharTokenizer.load(tok_path)
    base_vocab_size = tokenizer.vocab_size

    seqs = build_dataset(args.data_file, tokenizer, config.max_seq_len)
    new_vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {base_vocab_size} → {new_vocab_size}")

    # If vocab grew, we need to resize the model's embedding tables
    if new_vocab_size != config.vocab_size:
        print(f"  (model embedding resize: {config.vocab_size} → {new_vocab_size})")
        old_config = config
        new_config = ModelConfig(
            vocab_size=new_vocab_size,
            max_seq_len=old_config.max_seq_len,
            d_model=old_config.d_model,
            n_heads=old_config.n_heads,
            n_layers=old_config.n_layers,
            d_ff=old_config.d_ff,
            dropout=old_config.dropout,
        )
        model = TinyLLM(new_config)
        config = new_config

        # Copy old weights into the new model
        old_state = ckpt["model_state_dict"]
        new_state = model.state_dict()
        for name, param in old_state.items():
            if "token_embedding" in name or "lm_head" in name:
                # These have shape [vocab_size, d_model] — copy first N rows
                if param.shape[0] == base_vocab_size:
                    new_state[name][:base_vocab_size] = param
            else:
                # Non-vocab layers have identical shapes — copy directly
                if new_state[name].shape == param.shape:
                    new_state[name] = param
        model.load_state_dict(new_state)
        print("  ✓ pretrained weights restored into resized model")
    else:
        model = TinyLLM(config)
        model.load_state_dict(ckpt["model_state_dict"])

    model.train()

    # Optimizer — lower LR than pretraining since we're fine-tuning
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.05,
        betas=(0.9, 0.95),
    )

    print(f"\nFine-tuning for {args.steps} steps")
    print(f"Batch size: {args.batch_size}, LR: {args.lr}")
    print()

    t0 = time.time()
    for step in range(1, args.steps + 1):
        input_ids, targets = get_batch(seqs, batch_size=args.batch_size)
        _, loss = model(input_ids, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.log_every == 0:
            elapsed = time.time() - t0
            print(f"step {step:>4}/{args.steps} | loss {loss.item():.4f} | {elapsed:.1f}s")

    print(f"\nFine-tuning complete in {time.time() - t0:.1f}s")

    # Save model + extended tokenizer
    out_path = ckpt_dir / "finetuned.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "step": args.steps,
        },
        out_path,
    )
    print(f"Fine-tuned model saved: {out_path}")

    new_tok_path = ckpt_dir / "tokenizer_ft.json"
    tokenizer.save(new_tok_path)
    print(f"Extended tokenizer saved: {new_tok_path}")

    # ── Sample outputs ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SAMPLE CONVERSATIONS")
    print("=" * 60)

    model.eval()
    test_prompts = [
        "hi",
        "what is python",
        "who are you",
        "how do i sort a list in python",
        "thank you",
        "goodbye",
    ]

    def clean_reply(raw: str) -> str:
        """Extract the assistant reply from generated text."""
        # Split on assistant marker
        if ASSISTANT_MARKER in raw:
            reply = raw.split(ASSISTANT_MARKER, 1)[1]
        else:
            reply = raw
        # Stop at end marker or user marker
        for stop in (END_MARKER, USER_MARKER):
            if stop in reply:
                reply = reply.split(stop, 1)[0]
        # Remove any marker fragments that slipped through
        for frag in ("<|", "|>", "|", "<", ">", "="):
            reply = reply.replace(frag, "")
        return reply.strip()

    for user_msg in test_prompts:
        prompt = format_prompt_for_inference(user_msg)
        ids = tokenizer.encode(prompt, add_bos=True)
        x = torch.tensor([ids], dtype=torch.long)

        with torch.no_grad():
            out = model.generate(
                x,
                max_new_tokens=80,
                temperature=0.4,   # was 0.7 — more deterministic
                top_k=40,           # was 20 — less likely to sample garbage
            )

        generated = tokenizer.decode(out[0].tolist(), skip_special=True)
        reply = clean_reply(generated)

        print(f"\nUser: {user_msg}")
        print(f"Assistant: {reply}")

    print("\n" + "=" * 60)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-file", default="data/raw/conversations.jsonl")
    p.add_argument("--base-checkpoint", default="models/checkpoints/best.pt")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())