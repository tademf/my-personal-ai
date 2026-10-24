"""
=============================================================================
STAGE FIVE - LESSON 13: PRETRAINING VERSUS FINE-TUNING
File: lessons/stage5/lesson13_pretraining_vs_finetuning.py
=============================================================================
Modern AI assistants are not created in a single step; they are built in
two distinct evolutionary phases:

PHASE 1: PRETRAINING (Building the Base Model)
- Objective: Next-token prediction on trillions of words (web pages, books, code).
- What it learns: Grammar, world knowledge, common sense, coding patterns.
- Limitation: A base model is merely a text-completer, NOT a helpful chatbot.
  If you prompt a base model: "What is the capital of Italy?"
  It might complete: "What is the capital of France? What is the capital of Spain?"

PHASE 2: SUPERVISED FINE-TUNING (SFT / Instruction Tuning)
- Objective: Train the base model on curated Prompt-Response dialogues:
  Input:  "Human: Write a python function to add two numbers. Assistant:"
  Target: "def add(a, b): return a + b"
- What it learns: How to follow directions, be polite, formatting, and helpfulness.

DISCLAIMER:
This lesson uses a tiny 3-word demonstration to teach the mechanical difference.
It does NOT claim to reproduce industrial pretraining (which requires millions of
dollars of GPU compute).
=============================================================================
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from lessons.stage5.lesson10_training_validation import TinyLanguageModel
from lessons.stage5.lesson3_vocabulary_special_tokens import Vocabulary


def demonstrate_pretraining_vs_finetuning():
    """Demonstrates how fine-tuning adapts a model to instruction-following format."""
    print("=" * 75)
    print("STAGE FIVE - LESSON 13: PRETRAINING VS. FINE-TUNING")
    print("=" * 75)

    vocab = Vocabulary(min_freq=1)
    # Corpus contains both general knowledge tokens and instruction markers
    corpus = [
        ["the", "sky", "is", "blue"],
        ["user", ":", "sum", "assistant", ":", "add"],
        ["user", ":", "sub", "assistant", ":", "minus"]
    ]
    vocab.build_from_corpus(corpus)
    vocab_size = len(vocab)

    # 1. Base Pre-trained Model (Initialized and trained on general tokens)
    model = TinyLanguageModel(vocab_size=vocab_size, d_model=16, num_heads=2, d_ff=32)

    print("Phase 1: Pre-training establishes the model's base token representations.")
    print(f"Base Model Vocabulary Size: {vocab_size}")

    # 2. Supervised Fine-Tuning Step
    # We provide an instruction template: "user : sum assistant : add"
    instruction_seq = vocab.encode(["user", ":", "sum", "assistant", ":", "add"], add_special_tokens=True)
    input_ids = torch.tensor([instruction_seq[:-1]], dtype=torch.long)
    target_ids = torch.tensor([instruction_seq[1:]], dtype=torch.long)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    print("\nPhase 2: Fine-Tuning on Instruction-Response Example:")
    print(f"  Prompt tokens:   ['user', ':', 'sum']")
    print(f"  Target response: ['assistant', ':', 'add']")

    initial_loss = None
    for step in range(1, 21):
        model.train()
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
        loss.backward()
        optimizer.step()

        if step == 1:
            initial_loss = loss.item()

    print(f"  Initial SFT Loss: {initial_loss:.4f} -> Final SFT Loss: {loss.item():.4f}")
    assert loss.item() < initial_loss, "Fine-tuning failed to optimize on instruction format!"

    print("\nKey Takeaways:")
    print("1. Pre-training gives the model understanding of vocabulary and syntax.")
    print("2. Supervised Fine-Tuning (SFT) molds that knowledge into a helpful chat format.")
    print("3. In Stage Seven, we will run an existing pre-trained + instruction-tuned LLM locally via Ollama.")

    return model, initial_loss, loss.item()


if __name__ == "__main__":
    demonstrate_pretraining_vs_finetuning()
    print("\n" + "=" * 75)
    print("LESSON 13 COMPLETED SUCCESSFULLY!")
    print("=" * 75)
