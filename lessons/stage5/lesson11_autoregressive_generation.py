"""
=============================================================================
STAGE FIVE - LESSON 11: AUTOREGRESSIVE TEXT GENERATION
File: lessons/stage5/lesson11_autoregressive_generation.py
=============================================================================
How does ChatGPT or any Transformer actually output text?
Autoregressively — ONE TOKEN AT A TIME.

The Autoregressive Loop:
1. Provide an initial prompt (e.g., ["def", "greet"]).
2. Feed the prompt into the Transformer.
3. Extract the prediction logits from the LAST token position only.
4. Select the next token (Greedy Argmax or Sampling).
5. Append the new token to the sequence.
6. Check termination:
   - Did the model output the `<EOS>` (End-Of-Sequence) token? If yes, HALT.
   - Did we hit the maximum token limit (`max_new_tokens`)? If yes, HALT.
7. Repeat from Step 2 with the expanded sequence!
=============================================================================
"""

import sys
from pathlib import Path
from typing import List
import torch

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from lessons.stage5.lesson10_training_validation import train_toy_lm, TinyLanguageModel
from lessons.stage5.lesson3_vocabulary_special_tokens import Vocabulary


def generate_autoregressive(
    model: TinyLanguageModel,
    vocab: Vocabulary,
    prompt_tokens: List[str],
    max_new_tokens: int = 10
) -> List[str]:
    """
    Generates text token-by-token using greedy decoding.
    """
    model.eval()

    # Encode prompt tokens
    current_ids = vocab.encode(prompt_tokens, add_special_tokens=False)
    # Ensure BOS is at start
    if current_ids and current_ids[0] != vocab.bos_id:
        current_ids = [vocab.bos_id] + current_ids

    generated_ids = list(current_ids)

    with torch.no_grad():
        for step in range(max_new_tokens):
            input_tensor = torch.tensor([generated_ids], dtype=torch.long)

            # 1. Forward pass
            logits = model(input_tensor)  # (1, seq_len, vocab_size)

            # 2. Extract logits for the LAST token position only
            next_token_logits = logits[0, -1, :]  # (vocab_size,)

            # 3. Greedy selection: pick the token with the highest logit
            next_token_id = int(torch.argmax(next_token_logits).item())

            # 4. Append to generated sequence
            generated_ids.append(next_token_id)

            # 5. Halt if model generated <EOS>
            if next_token_id == vocab.eos_id:
                break

    # Decode IDs back to text tokens (skipping special tokens for readability)
    decoded_tokens = vocab.decode(generated_ids, skip_special_tokens=True)
    return decoded_tokens


def demonstrate_generation():
    """Demonstrates greedy autoregressive generation with our trained toy model."""
    print("=" * 75)
    print("STAGE FIVE - LESSON 11: AUTOREGRESSIVE TEXT GENERATION")
    print("=" * 75)

    # Train model on our tiny corpus
    model, vocab, _, _ = train_toy_lm(epochs=60, lr=0.01)

    # Prompt 1: ["def", "greet"] -> Training data had ["def", "greet", ":", "print", "hello"]
    prompt1 = ["def", "greet"]
    result1 = generate_autoregressive(model, vocab, prompt1, max_new_tokens=8)
    print(f"\nPrompt:    {' '.join(prompt1)}")
    print(f"Generated: {' '.join(result1)}")

    # Prompt 2: ["def", "compute"] -> Training data had ["def", "compute", ":", "return", "total"]
    prompt2 = ["def", "compute"]
    result2 = generate_autoregressive(model, vocab, prompt2, max_new_tokens=8)
    print(f"\nPrompt:    {' '.join(prompt2)}")
    print(f"Generated: {' '.join(result2)}")

    print("\nTakeaways:")
    print("1. Generation is iterative: each output token becomes part of the input for the next step.")
    print("2. The model learned both the grammatical structure (':') and vocabulary completions.")
    print("3. When max_new_tokens or <EOS> is reached, the generation terminates safely.")

    return result1, result2


if __name__ == "__main__":
    demonstrate_generation()
    print("\n" + "=" * 75)
    print("LESSON 11 COMPLETED SUCCESSFULLY!")
    print("=" * 75)
