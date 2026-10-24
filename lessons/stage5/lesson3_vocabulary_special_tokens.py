"""
=============================================================================
STAGE FIVE - LESSON 3: VOCABULARY AND SPECIAL TOKENS
File: lessons/stage5/lesson3_vocabulary_special_tokens.py
=============================================================================
A Vocabulary is a bidirectional mapping between textual tokens and integer IDs:
- Token to ID (Encoding):   "hello" -> 42
- ID to Token (Decoding):   42 -> "hello"

Special Tokens are non-linguistic markers that control Transformer behavior:
1. `<PAD>` (ID 0): Pads variable-length sequences so they can be grouped into
   uniform rectangular batch tensors.
2. `<UNK>` (ID 1): Represents words not present in the vocabulary.
3. `<BOS>` (ID 2): "Beginning of Sequence" — prompts the model that a generation starts.
4. `<EOS>` (ID 3): "End of Sequence" — the most critical generation token! It signals
   to the text generator that the answer is complete and generation should halt.
=============================================================================
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


class Vocabulary:
    """Bidirectional Vocabulary supporting special tokens and encoding/decoding."""
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"

    def __init__(self, min_freq: int = 1):
        self.min_freq = min_freq
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

        # Register special tokens with fixed reserved IDs
        self.special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]
        for idx, token in enumerate(self.special_tokens):
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

        self.pad_id = self.token_to_id[self.PAD_TOKEN]
        self.unk_id = self.token_to_id[self.UNK_TOKEN]
        self.bos_id = self.token_to_id[self.BOS_TOKEN]
        self.eos_id = self.token_to_id[self.EOS_TOKEN]

    def build_from_corpus(self, tokenized_texts: List[List[str]]):
        """Builds vocabulary mapping from tokenized sentences."""
        from collections import Counter
        counts = Counter()
        for tokens in tokenized_texts:
            counts.update(tokens)

        # Assign integer IDs to tokens meeting min_freq
        current_id = len(self.token_to_id)
        for token, count in counts.items():
            if count >= self.min_freq and token not in self.token_to_id:
                self.token_to_id[token] = current_id
                self.id_to_token[current_id] = token
                current_id += 1

    def encode(self, tokens: List[str], add_special_tokens: bool = False) -> List[int]:
        """Converts tokens to IDs. Unseen tokens map to unk_id."""
        ids = [self.token_to_id.get(tok, self.unk_id) for tok in tokens]
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> List[str]:
        """Converts IDs back to tokens."""
        tokens = []
        for i in ids:
            tok = self.id_to_token.get(i, self.UNK_TOKEN)
            if skip_special_tokens and tok in self.special_tokens:
                continue
            tokens.append(tok)
        return tokens

    def __len__(self) -> int:
        return len(self.token_to_id)


def pad_sequences(sequences: List[List[int]], pad_value: int = 0, max_len: Optional[int] = None) -> List[List[int]]:
    """
    Pads a batch of variable-length integer sequences to uniform length.
    """
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)

    padded_batch = []
    for seq in sequences:
        if len(seq) < max_len:
            padded_seq = seq + [pad_value] * (max_len - len(seq))
        else:
            padded_seq = seq[:max_len]
        padded_batch.append(padded_seq)
    return padded_batch


def demonstrate_vocabulary():
    """Demonstrates vocabulary construction, encoding, unknown tokens, and batch padding."""
    print("=" * 75)
    print("STAGE FIVE - LESSON 3: VOCABULARY & SPECIAL TOKENS")
    print("=" * 75)

    corpus = [
        ["the", "quick", "brown", "fox"],
        ["the", "lazy", "dog"],
        ["a", "brown", "dog", "barks"]
    ]

    vocab = Vocabulary(min_freq=1)
    vocab.build_from_corpus(corpus)

    print(f"Total Vocabulary Size: {len(vocab)} tokens")
    print("Special Tokens:")
    print(f"  PAD_ID = {vocab.pad_id} ('{vocab.PAD_TOKEN}')")
    print(f"  UNK_ID = {vocab.unk_id} ('{vocab.UNK_TOKEN}')")
    print(f"  BOS_ID = {vocab.bos_id} ('{vocab.BOS_TOKEN}')")
    print(f"  EOS_ID = {vocab.eos_id} ('{vocab.EOS_TOKEN}')")

    # Encode with unknown word
    test_sentence = ["the", "brown", "bear", "jumps"]  # 'bear' and 'jumps' are unknown
    encoded = vocab.encode(test_sentence, add_special_tokens=True)
    decoded = vocab.decode(encoded, skip_special_tokens=False)

    print(f"\nEncoding test sentence: {test_sentence}")
    print(f"Encoded IDs: {encoded}")
    print(f"Decoded back: {decoded}")
    print("Notice: 'bear' and 'jumps' were automatically mapped to <UNK> (ID 1)!")

    # Demonstrate Batch Padding
    seq1 = vocab.encode(["the", "dog"])                 # Length 2
    seq2 = vocab.encode(["the", "quick", "brown", "fox"]) # Length 4
    padded = pad_sequences([seq1, seq2], pad_value=vocab.pad_id)

    print("\nDemonstrating Batch Padding:")
    print(f"Sequence 1 (len 2): {seq1} -> Padded: {padded[0]}")
    print(f"Sequence 2 (len 4): {seq2} -> Padded: {padded[1]}")
    print("Both sequences now have identical length (4) and can form a 2x4 Tensor.")

    return vocab, encoded, padded


if __name__ == "__main__":
    demonstrate_vocabulary()
    print("\n" + "=" * 75)
    print("LESSON 3 COMPLETED SUCCESSFULLY!")
    print("=" * 75)
