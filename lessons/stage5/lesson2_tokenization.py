"""
=============================================================================
STAGE FIVE - LESSON 2: TOKENIZATION (CHARACTER, WORD, AND SUBWORD BPE)
File: lessons/stage5/lesson2_tokenization.py
=============================================================================
Tokenization is the process of breaking a continuous string of text into discrete
chunks called "Tokens". Each token is assigned a unique integer ID.

Three Primary Tokenization Strategies:
1. Character-Level Tokenization:
   - Vocab Size: Tiny (~50 to 128 characters).
   - Pros: Never encounters Out-Of-Vocabulary (OOV) words.
   - Cons: Sequences are 4x to 6x longer; model must learn basic spelling from scratch.

2. Word-Level Tokenization:
   - Vocab Size: Huge (50,000 to 1,000,000+ words).
   - Pros: Compact sequence length; words carry immediate meaning.
   - Cons: Massive memory footprint; fails on typos, rare words, or new slang.

3. Subword Tokenization (Byte-Pair Encoding / BPE):
   - The modern standard in GPT-2, GPT-4, Claude, and Llama!
   - Starts with basic characters and iteratively merges the most frequent adjacent
     character pairs into reusable subwords (e.g., 'play' + 'ing' -> 'playing').
=============================================================================
"""

import sys
from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def character_tokenize(text: str) -> List[str]:
    """Splits string into a list of single characters."""
    return list(text)


def word_tokenize(text: str) -> List[str]:
    """Splits string by whitespace into words."""
    return text.split()


class ToyBytePairEncoder:
    """
    A minimal, educational implementation of the Byte-Pair Encoding (BPE) algorithm.
    Demonstrates how subword merges are learned from text statistics.
    """
    def __init__(self, num_merges: int = 5):
        self.num_merges = num_merges
        self.merges: List[Tuple[str, str]] = []

    def get_stats(self, vocab: Dict[Tuple[str, ...], int]) -> Counter:
        """Counts frequency of all adjacent symbol pairs in the current vocabulary."""
        pairs = Counter()
        for word_tuple, freq in vocab.items():
            for i in range(len(word_tuple) - 1):
                pairs[(word_tuple[i], word_tuple[i + 1])] += freq
        return pairs

    def merge_vocab(self, pair: Tuple[str, str], vocab: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, ...], int]:
        """Replaces occurrences of (pair[0], pair[1]) with 'pair[0]pair[1]' across words."""
        new_vocab = {}
        first, second = pair
        replacement = first + second

        for word_tuple, freq in vocab.items():
            new_tuple = []
            i = 0
            while i < len(word_tuple):
                if i < len(word_tuple) - 1 and word_tuple[i] == first and word_tuple[i + 1] == second:
                    new_tuple.append(replacement)
                    i += 2
                else:
                    new_tuple.append(word_tuple[i])
                    i += 1
            new_vocab[tuple(new_tuple)] = freq
        return Counter(new_vocab)

    def fit(self, corpus_words: List[str]):
        """Learns the top BPE merges from a corpus of words."""
        # Represent each word as a tuple of characters with an end-of-word marker '_'
        vocab = Counter(tuple(list(word) + ["_"]) for word in corpus_words)

        print(f"Initial Vocabulary (Characters):")
        for word_tuple, count in vocab.most_common(5):
            print(f"  {word_tuple}: {count} times")

        for merge_idx in range(1, self.num_merges + 1):
            pairs = self.get_stats(vocab)
            if not pairs:
                break
            best_pair = pairs.most_common(1)[0][0]
            best_count = pairs[best_pair]
            self.merges.append(best_pair)
            vocab = self.merge_vocab(best_pair, vocab)
            print(f"Merge {merge_idx}: Combined '{best_pair[0]}' + '{best_pair[1]}' (Occurred {best_count} times)")

        print("\nFinal Subword Vocabulary Representation:")
        for word_tuple, count in vocab.most_common(5):
            print(f"  {word_tuple}: {count} times")

        return vocab


def demonstrate_tokenizers():
    """Demonstrates Character, Word, and BPE tokenization."""
    print("=" * 75)
    print("STAGE FIVE - LESSON 2: TOKENIZATION COMPARISON")
    print("=" * 75)

    sample_sentence = "low lower newest widest lowest"

    # 1. Character Tokenization
    char_tokens = character_tokenize(sample_sentence)
    print(f"1. Character Tokens ({len(char_tokens)} tokens):")
    print("  ", char_tokens[:15], "...")

    # 2. Word Tokenization
    word_tokens = word_tokenize(sample_sentence)
    print(f"\n2. Word Tokens ({len(word_tokens)} tokens):")
    print("  ", word_tokens)

    # 3. Subword BPE Tokenization
    print(f"\n3. Subword BPE Learning Demonstration:")
    training_corpus = ["low", "low", "low", "lower", "lower", "newest", "widest", "lowest", "lowest"]
    bpe = ToyBytePairEncoder(num_merges=4)
    bpe.fit(training_corpus)

    print("\nTakeaways:")
    print("1. Subword tokenization dynamically discovers common roots ('low') and suffixes ('er', 'est').")
    print("2. Rare or compound words like 'lowest' are broken into known pieces: ('low', 'est').")
    print("3. This allows modern LLMs to process arbitrary words without massive vocabularies.")

    return char_tokens, word_tokens, bpe


if __name__ == "__main__":
    demonstrate_tokenizers()
    print("\n" + "=" * 75)
    print("LESSON 2 COMPLETED SUCCESSFULLY!")
    print("=" * 75)
