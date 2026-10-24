"""
=============================================================================
STAGE FIVE - LESSON 1: TEXT PREPROCESSING AND NORMALIZATION
File: lessons/stage5/lesson1_text_preprocessing.py
=============================================================================
Computers do not understand raw strings or characters natively; they understand numbers.
Before text can be turned into numbers (tokens), it must undergo rigorous
preprocessing and normalization.

If raw text is dirty (erratic whitespace, hidden control characters, mixed encodings),
the vocabulary will explode with duplicate words ("Hello", "hello", "hello  ", "hello!").

This lesson demonstrates:
1. Unicode normalization (combining accents, compatibility characters).
2. Whitespace collapsing and control character stripping.
3. Case folding vs. case preservation (trade-offs in LLMs).
4. Punctuation segmentation.
=============================================================================
"""

import sys
import re
import unicodedata
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def normalize_unicode(text: str) -> str:
    """
    Normalizes Unicode representations using NFKD (Compatibility Decomposition).
    Ensures identical-looking characters use identical byte representations.
    Example: 'cafe\u0301' (e + combining acute accent) becomes 'cafe'.
    """
    normalized = unicodedata.normalize("NFKD", text)
    # Strip combining accent marks for simplified ASCII-focused vocabularies
    stripped = "".join(c for c in normalized if not unicodedata.combining(c))
    return stripped


def clean_whitespace_and_controls(text: str) -> str:
    """
    Removes invisible control characters and collapses redundant whitespace.
    Tabs, multiple spaces, and carriage returns are normalized into clean single spaces.
    """
    # Remove ASCII control characters (0x00 to 0x1F) except newline and tab
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
    # Collapse multiple consecutive spaces or tabs into a single space
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse multiple consecutive newlines into at most two newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def preprocess_text(text: str, lowercase: bool = True, strip_accents: bool = True) -> str:
    """
    Complete text normalization pipeline for educational language modeling.
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected string input, got {type(text).__name__}")

    # Step 1: Whitespace & control characters
    cleaned = clean_whitespace_and_controls(text)

    # Step 2: Unicode normalization
    if strip_accents:
        cleaned = normalize_unicode(cleaned)

    # Step 3: Case normalization
    if lowercase:
        cleaned = cleaned.lower()

    # Step 4: Ensure punctuation has clean spacing so tokens separate cleanly
    cleaned = re.sub(r"([.,!?;:()\"'])", r" \1 ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def demonstrate_preprocessing():
    """Demonstrates dirty real-world text being transformed into clean model-ready text."""
    print("=" * 75)
    print("STAGE FIVE - LESSON 1: TEXT PREPROCESSING & NORMALIZATION")
    print("=" * 75)

    sample_dirty_text = (
        "  Hello,   WORLD!!  \t This is an AI lesson.\n\n\n\n"
        "Let's test   caf\u00e9 (with accents)  and    extra    spaces!   "
    )

    print("Raw Dirty Input Text:")
    print(repr(sample_dirty_text))

    cleaned_text = preprocess_text(sample_dirty_text, lowercase=True, strip_accents=True)
    print("\nCleaned & Normalized Output:")
    print(repr(cleaned_text))
    print(f"Readable String: \"{cleaned_text}\"")

    print("\nKey Takeaways:")
    print("1. Redundant spaces and tabs were collapsed into single spaces.")
    print("2. Hidden newlines were standardized.")
    print("3. 'caf\u00e9' was normalized to 'cafe'.")
    print("4. Punctuation marks (',', '!', '.') were cleanly separated so tokenizers")
    print("   can process them as independent grammatical markers.")

    return cleaned_text


if __name__ == "__main__":
    demonstrate_preprocessing()
    print("\n" + "=" * 75)
    print("LESSON 1 COMPLETED SUCCESSFULLY!")
    print("=" * 75)
