"""
Character-level tokenizer.

Each unique character in the training text is assigned an integer ID.
Special tokens (BOS, EOS, PAD, UNK) get reserved IDs 0-3.
Regular characters get IDs starting at 4.

This is the simplest possible tokenizer:
- No unknown characters (any string can be encoded)
- Vocabulary is small (~100-200 chars for English)
- Sequences are longer than word-level (1 token per char)

It's ideal for learning because there are no heuristics to hide the
core mechanics. A production system would use BPE or WordPiece.
"""

from __future__ import annotations

import json
from pathlib import Path


class CharTokenizer:
    """Character-level tokenizer with special tokens."""

    # Reserved special-token IDs (fixed so saves are portable)
    BOS_ID = 0
    EOS_ID = 1
    PAD_ID = 2
    UNK_ID = 3

    SPECIAL_TOKENS = {
        "<bos>": BOS_ID,
        "<eos>": EOS_ID,
        "<pad>": PAD_ID,
        "<unk>": UNK_ID,
    }

    def __init__(self, char_to_id: dict[str, int] | None = None):
        """
        Args:
            char_to_id: Optional pre-existing mapping. If None, start empty
                        and call `fit()` on some text.
        """
        self.char_to_id: dict[str, int] = dict(char_to_id) if char_to_id else {}
        self.id_to_char: dict[int, str] = {v: k for k, v in self.char_to_id.items()}

    # ── Vocabulary building ─────────────────────────────────

    def fit(self, text: str) -> None:
        """
        Build the vocabulary from `text`.

        New characters are assigned IDs starting from the highest ID
        currently in use (including special tokens). Calling fit() twice
        with different text adds new characters without overwriting
        existing ones.
        """
        # FIX: start from the highest ID in use, not from 4.
        # Without this, a second fit() call would overwrite IDs of
        # existing characters, corrupting the vocabulary.
        existing_ids = (
            list(self.SPECIAL_TOKENS.values()) + list(self.char_to_id.values())
        )
        next_id = max(existing_ids) + 1 if existing_ids else 4

        for ch in sorted(set(text)):
            if ch not in self.char_to_id:
                self.char_to_id[ch] = next_id
                self.id_to_char[next_id] = ch
                next_id += 1

    @property
    def vocab_size(self) -> int:
        return len(self.char_to_id) + len(self.SPECIAL_TOKENS)

    # ── Encode / decode ─────────────────────────────────────

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """
        Convert `text` to a list of integer token IDs.

        Unknown characters map to UNK_ID. They should not appear in
        practice if `fit()` was called on the training corpus.
        """
        ids = []
        if add_bos:
            ids.append(self.BOS_ID)
        for ch in text:
            ids.append(self.char_to_id.get(ch, self.UNK_ID))
        if add_eos:
            ids.append(self.EOS_ID)
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """
        Convert token IDs back to a string.

        If `skip_special=True`, special tokens are omitted from the output.
        """
        chars = []
        for i in ids:
            if skip_special and i in self.id_to_char and i in self.SPECIAL_TOKENS.values():
                # unreachable — special IDs are not in id_to_char — kept for clarity
                continue
            if i in self.id_to_char:
                chars.append(self.id_to_char[i])
            elif not skip_special:
                # Reverse lookup for special tokens
                for tok, tid in self.SPECIAL_TOKENS.items():
                    if tid == i:
                        chars.append(tok)
                        break
        return "".join(chars)

    # ── Persistence ─────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save vocabulary to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "type": "char",
            "char_to_id": self.char_to_id,
            "special_tokens": self.SPECIAL_TOKENS,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        """Load vocabulary from a JSON file."""
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("type") != "char":
            raise ValueError(f"Not a char tokenizer file: {path}")
        return cls(char_to_id=data["char_to_id"])

    # ── Convenience ─────────────────────────────────────────

    def __repr__(self) -> str:
        return f"CharTokenizer(vocab_size={self.vocab_size})"