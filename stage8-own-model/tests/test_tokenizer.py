"""Unit tests for the character-level tokenizer."""

import sys
from pathlib import Path

# Make `src` importable when running `pytest` from the project root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from tokenizer import CharTokenizer


def test_special_token_ids():
    """Special tokens have fixed, documented IDs."""
    assert CharTokenizer.BOS_ID == 0
    assert CharTokenizer.EOS_ID == 1
    assert CharTokenizer.PAD_ID == 2
    assert CharTokenizer.UNK_ID == 3


def test_fit_builds_vocabulary():
    tok = CharTokenizer()
    tok.fit("hello world")
    # 8 unique chars: h,e,l,o,space,w,r,d
    assert len(tok.char_to_id) == 8
    assert tok.vocab_size == 8 + 4  # 8 chars + 4 special
def test_roundtrip_encode_decode():
    tok = CharTokenizer()
    text = "the quick brown fox"
    tok.fit(text)
    ids = tok.encode(text)
    assert tok.decode(ids) == text


def test_encode_is_deterministic():
    tok = CharTokenizer()
    tok.fit("abcabc")
    assert tok.encode("abc") == tok.encode("abc")


def test_unknown_character_maps_to_unk():
    tok = CharTokenizer()
    tok.fit("abc")
    # 'z' was not in training text
    ids = tok.encode("abz")
    assert ids[-1] == CharTokenizer.UNK_ID


def test_bos_and_eos_flags():
    tok = CharTokenizer()
    tok.fit("ab")
    ids = tok.encode("ab", add_bos=True, add_eos=True)
    assert ids[0] == CharTokenizer.BOS_ID
    assert ids[-1] == CharTokenizer.EOS_ID
    assert tok.decode(ids) == "ab"  # specials hidden by default


def test_save_and_load(tmp_path):
    tok = CharTokenizer()
    tok.fit("hello world")
    path = tmp_path / "tok.json"
    tok.save(path)

    loaded = CharTokenizer.load(path)
    assert loaded.char_to_id == tok.char_to_id
    assert loaded.vocab_size == tok.vocab_size
    assert loaded.encode("hello") == tok.encode("hello")


def test_fit_is_idempotent_on_same_text():
    tok = CharTokenizer()
    tok.fit("abc")
    size_before = tok.vocab_size
    tok.fit("abc")
    assert tok.vocab_size == size_before


def test_fit_extends_vocabulary():
    tok = CharTokenizer()
    tok.fit("abc")
    size1 = tok.vocab_size
    tok.fit("xyz")
    assert tok.vocab_size == size1 + 3  # 3 new chars


def test_decode_unknown_id_is_empty():
    """An ID not in the vocab decodes to nothing (safe default)."""
    tok = CharTokenizer()
    tok.fit("ab")
    assert tok.decode([9999]) == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])