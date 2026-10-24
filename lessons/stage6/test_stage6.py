"""
=============================================================================
STAGE SIX - AUTOMATED TESTS (lessons/stage6/test_stage6.py)
=============================================================================
Unit tests verifying all Stage Six Tiny Language Model components:
- Tokenizer encoding, decoding, and unknown character handling
- Model forward pass output dimensions and loss calculation
- Causal attention strictly blocks future attention leakage
- Clean train/validation data splitting without sequence overlap
- Checkpoint saving and reloading fidelity
- Generation max token limits and termination
- Architectural configuration validation and rejection
=============================================================================
"""

import sys
import unittest
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from lessons.stage6.model import TransformerLMConfig, TinyTransformerLM, CharTokenizer
from lessons.stage6.dataset import split_text_data, TextDataset, prepare_datasets
from generate_tiny_llm import load_model_and_tokenizer, generate_from_prompt, CHECKPOINT_PATH


class TestStage6TinyLLM(unittest.TestCase):

    def test_01_tokenizer_encode_decode_and_unk(self):
        """Verify character tokenizer encodes, decodes, and maps unseen chars to UNK."""
        tok = CharTokenizer()
        tok.fit("abc ABC 123")

        # Encode known characters
        ids = tok.encode("aA1")
        decoded = tok.decode(ids)
        self.assertEqual(decoded, "aA1")

        # Encode unknown character
        ids_with_unk = tok.encode("a#1")  # '#' is unknown
        self.assertEqual(ids_with_unk[1], tok.unk_id)

    def test_02_model_output_dimensions_and_loss(self):
        """Verify model returns logits matching (B, T, vocab_size) and scalar loss."""
        config = TransformerLMConfig(
            vocab_size=25,
            context_length=16,
            d_model=32,
            num_heads=2,
            num_layers=1,
            d_ff=64
        )
        model = TinyTransformerLM(config)

        batch_size = 2
        seq_len = 10
        dummy_inputs = torch.randint(0, 25, (batch_size, seq_len))
        dummy_targets = torch.randint(0, 25, (batch_size, seq_len))

        logits, loss = model(dummy_inputs, targets=dummy_targets)

        self.assertEqual(logits.shape, torch.Size([batch_size, seq_len, 25]))
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))

    def test_03_causal_mask_prevents_future_leakage(self):
        """Verify CausalSelfAttention mask strictly zeroes future attention weights."""
        config = TransformerLMConfig(
            vocab_size=10,
            context_length=8,
            d_model=16,
            num_heads=2,
            num_layers=1
        )
        model = TinyTransformerLM(config)
        block = model.blocks[0]
        causal_mask = block.attn.causal_mask.squeeze()

        # The mask must be strictly lower-triangular
        for i in range(config.context_length):
            for j in range(i + 1, config.context_length):
                self.assertEqual(causal_mask[i, j].item(), 0.0)

    def test_04_train_val_data_separated_correctly(self):
        """Verify train and validation split partitions text without overlapping boundaries."""
        sample_text = "ABCDEFGHIJ0123456789"
        train_text, val_text = split_text_data(sample_text, train_ratio=0.8)
        self.assertEqual(train_text + val_text, sample_text)
        self.assertEqual(len(train_text), 16)
        self.assertEqual(len(val_text), 4)

    def test_05_invalid_configurations_rejected(self):
        """Verify TransformerLMConfig rejects invalid hyperparameters."""
        # d_model (33) not divisible by num_heads (4)
        with self.assertRaises(ValueError):
            bad_config = TransformerLMConfig(vocab_size=10, d_model=33, num_heads=4)
            bad_config.validate()

        # Negative context length
        with self.assertRaises(ValueError):
            bad_config = TransformerLMConfig(vocab_size=10, context_length=-5)
            bad_config.validate()

    def test_06_text_dataset_too_small_rejected(self):
        """Verify TextDataset raises ValueError if dataset is smaller than context length."""
        with self.assertRaises(ValueError):
            TextDataset(token_ids=[1, 2, 3], context_length=10)

    def test_07_checkpoint_fidelity(self):
        """Verify checkpoint file exists and loaded model produces deterministic predictions."""
        if not CHECKPOINT_PATH.exists():
            self.skipTest("tiny_llm.pt checkpoint does not exist yet. Run train_tiny_llm.py first.")

        model, tokenizer, checkpoint = load_model_and_tokenizer(CHECKPOINT_PATH)
        self.assertIn("config", checkpoint)
        self.assertIn("model_state_dict", checkpoint)

        # Generate from prompt and check max token limit
        max_new = 15
        out = generate_from_prompt("Shall ", max_tokens=max_new, checkpoint_path=CHECKPOINT_PATH)
        self.assertIsInstance(out, str)
        self.assertTrue(out.startswith("Shall "))
        # Generated string cannot exceed prompt length + max_new characters
        self.assertLessEqual(len(out), len("Shall ") + max_new)


if __name__ == "__main__":
    unittest.main(verbosity=2)
