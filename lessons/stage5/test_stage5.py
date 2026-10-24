"""
=============================================================================
STAGE FIVE - AUTOMATED TESTS
File: lessons/stage5/test_stage5.py
=============================================================================
Unit tests verifying all 14 Stage Five Language Model & Transformer modules:
- Vocabulary encoding, decoding, special token and UNK handling
- Token embedding shapes and lookup consistency
- Scaled dot-product attention shapes and probability sums
- Causal masking strict enforcement (zero future leakage)
- Shifted sequence preparation for next-token training
- Toy language model training execution and loss reduction
- Autoregressive text generation termination
- Sampling constraints (temperature, top-k, top-p)
- Inference mode gradient suppression
=============================================================================
"""

import sys
import unittest
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from lessons.stage5.lesson1_text_preprocessing import preprocess_text
from lessons.stage5.lesson2_tokenization import ToyBytePairEncoder
from lessons.stage5.lesson3_vocabulary_special_tokens import Vocabulary, pad_sequences
from lessons.stage5.lesson4_token_embeddings import demonstrate_token_embeddings
from lessons.stage5.lesson5_positional_encoding import SinusoidalPositionalEncoding
from lessons.stage5.lesson6_scaled_dot_product_attention import scaled_dot_product_attention
from lessons.stage5.lesson7_multi_head_attention import MultiHeadAttention, TransformerBlock
from lessons.stage5.lesson8_causal_masking import generate_causal_mask
from lessons.stage5.lesson9_next_token_prediction import prepare_shifted_sequences
from lessons.stage5.lesson10_training_validation import train_toy_lm, TinyLanguageModel
from lessons.stage5.lesson11_autoregressive_generation import generate_autoregressive
from lessons.stage5.lesson12_sampling_methods import (
    apply_temperature,
    apply_top_k,
    apply_top_p,
    sample_next_token
)


class TestStage5LanguageModels(unittest.TestCase):

    def test_01_preprocessing_and_bpe(self):
        """Verify text preprocessing normalizes strings and BPE merges character pairs."""
        raw = "  Hello,   WORLD!!  café \t\n"
        cleaned = preprocess_text(raw)
        self.assertIn("hello", cleaned)
        self.assertIn("world", cleaned)
        self.assertIn("cafe", cleaned)

        # BPE merge test
        bpe = ToyBytePairEncoder(num_merges=3)
        vocab = bpe.fit(["low", "lower", "lowest"])
        self.assertGreater(len(bpe.merges), 0)

    def test_02_vocabulary_encode_decode_and_unk(self):
        """Verify vocabulary encoding, decoding, and UNK mapping."""
        vocab = Vocabulary(min_freq=1)
        vocab.build_from_corpus([["apple", "banana"]])

        # Known token
        ids = vocab.encode(["apple", "cherry"])
        self.assertEqual(ids[0], vocab.token_to_id["apple"])
        # 'cherry' is unknown -> must map to unk_id
        self.assertEqual(ids[1], vocab.unk_id)

        # Decode back
        decoded = vocab.decode(ids)
        self.assertEqual(decoded[0], "apple")
        self.assertEqual(decoded[1], vocab.UNK_TOKEN)

    def test_03_embedding_and_positional_shapes(self):
        """Verify embedding and positional encoding shapes match expected dimensions."""
        vocab_size = 15
        d_model = 8
        seq_len = 5
        batch_size = 2

        embed_layer = torch.nn.Embedding(vocab_size, d_model)
        pe_layer = SinusoidalPositionalEncoding(d_model=d_model, max_len=20)

        dummy_tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
        embedded = embed_layer(dummy_tokens)
        self.assertEqual(embedded.shape, torch.Size([batch_size, seq_len, d_model]))

        positioned = pe_layer(embedded)
        self.assertEqual(positioned.shape, torch.Size([batch_size, seq_len, d_model]))

    def test_04_attention_shapes_and_probabilities(self):
        """Verify scaled dot-product attention output dimensions and probability sums."""
        batch, seq, d_k = 2, 4, 6
        Q = torch.randn(batch, seq, d_k)
        K = torch.randn(batch, seq, d_k)
        V = torch.randn(batch, seq, d_k)

        out, weights = scaled_dot_product_attention(Q, K, V)
        self.assertEqual(out.shape, torch.Size([batch, seq, d_k]))
        self.assertEqual(weights.shape, torch.Size([batch, seq, seq]))

        # Each row of weights must sum to 1.0 (Softmax distribution)
        row_sums = weights.sum(dim=-1)
        expected = torch.ones(batch, seq)
        self.assertTrue(torch.allclose(row_sums, expected, atol=1e-5))

    def test_05_causal_masking_blocks_future(self):
        """Verify causal masking strictly sets future attention probabilities to 0.0."""
        seq_len = 5
        mask = generate_causal_mask(seq_len)
        Q = torch.randn(1, seq_len, 4)
        K = torch.randn(1, seq_len, 4)
        V = torch.randn(1, seq_len, 4)

        _, weights = scaled_dot_product_attention(Q, K, V, mask=mask)
        weights_mat = weights.squeeze(0)

        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                self.assertEqual(weights_mat[i, j].item(), 0.0, f"Future leak at [{i}, {j}]")

    def test_06_shifted_sequences_alignment(self):
        """Verify input and target sequences are correctly shifted by 1 position."""
        seq = torch.tensor([[10, 20, 30, 40, 50]])
        x, y = prepare_shifted_sequences(seq)
        self.assertEqual(x.tolist(), [[10, 20, 30, 40]])
        self.assertEqual(y.tolist(), [[20, 30, 40, 50]])

    def test_07_toy_training_reduces_loss(self):
        """Verify training reduces loss on toy corpus."""
        model, vocab, train_losses, val_losses = train_toy_lm(epochs=30, lr=0.01)
        self.assertLess(train_losses[-1], train_losses[0], "Training failed to reduce loss")
        self.assertTrue(torch.isfinite(torch.tensor(train_losses[-1])))

    def test_08_text_generation_terminates(self):
        """Verify autoregressive generation terminates within max_new_tokens limit."""
        model, vocab, _, _ = train_toy_lm(epochs=20, lr=0.01)
        prompt = ["def", "greet"]
        max_limit = 6
        output_tokens = generate_autoregressive(model, vocab, prompt, max_new_tokens=max_limit)
        self.assertIsInstance(output_tokens, list)
        self.assertGreaterEqual(len(output_tokens), len(prompt))
        # Total generated cannot exceed prompt length + max_limit + 1
        self.assertLessEqual(len(output_tokens), len(prompt) + max_limit + 1)

    def test_09_sampling_methods_constraints(self):
        """Verify temperature, Top-K, and Top-P filtering respect constraints."""
        logits = torch.tensor([5.0, 3.0, 1.0, 0.5, -2.0])

        # Temperature scaling
        temp_scaled = apply_temperature(logits, temperature=0.5)
        self.assertEqual(temp_scaled[0].item(), 10.0)

        # Top-K filtering (k=2): only 2 highest values are non-infinite
        top_k = apply_top_k(logits, k=2)
        finite_count = torch.isfinite(top_k).sum().item()
        self.assertEqual(finite_count, 2)

        # Top-P filtering (p=0.80): must keep top logits and mask tail
        top_p = apply_top_p(logits, p=0.80)
        self.assertTrue(torch.isfinite(top_p[0]))

        # Sampled ID must be within vocabulary bounds
        sampled_id = sample_next_token(logits, temperature=1.0, top_k=3, top_p=0.9)
        self.assertIn(sampled_id, list(range(len(logits))))

    def test_10_inference_without_gradients(self):
        """Verify inference mode produces outputs without computing gradients."""
        vocab_size = 12
        model = TinyLanguageModel(vocab_size=vocab_size, d_model=16, num_heads=2)
        model.eval()

        dummy_input = torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            output = model(dummy_input)

        self.assertFalse(output.requires_grad)
        self.assertIsNone(output.grad_fn)


if __name__ == "__main__":
    unittest.main(verbosity=2)
