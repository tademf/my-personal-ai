# Personal AI from Zero

A step-by-step implementation of an autonomous, local-first personal AI assistant. This repository traces the complete evolutionary arc of modern AI: from classical machine learning to deep neural networks, transformer architectures, local LLM orchestration (Ollama), and a full-stack personal assistant interface.

---

## 1. Project Overview & Philosophy
- **Local-first & private:** Designed to run on personal hardware without forced reliance on external cloud APIs.
- **Genuine implementation:** Every stage implements real training loops, parameter fitting, genuine inference, and rigorous test coverage.
- **Progressive milestones:** Advanced concepts build upon working foundational code.

---

## 2. Environment & Prerequisites

- **OS:** Windows 10/11
- **Python Version:** 3.12.3
- **Virtual Environment:** `.venv` (located in root directory)
- **Verified Libraries:**
  - `torch` (2.14.1+cpu)
  - `numpy` (2.5.4)
  - `pandas` (3.1.0)
  - `scikit-learn` (1.9.1)
  - `joblib` (1.6.0)
  - `scipy` (1.18.1)
  - `matplotlib` (3.11.2)

---

## 3. Quick Start (Windows PowerShell)

### Activate the Virtual Environment
```powershell
# Navigate to project directory
Set-Location "C:\Users\A\my-personal-ai"

# Activate virtual environment
.\.venv\Scripts\Activate.ps1
```

### Run Stage Two Training
```powershell
python train_ai.py
```

### Run Automated Tests
```powershell
# Run Stage 2 model tests
python -m unittest test_model.py -v

# Run Stage 3 machine learning fundamentals tests
python -m unittest lessons.stage3.test_stage3_lessons -v

# Run Stage 4 neural network tests
python -m unittest lessons.stage4.test_stage4 -v

# Run Stage 5 language models and transformer tests
python -m unittest lessons.stage5.test_stage5 -v

# Run Stage 6 tiny LLM from scratch tests
python -m unittest lessons.stage6.test_stage6 -v

# Run entire project test suite (all 33 tests)
python -m unittest discover -s . -p "test_*.py" -v
```

### Train and Generate with Tiny LLM (Stage Six)
```powershell
# Train the decoder-only Transformer on CPU (takes ~9 seconds):
python train_tiny_llm.py --epochs 120 --batch-size 16 --context-length 48 --lr 0.003

# Generate text from a prompt:
python generate_tiny_llm.py --prompt "Shall I compare thee " --max-tokens 80 --temp 0.8 --top-k 5
```

---

## 4. Completed Milestones

### Stage Two — Classical Supervised Learning
- Implemented in `train_ai.py` with test suite `test_model.py`.
- Trained linear regression model predicting delivery hours from complexity points and experience.
- Verified train/test split, MAE/RMSE/R^2 metrics, input validation, and joblib serialization.

### Stage Three — Machine Learning Foundations
Implemented in `lessons/stage3/`:
- **Lesson 1:** Supervised (Classification with Logistic Regression) vs. Unsupervised (K-Means Clustering).
- **Lesson 2:** Underfitting (Degree 1, High Bias) vs. Overfitting (Degree 15, High Variance) vs. Balanced Generalization (Degree 4). Generated visual plot `underfitting_overfitting.png`.
- **Lesson 3:** Loss functions: MSE, MAE, Binary Cross-Entropy, and Categorical Cross-Entropy (the backbone of next-token prediction in LLMs).
- **Lesson 4:** Gradient Descent optimization: mathematical 1D convergence, learning rate tuning (slow vs. optimal vs. exploding/overshooting), and training Linear Regression from scratch with MSE gradients. Generated trajectory plot `gradient_descent_convergence.png`.
- **Lesson 5:** Regression vs. Classification: why metrics (MAE/R^2 vs. Accuracy/Precision/Recall/F1) cannot be interchanged.
- **Lesson 6:** Data quality and generalization: outlier vulnerability, data leakage in preprocessing, distribution shift, 3-way train/validation/test splits, and synthetic data limitations.
- **Automated Tests:** `lessons/stage3/test_stage3_lessons.py` (6 tests passing).

### Stage Four — Building First Neural Network (PyTorch)
Implemented in `lessons/stage4/`:
- **Lesson 1 (`lesson1_tensors.py`):** Multi-dimensional tensors (0D scalar to 3D batch sequence), slicing, views/reshapes, squeezes, element-wise math, and matrix multiplication (`@`).
- **Lesson 2 (`lesson2_autograd.py`):** Dynamic computational graphs, automatic differentiation (`backward()`), gradient accumulation, zeroing, and inference optimization with `torch.no_grad()`.
- **Lesson 3 (`lesson3_neural_network_basics.py`):** Layers, weights, biases, and activation functions (ReLU, Sigmoid, Tanh, GELU). Mathematical proof that stacking linear layers without activation collapses into a single linear layer.
- **Lesson 4 (`lesson4_forward_pass.py`):** Implementing `torch.nn.Module` and `nn.Linear` with shape tracking across layers and parallel batch inference.
- **Lesson 5 (`lesson5_backpropagation.py`):** The 5-step PyTorch training cycle: zero gradients -> forward pass -> loss evaluation -> `loss.backward()` -> `optimizer.step()`.
- **Lesson 6 (`lesson6_train_neural_network.py`):** Training a 3-layer MLP (`CodeQualityClassifier`) on CPU with Adam optimizer, CrossEntropyLoss, and zero data leakage.
- **Lesson 7 (`lesson7_evaluate_and_save.py`):** Held-out test evaluation (95.8% accuracy, 0.973 F1), saving model `state_dict` to `models/code_quality_nn.pt`, reloading into a clean instance, bit-exact verification, and single-component inference with input bounds validation.
- **Automated Tests:** `lessons/stage4/test_stage4.py` (6 tests passing).

### Stage Five — Understand Language Models and Transformers
Implemented in `lessons/stage5/`:
- **Lesson 1 (`lesson1_text_preprocessing.py`):** Unicode normalization (NFKD), whitespace and control character cleaning, punctuation spacing.
- **Lesson 2 (`lesson2_tokenization.py`):** Character, word, and subword tokenization with a working educational Byte-Pair Encoding (BPE) tokenizer.
- **Lesson 3 (`lesson3_vocabulary_special_tokens.py`):** Bidirectional vocabulary encoding/decoding with `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>` and batch padding.
- **Lesson 4 (`lesson4_token_embeddings.py`):** Dense lookup embeddings via `torch.nn.Embedding(vocab_size, d_model)`.
- **Lesson 5 (`lesson5_positional_encoding.py`):** Sinusoidal positional encodings solving permutation invariance.
- **Lesson 6 (`lesson6_scaled_dot_product_attention.py`):** Scaled dot-product attention $\text{softmax}(QK^T / \sqrt{d_k})V$ with shape tracking and probability distribution verification.
- **Lesson 7 (`lesson7_multi_head_attention.py`):** Multi-Head Attention, residual connections, LayerNorm, and position-wise Feed-Forward Networks.
- **Lesson 8 (`lesson8_causal_masking.py`):** Lower-triangular causal masking ensuring future attention weights are strictly 0.0.
- **Lesson 9 (`lesson9_next_token_prediction.py`):** Shifted input-target sequence alignment ($x_{0..T-2} \rightarrow y_{1..T-1}$) and cross-entropy loss over vocab logits.
- **Lesson 10 (`lesson10_training_validation.py`):** Decoder-only `TinyLanguageModel` training and validation loop on an educational corpus.
- **Lesson 11 (`lesson11_autoregressive_generation.py`):** Token-by-token generation loop terminating on `<EOS>` or `max_new_tokens`.
- **Lesson 12 (`lesson12_sampling_methods.py`):** Temperature scaling ($T$), Top-K filtering, and Top-P (nucleus) sampling.
- **Lesson 13 (`lesson13_pretraining_vs_finetuning.py`):** Base model pretraining mechanics vs. Supervised Fine-Tuning (SFT) on prompt-response pairs.
- **Lesson 14 (`lesson14_training_vs_inference.py`):** `model.train()` vs `model.eval()`, `torch.no_grad()`, determinism, and KV-caching concept.
- **Automated Tests:** `lessons/stage5/test_stage5.py` (10 tests passing).

### Stage Six — Train Tiny Language Model from Scratch
Implemented in `lessons/stage6/`, `train_tiny_llm.py`, and `generate_tiny_llm.py`:
- **Corpus (`data/tiny_corpus.txt`):** Curated public-domain text from Shakespeare's Sonnets (CC0 / Unrestricted public domain, 2,656 chars, 64 unique tokens).
- **Architecture (`lessons/stage6/model.py`):** Configurable `TinyTransformerLM` with token and learnable positional embeddings, Pre-LN residual Transformer blocks, multi-head causal self-attention, and projection head. 78,336 parameters on CPU.
- **Data Integrity (`lessons/stage6/dataset.py`):** Clean 85/15 train/val text partitioning before sequence sampling to eliminate boundary leakage.
- **Training Pipeline (`train_tiny_llm.py`):** Next-token cross-entropy training with AdamW, gradient clipping, periodic validation under `torch.no_grad()`, checkpoint persistence to `models/tiny_llm.pt`, and metric logging to `training_log.json`. Training loss dropped from 4.34 to 2.15; validation loss reached 2.46 in ~8.8 seconds.
- **Inference (`generate_tiny_llm.py`):** Autoregressive text generation with prompt handling, temperature scaling, top-k sampling, and graceful unknown character fallback.
- **Automated Tests:** `lessons/stage6/test_stage6.py` (7 tests passing).
- **Project Test Suite:** 33 tests passing across all completed stages.

---

## 5. Completed & Upcoming Milestones

| Stage | Description | Status |
|---|---|---|
| **Stage 1** | Inspect environment, packages, and establish architecture | **Complete** |
| **Stage 2** | Classical ML Model (`train_ai.py` & `test_model.py`) | **Complete** |
| **Stage 3** | Deep dive into ML Foundations (`lessons/stage3/`) | **Complete** |
| **Stage 4** | PyTorch Neural Network (`lessons/stage4/`) | **Complete** |
| **Stage 5** | Transformer & LLM Concepts Deep Dive (`lessons/stage5/`) | **Complete** |
| **Stage 6** | Tiny Language Model from Scratch (`train_tiny_llm.py`) | **Complete** |
| **Stage 7** | Local LLM Inference (Ollama Integration) | *Next Up* |
| **Stage 8** | FastAPI Backend with SQLite & Session Store | Planned |
| **Stage 9** | Modern React + TypeScript + Tailwind Chat UI | Planned |
| **Stage 10** | Persistent Memory Engine with User Approval | Planned |
| **Stage 11** | Local Retrieval-Augmented Generation (RAG) | Planned |
| **Stage 12** | Security Hardening & Full Documentation | Planned |
