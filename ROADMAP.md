# Tabula Rasa — Project Roadmap

> **Vision:** An AI system that learns from a blank slate, one specialist at a time — transferable, inspectable, and scalable.

## Current Status

| Phase | Focus | Status |
|-------|-------|--------|
| **1** | Single-task arithmetic mastery | ✅ **COMPLETE** |
| **2** | Continual learning via Online EWC | ✅ **COMPLETE** |
| **3** | Reasoning — Socratic Engine, self-play | ❌ **FUTURE** |

---

## Phase 1: Arithmetic Specialist (COMPLETE)

A 1M-parameter transformer trained from scratch to perform arithmetic with 100% accuracy.

### Delivered
- [x] Causal transformer with RoPE, RMSNorm — 1M params (128-dim, 4-layer, 4-head, 512-FF)
- [x] Fused carry-digit tokenizer (20 tokens encoding carry+digit per column)
- [x] Addition specialist: 100% accuracy in 3,000 steps (1-digit, CPU)
- [x] Curriculum learning — 4 phases from 1-digit to 4-digit
- [x] Config-driven design — all knobs in `config.py`, editable via dashboard
- [x] Web dashboard with 25+ views (Training, Config, Evaluation, Checkpoints, Chat)
- [x] REST API for inference, training control, checkpoint management
- [x] Auto-train loop — evaluates all operations, trains the weakest, repeats

---

## Phase 2: Continual Learning (COMPLETE)

The system learns sequential tasks without catastrophic forgetting using **Online Elastic Weight Consolidation (EWC)**.

### Delivered
- [x] Online EWC module — Fisher matrix computation, exponential decay merge (γ=0.9)
- [x] 39 parameter groups tracked across 1M-param model
- [x] Hippocampus (SQLite) — stores high-surprise experiences for replay
- [x] Sleep cycle daemon — periodic replay → Fisher compute → merge → train → consolidate
- [x] CLI integration: `--ewc` flag on `train_specialist.py`
- [x] Sequential retention validated: addition 83% → subtraction + EWC → addition **92%**, subtraction **97%**
- [x] Ablation 1: No-EWC control — catastrophic forgetting without EWC (31pp drop)
- [x] Ablation 2: Lambda sensitivity sweep — flat Pareto frontier (all λ≥500 → 100/100)
- [x] Ablation 3: Three-task sequence — scalability boundary identified (3 tasks: collapse)

### Validated Findings
- [x] EWC prevents forgetting reliably: **100.0±0.0%** retention vs **64.0±55.4%** without (bimodal)
- [x] Training stability matters more than final accuracy — trajectories reveal hidden collapses
- [x] Lambda robustness: flat Pareto frontier across 20x range
- [x] Three-task limit is **architectural, not algorithmic** — both merged and per-task EWC fail identically
- [x] Adaptive gamma correctly rejected (Fisher overlap=0.977)

---

## Phase 3: Reasoning (FUTURE)

Phase 3 pushes beyond arithmetic into general reasoning, self-play, and language acquisition. These are active research targets.

### 3A. Socratic Engine — Dialectical Self-Play
- [x] **Stage 1 prototype** — `egefalos/socratic_stage1.py`: two agents, compare solutions
- [x] **Stage 2 prototype** — `egefalos/socratic_stage2.py`: critique + refine
- [x] **Stage 3 prototype** — `egefalos/socratic_stage3.py`: multi-turn dialectic
- [ ] Integrate Socratic stages into training loop — verify improvement over baseline
- [ ] Publish comparative results: Socratic-trained vs standard-trained specialist accuracy
- [ ] Benchmark on held-out digit ranges (generalization test)

### 3B. Language AlphaZero
- [x] **AlphaZero framework** — `egefalos/language_az.py`: MCTS + value network + self-play
- [x] **Micro-MCTS** — `egefalos/mcts.py`: 16-simulation search tree
- [x] **Config scaffolding** — `config.py` includes `use_language_az`, `language_vocab_size`, etc.
- [ ] Train a GPT-2-compatible tokenizer that handles general text
- [ ] Validate on a simple language task (e.g., reversed-string prediction, next-token on WikiText-2)
- [ ] Compare language acquisition vs supervised fine-tuning

### 3C. Code AlphaZero
- [x] **Code sandbox** — `egefalos/code_sandbox.py`: Python execution sandbox
- [x] **Code curriculum** — `egefalos/code_curriculum.py`: syntax → fuzzing → algorithm
- [ ] Train a code-specialist model through self-play
- [ ] Validate on simple algorithmic tasks (fizzbuzz, sorting)
- [ ] Benchmark against supervised code generation baselines

### 3D. Multi-Specialist Orchestration
- [ ] Formalize the "Hippocampus" memory architecture — working memory to long-term storage transition
- [ ] Route queries across specialists with confidence scoring
- [ ] Implement specialist merging (combine two specialists into one without retraining)
- [ ] Achieve structured continual learning across 5+ tasks without forgetting

### 3E. General Text Tokenizer
- [ ] Extend tokenizer to handle general Unicode text (beyond math)
- [ ] Options: BPE tokenizer (`bpe_tokenizer.py` exists as reference), or learned token embeddings
- [ ] Validate: train a language specialist on a small text corpus with the extended tokenizer
- [ ] Goal: drop-in replacement for a GPT-2 tokenizer on simple text tasks

---

## Infrastructure & Community

### Documentation
- [x] **README.md** — project overview, getting started, architecture, API
- [x] **RESULTS.md** — validated claims with experimental data
- [x] **ROADMAP.md** — this file
- [x] **LICENSE** — MIT license
- [x] **CODE_OF_CONDUCT.md** — contributor covenant
- [x] **CONTRIBUTING.md** — developer guide, style guide, PR process
- [x] **CONTRIBUTING_EXTENDING.md** — step-by-step tutorial for adding new operations
- [ ] Add quickstart Jupyter notebook (`quickstart.ipynb`) — train, evaluate, visualize

### Code Quality
- [ ] Refactor to src-layout (`src/tabula_rasa/` package structure)
- [ ] Add pre-commit hooks (formatting, linting)
- [ ] Add CI pipeline (GitHub Actions): test, lint, experiment validation
- [ ] Mixed precision training (AMP) support for GPU
- [ ] Gradient accumulation for smaller GPU training
- [ ] Weights & Biases integration as optional logger

### Community
- [ ] Publish paper on arXiv
- [ ] Create project logo and banner
- [ ] Add GitHub release with version tags
- [ ] Create GitHub Actions CI badge
- [ ] Post on r/LocalLLaMA, r/MachineLearning
- [ ] Share on Hacker News
- [ ] Share on AI Discord communities

### Experiments
- [ ] Profile inference speed (validate O(1) overhead claim)
- [ ] Test on held-out unseen digit counts (1-3 digit training → 4-5 digit evaluation)
- [ ] Validate against CL-Bench continual learning benchmarks
- [ ] Run error bar experiments with 5+ random seeds
- [ ] Per-task EWC vs Online EWC comparison (3+ tasks)

---

## How to Contribute

See [CONTRIBUTING.md](CONTRIBUTING.md) and [CONTRIBUTING_EXTENDING.md](CONTRIBUTING_EXTENDING.md).

Priority areas for contribution:

1. **Phase 3 items** — pick a `[ ]` checkbox and make it `[x]`
2. **Infrastructure** — src-layout refactor, CI pipeline, mixed precision
3. **Experiments** — run the planned experiments and update results
4. **Documentation** — quickstart notebook, tutorials

---

*Last updated: June 2026*
