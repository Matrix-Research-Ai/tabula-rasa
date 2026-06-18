# Tabula Rasa — Experimental Results

## Multi-Seed Accuracy (1-digit, 15K steps, curriculum)

| Operation | Seeds | Mean | Std | Best | Worst |
|-----------|-------|------|-----|------|-------|
| Addition | 5 | 100.0% | 0.00% | 100.0% | 100.0% |
| Subtraction | 5 | 100.0% | 0.00% | 100.0% | 100.0% |
| Multiplication | 5 | 31.0% | 0.00% | 31.0% | 31.0% |

## Length Generalization (train 2-digit, eval 3-digit, 5K steps)

| PE | Train | OOD (3d) | Seeds | Notes |
|----|-------|----------|-------|-------|
| ROPE | 100.0% | 0.0% | 3/3 | Classic OOD cliff |
| Abacus | 100.0% | 0.0% | 3/3 | No improvement |
| None | 100.0% | 0.0% | 3/3 | Pos signal not needed for 2d |
| Alibi | ~0% | 0.0% | 2/3 | Failed to converge |

## Key Findings

- 1M-parameter transformer, trained from scratch, reaches **100% on 1-digit addition and subtraction** in 15K steps (7.5 min on RTX A4500)
- **1-digit multiplication reaches 31%** — operation-specific scratchpad format works but needs further optimization
- **All PE variants show zero generalization** from 2→3 digit at 5K steps — length generalization remains an open challenge
- **Curriculum (1-digit → multi-digit) boosts convergence**: 1-digit starts at 100% within 1K steps, but multi-digit requires phase 2+ training
