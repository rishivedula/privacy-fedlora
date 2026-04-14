# Experiment 1: FedLoRA-KD Cross-Domain Transfer and Privacy Analysis

## Overview

This experiment evaluates the FedLoRA-KD framework across multiple domains, measuring both task performance and privacy leakage via membership inference attacks (MIA).

**Setup:**
- Base Model (BM): Llama-3.2-3B
- Universal Model (UM): Aggregated LoRA adapters from C1 (SQuAD) + C2 (NaturalQuestions)
- C3 Clients: SciQ, SAMSum, BillSum (held-out domains)

**Model Variants:**
| Abbreviation | Description |
|--------------|-------------|
| BM | Base model (frozen, no fine-tuning) |
| UM | Universal Teacher (BM + aggregated C1,C2 adapters) |
| BM + (C3) | Base model fine-tuned directly on C3 data |
| BM + (C3 w UM KD) | Base model trained on C3 with KD from UM |
| BM + AVG(C1,C2,C3) | Base model with averaged adapters from all clients |

---

## Privacy Leakage Analysis

### Membership Inference Attack (MIA) on C3 = SciQ

| Model | Accuracy | AUC | Precision | Recall |
|-------|----------|-----|-----------|--------|
| BM | 0.4750 | 0.4862 | 0.4490 | 0.2200 |
| UM | 0.4750 | 0.4869 | 0.4619 | 0.3033 |
| BM + AVG(C1,C2,C3) | 0.4783 | 0.4871 | 0.4667 | 0.3033 |
| BM + (C3 w KD) | 0.4717 | 0.4871 | 0.4564 | 0.2967 |
| **BM + (C3)** | **0.5267** | **0.5232** | **0.5196** | **0.7067** |

**Note:** AUC closer to 0.5 = less leakage (better privacy); AUC closer to 1.0 = more leakage (worse privacy)

### Key Finding

Direct fine-tuning on client data (BM + C3) exhibits **measurable membership inference vulnerability** (AUC = 0.523), while all other approaches maintain near-random-guess privacy (AUC ~ 0.487). The KD-based approach successfully preserves privacy.

---

## Task Performance Results

### C3 = SAMSum (Dialogue Summarization)

| Model | F1 | EM | Contains | PPL | BERT-F1 | BLEU | ROUGE |
|-------|----|----|----------|-----|---------|------|-------|
| BM | 0.5113 | 0.3660 | 0.8100 | 8.65 | 0.8957 | 0.4615 | 0.5078 |
| UM | 0.5260 | 0.3520 | 0.8000 | 7.79 | 0.8968 | 0.4683 | 0.5233 |
| **BM + (C3)** | **0.8441** | **0.7480** | **0.8100** | **4.04** | **0.9733** | **0.8172** | **0.8435** |
| BM + (C3 w UM KD) | 0.5132 | 0.3460 | 0.8000 | 7.48 | 0.8959 | 0.4573 | 0.5110 |
| BM + AVG(C1,C2,C3) | 0.7042 | 0.5760 | 0.8260 | 6.17 | 0.9320 | 0.6651 | 0.7025 |

### C3 = BillSum (Legal Summarization)

| Model | ROUGE-L | BLEU | BERT-F1 | PPL | F1 | EM |
|-------|---------|------|---------|-----|----|----|
| BM | 0.1730 | 0.0627 | 0.8549 | 3.77 | 0.2163 | 0.0000 |
| UM | 0.1671 | 0.0412 | 0.8608 | 3.64 | 0.2000 | 0.0000 |
| **BM + (C3)** | **0.2749** | **0.1034** | **0.8745** | **2.80** | **0.3139** | 0.0000 |
| BM + (C3 w UM KD) | 0.1682 | 0.0426 | 0.8613 | 3.65 | 0.2011 | 0.0000 |
| BM + AVG(C1,C2,C3) | 0.1950 | 0.0644 | 0.8624 | 3.32 | 0.2372 | 0.0000 |

---

## Summary of Findings

### 1. Privacy-Utility Tradeoff Confirmed

| Approach | Privacy (MIA AUC) | Utility | Verdict |
|----------|-------------------|---------|---------|
| BM + (C3) | 0.523 (worst) | Best | High utility, privacy leak |
| BM + (C3 w UM KD) | 0.487 (best) | ~Baseline | Privacy preserved, no utility gain |
| BM + AVG(C1,C2,C3) | 0.487 (best) | Medium | Good balance |

### 2. KD Not Transferring Sufficient Knowledge

The BM + (C3 w UM KD) results are **nearly identical to baseline BM/UM**, suggesting:
- The Universal Teacher may not provide enough signal for these target domains
- Current KD hyperparameters (alpha_base=0.25, alpha_universal=0.25) may be suboptimal
- Domain mismatch: C1/C2 (QA extractive) may not transfer well to summarization tasks

### 3. Aggregation Provides Middle Ground

BM + AVG(C1,C2,C3) shows reasonable privacy AND decent performance, supporting the claim that "incrementally incorporating new clients improves downstream performance."

---

## Recommendations for Next Experiments

1. **Diagnose KD gap**: Measure KL divergence between BM and UM outputs on C3 data
2. **Hyperparameter sweep**: Increase alpha_universal (try 0.4-0.5)
3. **Confidence-weighted dual-teacher KD**: Only distill from UM when confident
4. **DP regularization**: Find minimal noise to close privacy gap while preserving utility
5. **Task-aware aggregation**: Consider domain similarity when aggregating adapters

---

## Configuration

```json
{
    "model": "meta-llama/Llama-3.2-3B",
    "lora": {"r": 16, "lora_alpha": 32},
    "knowledge_distillation": {
        "temperature": 2.0,
        "alpha": 0.5,
        "alpha_base": 0.25,
        "alpha_universal": 0.25
    },
    "training": {
        "local_epochs": 3,
        "batch_size": 4,
        "learning_rate": 2e-4
    }
}
```
