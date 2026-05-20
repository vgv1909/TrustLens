# TrustLens: Hybrid LLM Hallucination Detection

A six-stage NLP pipeline that detects hallucinations in LLM outputs by combining Wikipedia evidence retrieval with self-consistency sampling, fused via calibrated logistic regression.

Built for DTSC 5525 — Advanced NLP and LLM Applications, University of North Texas.

---

## Results

| Metric | Held-Out (n=200) | 95% CI | Target | Status |
|---|---|---|---|---|
| AUROC | 0.811 | [0.755, 0.864] | > 0.75 | Met |
| F1 Score | 0.797 | [0.738, 0.850] | > 0.70 | Met |
| Spearman | 0.622 | [0.522, 0.712] | > 0.60 | Met |
| ECE | 0.142 | [0.093, 0.195] | < 0.10 | Partial |

DeLong paired AUROC test vs RoBERTa-MNLI baseline: Z = -11.273, p < 0.0001

**Key finding:** Wikipedia evidence retrieval contributes +0.182 AUROC — the single largest component. LLM self-judgment without external evidence (B5) achieves AUROC 0.348, worse than random guessing, confirming that LLMs cannot self-verify without an independent evidence source.

---

## Pipeline Architecture

```
Input QA Pair
    |
    v
Stage 1 — Response Generation       FLAN-T5-large (780M params, FP16, greedy decoding)
    |
    v
Stage 2 — Claim Decomposition       Punctuation-boundary segmentation, min 5 words
    |
    v
Stage 3 — Evidence Retrieval        FAISS IndexFlatIP + all-MiniLM-L6-v2 embeddings
                                    Top-5 Wikipedia paragraphs per claim
    |
    v
Stage 4 — NLI Verification          MiniCheck-RoBERTa (1.4 GB VRAM)
                                    NLI score = avg_supported - 1.5 x avg_unsupported
    |
    v
Stage 5 — Self-Consistency          k=5 samples at temperature 0.7, Jaccard overlap
    |
    v
Stage 6 — Hybrid Trust Score        Logistic regression fusion (trained on 100 HaluEval samples)
    |
    v
Output: Trust Score [0,1] + per-claim verdicts (Supported / Contradicted / Uncertain / Unverifiable)
```

---

## Repo Structure

```
trustlens/
├── trustlens_notebook.ipynb        Full pipeline + all evaluations (Sections A through H)
├── trustlens_app.py                Streamlit demo app (single file)
├── requirements.txt                Pinned dependencies for the notebook
├── app_requirements.txt            Lightweight dependencies for the demo app
├── README.md                       This file
├── .streamlit/
│   └── config.toml                 Forces light theme
└── jsonl/
    ├── trustlens_results.jsonl     Per-sample evaluation log
    ├── expanded_results.json       Held-out n=200 results + bootstrap CIs
    ├── delong_results.json         DeLong AUROC significance test
    ├── k5_results.json             k=2 vs k=5 ablation
    └── custom_test_results.json    50-question custom test set
```

---

## Quickstart

### Run the Streamlit demo (no GPU needed)

```bash
pip install -r app_requirements.txt
python3 -m streamlit run trustlens_app.py
```

Opens at http://localhost:8501. Runs in Demo mode by default using pre-computed results — no GPU required.

### Run the full pipeline (Google Colab, T4 GPU recommended)

1. Upload `trustlens_notebook.ipynb` to Google Colab
2. Runtime > Change runtime type > T4 GPU
3. Run Cell A1 to install dependencies
4. Restart runtime (required after numpy pin)
5. Run all remaining cells in order (Sections A through H)

### Run locally (requires CUDA GPU, 8+ GB VRAM)

```bash
pip install -r requirements.txt
jupyter notebook trustlens_notebook.ipynb
```

---

## Demo App Tabs

| Tab | Description |
|---|---|
| Analyze Response | Enter a question and get a Trust Score gauge and per-claim color-coded verdicts |
| Baseline Comparison | AUROC and F1 charts for B1 through B5, Hybrid, and LogReg with ablation table |
| FLAN-T5 vs Mistral-7B | Side-by-side generator comparison on the same question |
| About TrustLens | Architecture overview, results summary, model details |

---

## Hardware and Cost

| Component | Spec |
|---|---|
| Platform | Google Colab (T4 GPU, free tier) |
| GPU | NVIDIA Tesla T4, 15.6 GB VRAM, CUDA 12.8 |
| Peak VRAM usage | ~6.6 GB (all models loaded simultaneously) |
| End-to-end runtime | ~4-5 hours for n=300 samples |
| Total cost | $0 |

---

## Datasets

| Dataset | Source | License | Use |
|---|---|---|---|
| HaluEval QA | pminervini/HaluEval on HuggingFace | MIT | Primary, secondary, and held-out evaluation |
| TruthfulQA | truthful_qa on HuggingFace | Apache 2.0 | Scope boundary analysis |
| Wikipedia | wikipedia-api Python library | CC BY-SA | Live evidence retrieval |
| Custom test set | Hand-crafted (in notebook) | — | 50 QA pairs across 5 domains |

---

## Ablation Summary

| Component Added | AUROC Change | Interpretation |
|---|---|---|
| Wikipedia retrieval | +0.182 | Largest single contribution |
| MiniCheck over RoBERTa-MNLI | 0.000 AUROC, +0.175 F1 | Better precision-recall balance |
| Self-consistency at k=2 | -0.111 | Anti-correlated at low sampling depth |
| Self-consistency at k=5 | +0.091 | Complementary signal at sufficient depth |
| Learned fusion weights | +0.131 | Optimal weight combination |

---

## Reproducibility

Five mechanisms ensure deterministic results:
1. Fixed random seeds (numpy, random, HuggingFace set_seed, all set to 42)
2. Centralised CONFIG dictionary — no hardcoded hyperparameters elsewhere
3. JSONL checkpointing — resume after Colab session timeout
4. Dependency pinning — numpy==1.26.4 locked before other installs
5. Bootstrap CIs from 1,000 resamples with fixed numpy seed=42

---

## References

- HaluEval — Li et al. (2023) arXiv:2305.11747
- MiniCheck — Tang et al. (2024) arXiv:2404.10774
- SelfCheckGPT — Manakul et al. (2023) EMNLP
- FactScore — Min et al. (2023) arXiv:2305.14251
- FLAN-T5 — Chung et al. (2022) arXiv:2210.11416
- DeLong test — DeLong et al. (1988) Biometrics
