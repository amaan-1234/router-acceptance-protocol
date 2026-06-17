# Before You Trust Disagreement-Based Routing: A Router Acceptance Protocol

**A Router Acceptance Protocol and Its Validation Across Scale, Modality, and Domain**

This repository accompanies the paper submitted to AAAI 2026 (double-blind review). We turn the implicit "check before you trust it" advice around disagreement-based routing into a concrete, reusable **router acceptance protocol** with four pre-deployment checks and pass/fail rules — then validate each check against human label distributions across model scale, two modalities, two task types, and a frontier control.

---

## The Protocol (A1–A4)

| Check | Failure mode it catches | Pass criterion |
|-------|------------------------|----------------|
| **A1** Utility over correlation | Trusting a correlation that yields no routing benefit | Calibrated cascade matches or beats single-model baseline at target coverage (AURC / selective accuracy) |
| **A2** Capacity sweep | Attributing behavior to "small models" from a two-point contrast | Characterized trend across ≥4 capacity points; outliers flagged |
| **A3** Error independence | Assuming ensemble members err independently | Mean pairwise φ is low; fail if errors are strongly correlated |
| **A4** Measured frontier | Cost projections built on an assumed accuracy ceiling | Projections use the *measured* frontier accuracy for the target domain |

---

## Key Results

### A1 — Routing utility decouples from correlation (CIFAR-10H)

| System | Acc | Frontier % | Cost reduction |
|--------|-----|------------|----------------|
| Best single model (2B) | 0.940 | 100% | 1× |
| Calibrated cascade (2B) | **0.965** | 22.8% | **4.4×** |

Disagreement–difficulty correlation: r = 0.32. Utility and correlation are empirically distinct.

### A2 — Capacity is a curve; AURC is the cleaner axis

**ChaosNLI within-family sweep (450-item stratified subset):**

| Model | r | AURC ↓ | sel@50 | sel@70 |
|-------|---|--------|--------|--------|
| Qwen3.5-0.8B | 0.138 | 0.465 | 0.529 | 0.486 |
| Qwen3.5-2B | 0.132 | 0.424 | 0.578 | 0.606 |
| Qwen3.5-4B ★ | 0.377 | 0.230 | 0.787 | 0.686 |
| Qwen3.5-9B | 0.304 | 0.306 | 0.618 | 0.584 |
| Qwen3.5-27B | 0.271 | 0.438 | 0.520 | 0.476 |
| Qwen3.5-122B (MoE) † | 0.268 | 0.387 | 0.596 | 0.495 |
| Gemma4-E2B ‡ | 0.150 | 0.375 | 0.644 | 0.600 |
| Gemma4-31B ‡ | 0.383 | 0.158 | 0.822 | 0.813 |
| Qwen3.6-27B ‡ | 0.352 | **0.146** | 0.853 | 0.810 |
| Frontier (Sonnet 4.6) | **0.471** | 0.178 | — | — |

★ Positive outlier; flagged per A2. † MoE — reported separately, not on the dense trend. ‡ Cross-family comparison; not placed on the Qwen3.5 capacity curve.

**CIFAR-10H within-family sweep (450-item stratified subset, appendix):**

| Model | r | AURC ↓ | AUROC | sel@70 |
|-------|---|--------|-------|--------|
| Qwen3.5-0.8B | 0.137 | 0.331 | 0.580 | 0.692 |
| Qwen3.5-2B | 0.376 | 0.137 | 0.682 | 0.876 |
| Qwen3.5-4B | 0.447 | 0.137 | 0.781 | 0.829 |
| Qwen3.5-9B | **0.529** | **0.098** | **0.804** | 0.857 |
| Qwen3.5-27B ★ | 0.547 | 0.206 | 0.781 | 0.651 |
| Gemma4-31B ‡ | 0.545 | 0.174 | 0.776 | 0.717 |
| Qwen3.6-27B ‡ | 0.486 | 0.152 | 0.769 | 0.771 |

★ Computed over 224/450 parseable items. ‡ Cross-family comparison.

### A3 — Small dense models err *together* (ChaosNLI)

Pairwise error correlation φ among Qwen3.5 small dense models: **0.78–0.83**. Error independence appears only at the 122B MoE model (φ ∈ [−0.05, 0.14]). The finding does not replicate in the vision setting (φ ∈ [0.08, 0.48]), confirming A3 conclusions are domain-specific.

### A4 — Measured frontier vs. assumed ceiling (ChaosNLI)

| System | Acc | corr(unc., H) | 95% CI |
|--------|-----|---------------|--------|
| Small models (Qwen3.5 ≤ 2B) | ~0.49 | ~0.13 | — |
| Frontier (Sonnet 4.6) | **0.673** | **0.515** | [0.446, 0.578] |

Commonly assumed ceiling: 0.95. Measured: 0.673 — any cost projection using the assumed ceiling is invalid per A4.

---

## Environment

```bash
conda activate <env>
python -c "import sentence_transformers as st; print(st.__version__)"  # must be 2.7.0
```

**Pinned dependencies:** `sentence-transformers==2.7.0`, `torch==2.4.0+cu121`, `vllm==0.6.3.post1`, `transformers==4.46.3`, `numpy==1.26.4`. See `requirements-gpu.txt` (full GPU stack), `requirements-light.txt` (analysis only).

**Compute:** One node, two NVIDIA A100 GPUs (driver 595.71.05, CUDA 13.2), 20× Intel Xeon Platinum 8468, 234 GB RAM.

---

## Models

**Capacity sweep (within-family):** Qwen3.5 — 0.8B, 2B, 4B, 9B, 27B (dense), 122B (MoE, ~10B active). Endpoints 0.8B / 27B / 122B served via hosted API; 2B / 4B / 9B served locally via vLLM.

**Cross-family comparison:** Gemma4-E2B, Gemma4-31B, Qwen3.6-27B (hosted API).

**Cascade ensembles:**
- Vision 2B: Qwen2-VL-2B-Instruct + InternVL2-2B
- Language: Qwen2.5-3B-Instruct + Gemma-2-2B-it

**Frontier control:** Claude Sonnet 4.6 (ChaosNLI, 450-item stratified subset).

---

## Reproduction

### Data

```bash
# CIFAR-10H
python -m run.prepare_data --download-models

# ChaosNLI
wget -O data/chaosnli/chaosNLI_v1.0.zip \
  "https://dl.dropboxusercontent.com/s/h4j7dqszmpt2679/chaosNLI_v1.0.zip?dl=1"
cd data/chaosnli && unzip chaosNLI_v1.0.zip && cd -
```

### Capacity sweep (API-based, any OpenAI-compatible endpoint)

```bash
export LLM_API_KEY=...
BASE=https://<your-endpoint>/v1

# ChaosNLI — one call per model
python run/run_ensemble_api.py --model qwen35-0p8b   --base-url $BASE --key-env LLM_API_KEY --out-tag sub450
python run/run_ensemble_api.py --model qwen35-27b-fp8 --base-url $BASE --key-env LLM_API_KEY --out-tag sub450
# ... repeat for each model in the sweep

# Vision — CIFAR-10H
python run/run_vision_api.py --model qwen35-2b --base-url $BASE --key-env LLM_API_KEY --out-tag sub450

# Evaluate
python run/eval_sweep.py --prefix sweep_chaosnli_
python run/eval_sweep.py --prefix vision_cifar_
```

Results append to `outputs/sweep_chaosnli_<model>.jsonl` and `outputs/vision_cifar_<model>.jsonl`; runs are resumable (completed items are skipped).

### Vision cascade (A1, local vLLM)

```bash
python -m run.run_vision_inference --n 10000
python -m run.compute_mta
python -m run.report --dataset cifar10h
python -m run.run_pipeline --real --dataset cifar10h
```

### Language cascade (A1, local vLLM)

```bash
python -m run.run_text_inference --n 3113
python -m run.compute_mta
python -m run.report --dataset chaosnli
python -m run.run_pipeline --real --dataset chaosnli
```

### Frontier control (A4)

```bash
# Requires Anthropic API key
python run/run_frontier.py \
  --subset outputs/chaosnli_frontier_subset.json \
  --model claude-sonnet-4-6
```

The stratified 450-item subset (entropy terciles) is frozen in `outputs/chaosnli_frontier_subset.json`.

### Open-ended grading (second domain)

```bash
python run/run_ellipse_graders.py   # ELLIPSE short-answer + code grading
python run/eval_ellipse.py
```

---

## Repository Layout

```
cac/
  config.py                paths, device(), load_yaml()
  data/
    cifar10h.py            CIFAR-10H loader
    chaosnli.py            ChaosNLI loader (label_dist index: [entailment, neutral, contradiction])
    ellipse.py             ELLIPSE grading loader
    labels.py              10-class vision label space
    labels_nli.py          3-class NLI label space
    target_source.py       dataset-agnostic human_probs()/embeddings() shim
  models/
    vllm_runner.py         VLMRunner (vision, local)
    text_runner.py         TextRunner (NLI, local)
    schema.py              10-class guided JSON schema
    schema_nli.py          3-class guided JSON schema
  ensemble/
    inference.py           load_distributions(), load_rationales()
    jsd.py                 mean_pairwise_jsd()
  mta/
    cross_encoder.py       MTAScorer
  pipeline/
    metrics.py             correlations, AURC, cascade_accuracy
    calibration.py         isotonic / Platt + ECE
    prerouter.py           skip-stage classifier
    weights.py             MI / logistic alpha-beta fitting
    figures.py, results.py
  targets.py               human_entropy, hard_mask, BUDGET
run/
  run_ensemble_api.py      capacity sweep — ChaosNLI (hosted API)
  run_vision_api.py        capacity sweep — CIFAR-10H (hosted API)
  run_frontier.py          frontier control (Anthropic API)
  run_ellipse_graders.py   open-ended grading track
  eval_sweep.py            sweep evaluation (r, AURC, AUROC, φ matrix)
  eval_ellipse.py          grading evaluation
  run_vision_inference.py  cascade inference — vision (local vLLM)
  run_text_inference.py    cascade inference — language (local vLLM)
  compute_mta.py           rationale textual agreement
  report.py, run_pipeline.py
paper_overleaf/
  extracted_v3/AnonymousSubmission/LaTeX/   working LaTeX copy
outputs/                   model outputs, figures (gitignored except paper assets)
logs/                      run logs
config/
  models_vision.yaml, models_text.yaml
```

> **Note — multi-track workspace:** `load_distributions()` reads every `*.jsonl` in `outputs/raw/`; keep CIFAR and NLI files separate (e.g. move inactive files to `outputs/raw_cifar10h/`) to avoid shape mismatches. `compute_mta` always writes `outputs/cifar10h_mta.npy` regardless of dataset — copy aside before switching tracks.

---

## Citation

```bibtex
@misc{anonymous2026rap,
  title  = {Before You Trust Disagreement-Based Routing:
            A Router Acceptance Protocol and Its Validation
            Across Scale, Modality, and Domain},
  author = {Anonymous},
  year   = {2026},
  note   = {Submitted to AAAI 2026 (double-blind review)}
}
```

**ChaosNLI:** Nie, Zhou, Bansal. *What Can We Learn from Collective Human Opinions on NLI Data?* EMNLP 2020.  
**CIFAR-10H:** Peterson et al. *Human Uncertainty Makes Classification More Robust.* ICCV 2019.
