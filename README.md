# Cost-Aware Calibration (CAC)

**When Does Ensemble Disagreement Track Human Uncertainty? A Cost-Aware Study Across Model Scale, Sample Size, and Modality.**

CAC tests a simple intuition: when a small ensemble of models disagrees on an input, is that input genuinely hard for humans too — and can that disagreement cheaply route annotation, escalating only contested items to an expensive model or human? We evaluate this against **human label distributions** (CIFAR-10H for vision, ChaosNLI for language) and map where it holds and where it breaks.

A heterogeneous small-model ensemble produces two disagreement signals — Jensen–Shannon divergence over class posteriors (JSD) and rationale dissimilarity (1 − MTA) — which are fused, calibrated to an escalation probability, and used to drive a three-stage **skip / consensus / frontier** cascade.

## Key results

| Domain | Ensemble (N) | r_jsd | r_dual | Δr | Single | Cascade | Frontier % |
|---|---|---|---|---|---|---|---|
| Vision (CIFAR-10H) | 2B (10,000) | 0.311 | 0.319 | +0.008 | 0.940 | **0.965** | 22.8% |
| Vision (CIFAR-10H) | 7–8B (100) | 0.292 | 0.309 | +0.018 | 0.967 | 0.922 | 10.0% |
| Language (ChaosNLI) | 2×small (3,113) | 0.046 | 0.047 | +0.001 | 0.564 | 0.589 | 1.0% |

- **Routing utility can decouple from correlation:** on CIFAR-10H the cascade beats the best single model (0.965 vs 0.940) at 4.4× lower frontier cost, despite only a modest correlation (r_dual = 0.319, 95% CI [0.299, 0.339]).
- **The signal has boundaries:** it degrades with base-model scale, decays with evaluation size (r_jsd 0.41 → 0.31 from N=100 to 10,000), and does not transfer to language (ChaosNLI r_jsd = 0.046, 95% CI [0.011, 0.079]; cascade collapses).
- The textual lift is **not** statistically significant at scale (Δr = +0.008, 95% CI [−0.003, +0.017]).

## Environment

Single NVIDIA A100-80GB. Inference via vLLM 0.6.3 with `lm-format-enforcer` constrained decoding.

```bash
conda activate <env>           # e.g. /home/$USER/.conda/envs/cac
python -c "import sentence_transformers as st; print(st.__version__)"   # must be 2.7.0
```

**Pinned dependencies (critical):** `sentence-transformers==2.7.0` (newer versions change the CrossEncoder API and break MTA), `torch==2.4.0+cu121`, `vllm==0.6.3.post1`, `transformers==4.46.3`, `numpy==1.26.4`. See `requirements.txt`.

## Models

Served offline from a local weight cache (`.hf_cache/`). Gated models (Gemma-2, Llama) require `huggingface-cli login` plus license acceptance on the model page.

- **Vision 2B:** Qwen2-VL-2B-Instruct, InternVL2-2B
- **Vision 7–8B:** Qwen2-VL-7B (AWQ), InternVL2-8B (fp16), Phi-3.5-Vision
- **Language:** Qwen2.5-3B-Instruct, Gemma-2-2B-it (small, cross-family — chosen because stronger models suppress useful disagreement)

## Reproduction

### Data

```bash
# CIFAR-10H: 10,000 CIFAR-10 test images + human soft labels
python -m run.prepare_data --download-models

# ChaosNLI: SNLI+MNLI subset (3,113 examples), αNLI excluded
wget -O data/chaosnli/chaosNLI_v1.0.zip "https://dl.dropboxusercontent.com/s/h4j7dqszmpt2679/chaosNLI_v1.0.zip?dl=1"
cd data/chaosnli && unzip chaosNLI_v1.0.zip && cd -
```

> Note: ChaosNLI stores label entropy in **bits (base-2)**; `cac/data/chaosnli.py` cross-checks parsed `label_dist` against the file's `entropy` field on load. The `label_dist` index order is `[entailment, neutral, contradiction]`.

### Vision track (CIFAR-10H)

```bash
# 1. ensemble inference -> outputs/raw/<model>.jsonl
python -m run.run_vision_inference --n 10000

# 2. rationale textual agreement (cross-encoder) -> outputs/cifar10h_mta.npy
python -m run.compute_mta

# 3. correlation report + full calibration/routing pipeline
python -m run.report --dataset cifar10h
python -m run.run_pipeline --real --dataset cifar10h
```

### Language track (ChaosNLI)

```bash
# 1. ensemble inference (Qwen2.5-3B + Gemma-2-2B)
python -m run.run_text_inference --n 3113

# 2-3. MTA + report + pipeline
python -m run.compute_mta
python -m run.report --dataset chaosnli
python -m run.run_pipeline --real --dataset chaosnli
```

> **⚠️ Known footgun — running both tracks in one workspace.** Two paths are
> currently hardcoded to CIFAR names and are *not* dataset-aware, so switching
> between the vision and language tracks needs manual care:
>
> 1. **`outputs/raw/` must hold only one dataset at a time.** `load_distributions()`
>    reads *every* `*.jsonl` in `outputs/raw/`; the 10-class CIFAR files and 3-class
>    NLI files cannot be stacked together and will raise (or, if N coincides,
>    silently mix shapes). Keep the inactive dataset's files elsewhere (e.g. the
>    CIFAR files in `outputs/raw_cifar10h/` while the NLI track runs).
> 2. **`compute_mta` always writes `outputs/cifar10h_mta.npy`** regardless of
>    `--dataset`. Running it on the NLI track overwrites the CIFAR MTA array. Copy
>    the file aside (e.g. `cifar10h_mta_CIFAR.npy`) before switching tracks, or
>    you will have to regenerate it.
>
> Both are workarounds, not fixes — a future cleanup should make the MTA/embedding
> artifact paths dataset-aware (e.g. `chaosnli_mta.npy` vs `cifar10h_mta.npy`).

### Analyses (no GPU)

Bootstrap CIs, the frontier-budget sweep, and the failure taxonomy are pure post-processing over the saved `outputs/raw/` distributions and MTA arrays. See `analysis/` (or the snippets in the paper appendix).

### Frontier baseline (optional, paid API)

```bash
python -m run.run_frontier --subset outputs/chaosnli_frontier_subset.json --model <model>
```
The stratified 450-item subset (entropy terciles) is frozen in `outputs/chaosnli_frontier_subset.json`.

## Repository layout

```
cac/
  config.py              paths, device(), load_yaml()
  data/
    cifar10h.py          CIFAR-10H loader (prepare, download_probs)
    chaosnli.py          ChaosNLI loader (load, human_distributions, sentence_embeddings)
    labels.py            10-class label space (vision)
    labels_nli.py        3-class label space (NLI; A/B/C, K=3 logprobs_to_dist)
    target_source.py     dataset-agnostic human_probs()/embeddings() shim
  models/
    vllm_runner.py       VLMRunner (vision)
    text_runner.py       TextRunner (NLI; family-aware prompts, no system role for Gemma)
    schema.py            10-class guided JSON schema, rationale_to_text()
    schema_nli.py        3-class guided JSON schema
  ensemble/
    inference.py         load_distributions(), load_rationales()
    jsd.py               mean_pairwise_jsd()
  mta/
    cross_encoder.py     MTAScorer (rationale sanitizer + manual tokenization)
  pipeline/
    metrics.py           correlations, cost_stages, cascade_accuracy, normalise_01
    calibration.py       isotonic / Platt fit + ECE
    prerouter.py         skip-stage classifier on item embeddings
    weights.py           MI / logistic alpha-beta fitting
    figures.py, results.py
  targets.py             human_entropy, hard_mask, BUDGET
config/
  models_vision.yaml, models_text.yaml
run/
  run_vision_inference.py, run_text_inference.py
  compute_mta.py, report.py, run_pipeline.py
outputs/                 raw model outputs, MTA arrays, figures (gitignored)
```

## Citation

```bibtex
@misc{kalemullah2026cac,
  title  = {When Does Ensemble Disagreement Track Human Uncertainty?
            A Cost-Aware Study Across Model Scale, Sample Size, and Modality},
  author = {Kalemullah, Amaan Mohamed and H., Yixuan},
  year   = {2026},
  note   = {Arizona State University}
}
```

ChaosNLI: Nie, Zhou, Bansal, *What Can We Learn from Collective Human Opinions on NLI Data?*, EMNLP 2020.
CIFAR-10H: Peterson et al., *Human Uncertainty Makes Classification More Robust*, ICCV 2019.
