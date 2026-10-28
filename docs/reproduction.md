# Reproducing BeyondAccuracy

Run commands from the repository root. The release has three entry levels: inspect released numbers, regenerate figures, or rerun model experiments. The first requires only Python; the second requires the plotting dependencies; the third requires model weights, datasets, activation caches, and suitable compute.

## 1. Released results and figures

```bash
python scripts/show_results.py
python paper/workshop/figures/make_rewrite_assets.py
python paper/workshop/figures/make_appendix_b.py
```

The figure scripts read aggregate CSVs in `results/` and JSON snapshots in `paper/workshop/numbers/`. They do not perform inference. The main script regenerates capture curves, probe curves, the intervention trade-off, the grid heatmap, and compact tables. The appendix script regenerates probe sweeps. Historical filenames use `rewrite` because they were produced during paper revision.

## 2. Install the research environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -m 'not slow'
```

Python 3.11+ is required. Install the appropriate PyTorch CUDA build for GPU work. The default Hugging Face path supports ordinary model inference; `E5_ENGINE=hf` is the documented baseline. A separate optional vLLM backend is included, but is not installed by default: it uses version-sensitive model internals and requires engine parity checks before a new run. See `scripts/e9_smoke_compare.py`, `validate_engine_parity.py`, and the vLLM-specific tests. Do not treat installation of an arbitrary vLLM version as a validated reproduction environment.

The selected source files are recorded in [the release manifest](release-manifest.json). Version constraints in `pyproject.toml` describe installation requirements, not a frozen recreation of the original GPU environment. Record `pip freeze`, model revisions, data revisions, and run variables with any new experiment.

## 3. Obtain aligned MATH inputs

```bash
python scripts/fetch_data.py
```

This downloads eleven aligned language JSON files from public [Language–Reasoning Disentangle](https://github.com/MuyuenLP/Language-Reasoning-Disentangle), pinned to commit `059b57d8895a7ce5b525431b0abbff620ebd33f1`. It writes to `data/parallel_multilingual_math/` and records the upstream revision. The revision is pinned for this release; it is not a claim that the original paper run recorded this exact upstream SHA. The upstream dataset terms apply; see [NOTICE](../NOTICE.md).

Alternatively, place the eleven `{lang}.json` files there yourself. Each must contain index-aligned records with `translated_prompt`, and the level/type fields used for stratification. Alignment by problem identity is essential: do not independently shuffle each language. MATH provides 7,500 training problems and the pipeline uses stratified sampling with seed zero.

MGSM is loaded by `src/benchmarks.py` through the Hugging Face datasets library (`juletxara/mgsm`). That loader needs network access on first use. Datasets and model weights retain their own terms.

No private cache access is required. `Config.data_repo` and `Config.acts_repo` default to empty strings. Optional `LRD_DATA_REPO` / `LRD_ACTS_REPO` overrides let you supply your own cache repositories. Without an activation-cache override, extraction runs locally.

## 4. Extract activations

```bash
python scripts/extract_activations.py --model Qwen/Qwen2.5-7B-Instruct --n 7500 --batch-size 4
LRD_MODEL=Qwen/Qwen2.5-7B-Instruct E3M_NSHOT=0 python scripts/e3_mgsm_acts.py
```

The first command extracts MATH final-prompt-token block outputs for every language and layer; the second extracts zero-shot MGSM activations. They save tensor caches locally under `data/activations/`. The MATH sidecar records selected indices, strata, seed, and model name. `LRD_MODEL_REVISION` can pin a model commit; use the same revision throughout extraction and generation. Set hardware batch sizes conservatively.

For a smaller exploratory run, reduce `--n` and set the same `LRD_N_SAMPLES` / `E5_N_SAMPLES` downstream. Smaller runs are smoke tests and do not reproduce the paper's reported numbers.

## 5. Geometry: paper Section 4 / E3

```bash
LRD_MODEL=Qwen/Qwen2.5-7B-Instruct LRD_N_SAMPLES=7500 E3R_MODE=math python scripts/e3_rank_sweep.py
LRD_MODEL=Qwen/Qwen2.5-7B-Instruct LRD_N_SAMPLES=7500 E3R_MODE=mgsm python scripts/e3_rank_sweep.py
python scripts/extract_trough.py --help
python scripts/extract_trough.py --model Qwen/Qwen2.5-7B-Instruct --n 7500
```

MATH mode fits on a seeded 50% problem split and evaluates held-out translation vectors. MGSM mode fits on all MATH inputs and evaluates cross-corpus. Comparisons include MEAN-SVD (`alg1`), DIFFPCA (`diffpca`), ACTIVATION-PCA (`pooled`), and RANDOM. DIFFPCA uses normalized within-problem residuals. The two mean-based estimators are capped at `L-1` = 10 language-varying dimensions.

Repeat extraction and analysis with `Qwen/Qwen3-8B`, `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`, and `THUDM/GLM-Z1-9B-0414` for the other paper models.

## 6. Probes after projection: paper Section 5 / E10

```bash
LRD_MODEL=Qwen/Qwen2.5-7B-Instruct LRD_N_SAMPLES=7500 python scripts/e10_probe_erasure.py
```

Requires both activation caches and the trough metadata from the previous step. The script uses grouped five-fold cross-validation: all translations of a problem stay together. Frozen probes train on original activations and test on projected activations; retrained probes train and test on projected activations. Layer, rank, and dose sweeps test different questions. Partial projection (`0 <= lambda < 1`) is invertible, so it cannot remove linear separability in principle. Local probe results do not establish how surviving information is used during generation.

## 7. Generation grid: paper Section 6 / E9

```bash
E5_MODEL=Qwen/Qwen2.5-7B-Instruct E5_N_SAMPLES=7500 python scripts/e9_dose_calibrate.py
E5_MODEL=Qwen/Qwen2.5-7B-Instruct E5_ENGINE=hf python scripts/e9_tradeoff_grid.py
E8_ITEMS_DIR=results/E9/items E8_OUT_DIR=results/E9 E8_LOW=bn,sw,te,th E8_REFS=clean,e9_alg1_proj_r10_d100 python scripts/e8_joint_metric.py
```

Calibration fits operators and coefficients from MATH activations, not MGSM outcomes. The grid reads those frozen edit specifications. The full grid has 37 configurations across subspace family, operator, rank, and dose. The fixed reference schedule suppresses layers 10–19 with `+0.3` and amplifies layers 20–27 with `-0.3`; off-center doses are energy-calibrated. `dose=1` denotes reference edit energy, not a projection coefficient of one.

Edits apply to prompt tokens during prefill and are removed before autoregressive decoding. MGSM generation uses eight native-language demonstrations and a 512-token output limit for the Qwen2.5 grid. Preserve the context window: some languages need at least 8,192 tokens. The scoring pipeline checks truncation differences rather than silently treating unequal truncation as an unconfounded gain.

This is a fixed-configuration grid, distinct from Zhao et al.'s per-language coefficient-tuning protocol. The paper's grid maxima are descriptive; choosing the best cell on the evaluation set is not independent held-out confirmation.

## 8. Prompt controls: paper Section 7 / E8

```bash
E5_MODEL=Qwen/Qwen2.5-7B-Instruct E5_ENGINE=hf PC_PIVOTS=en,zh PC_REQUIRE_NATIVE=1 python scripts/e8_prompt_control.py
```

Keep the target-language problem fixed while changing the demonstrations and/or answer cue to English or Chinese. `PC_N` limits problems for a smoke run; `PC_LANGS`, `PC_CONDITIONS`, `PC_NSHOT`, and `PC_SEED` configure the run. Preserve the model-specific decoding policies in `src/models.py`; reasoning-model settings and thinking/non-thinking conditions differ from the Qwen2.5 grid. Script outputs and released aggregates keep condition labels so these are not pooled accidentally.

## Metrics and uncertainty

- **Accuracy:** numeric answer correctness.
- **Fidelity:** output identified as the requested language. Strict scoring counts unidentified output as failure.
- **Joint:** both conditions hold for the same response.

Pool by averaging within each language and then weighting languages equally. Resample matched problem indices for paired bootstrap comparisons, carrying all translations/conditions together. The paper uses 4,000 replicates, pointwise 95% intervals, and family-wise correction where reported. Do not multiply aggregate accuracy and fidelity to estimate joint success.

## What is released

The repository includes aggregate results, numerical snapshots, source, and figure generators. It does not include raw generations, complete GPU environment snapshots, model weights, or activation caches. You can regenerate the included plots directly; end-to-end reruns require obtaining the inputs and executing inference. Exact numerical agreement across new library versions or hardware is not established by the offline tests.
