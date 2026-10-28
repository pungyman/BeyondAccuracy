# From paper to code

| Paper question | Implementation | Run it |
|:--|:--|:--|
| What translation variation is captured? | `subspace.py`, `candidates.py`, `controls.py`, `geometry.py` | `scripts/e3_rank_sweep.py` |
| What language information survives projection? | `probe.py`, `intervene.py`, `trough.py` | `scripts/e10_probe_erasure.py` |
| What improves during generation? | `operators.py`, `dose.py`, `e9_grid.py`, `evaluate.py` | `scripts/e9_dose_calibrate.py`, `scripts/e9_tradeoff_grid.py` |
| Can prompts produce the same trade-off? | `prompts.py`, `prompt_checkpoint.py`, `scoring.py` | `scripts/e8_prompt_control.py` |
| Do correctness and language agree on each response? | `scoring.py` and paired aggregation | `scripts/e8_joint_metric.py` |

All core modules live in [`src/`](../src). Historical names are retained to preserve compatibility with the research scripts. The package distribution is called `beyondaccuracy`; imports remain flat (`from candidates import diffpca_subspace`). Installing the project exposes those modules.

## Subspace names in saved results

| Saved identifier | Paper name |
|:--|:--|
| `alg1` | MEAN-SVD |
| `diffpca` | DIFFPCA |
| `pooled` | ACTIVATION-PCA |
| `random` | RANDOM |
| `proj` | Orthogonal projection |
| `leace` | Covariance-aware oblique removal |
| `wiener` | Signal-to-noise shrinkage |

`e9_grid.py` defines the allowed cells once for calibration and generation. `dose.py` calibrates edit energy by language. `intervene.py` provides Hugging Face hooks; `vllm_backend.py` provides the optional engine route and prompt masking. See the [module reference](README.md) and [reproduction guide](reproduction.md).
