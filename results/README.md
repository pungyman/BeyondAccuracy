# Released results

These are selected aggregate research outputs used by the accepted paper and its figure generators. Values are retained from the research workspace; they are not fresh inference runs performed during repository packaging.

| Directory | Evidence |
|:--|:--|
| `E3/<model>/` | MATH held-out and MGSM cross-corpus capture by depth/rank |
| `E10/<model>/` | Frozen/retrained language probes after projection |
| `E9/` | Pooled and per-language joint scoring for the Qwen2.5 intervention grid |
| `E8/` | Prompt-control summaries and strict joint-scoring aggregates |

Four-model geometry/probe files correspond to Qwen2.5-7B, Qwen3-8B, DeepSeek-R1-Distill-Qwen-7B, and GLM-Z1-9B. Historical identifiers are explained in [the code map](../docs/code-map.md).

`E9/joint_pooled_Qwen2.5-7B-Instruct.csv` is the source for the README's three-row comparison. `all11` denotes equal weighting across eleven languages, not pooling all responses with unequal weights. Scores in CSVs are fractions; README percentages multiply by 100.

Raw per-response generations, model weights, and activation tensors are not redistributed. Figure generation uses these aggregates and the numerical snapshots in `paper/workshop/numbers/`; confidence intervals cannot be recomputed from aggregate means alone. To recompute intervals, rerun generation and retain the per-item outputs.
