# `prompts` — how an MGSM prompt is actually assembled

`src/prompts.py` is the smallest module in the package and the easiest one to misread, because
almost nothing in it is a *choice we made*: the exemplars, the labels, and the answer keywords are
all inherited verbatim from the MGSM benchmark. This page is an exposition rather than an API
reference — it walks one real prompt end to end, then shows exactly which bytes the E8 `c3` prompt
control moves. Everything quoted below is rendered by `build_mgsm_prompt`, not paraphrased.

The module is pure: it never touches a model or a tokenizer. Chat-templating happens later, in the
harness. That is why the same builder serves the HF and vLLM paths identically.

## What "8-shot exemplars" means

MGSM is a **few-shot** benchmark. Before the model ever sees the question we want answered, the
prompt shows it eight already-solved problems — the *exemplars* — each rendered as a question
followed by a full worked chain of thought. "8-shot" is just the count of those worked examples.

They are not sampled or chosen. **MGSM's `train` split contains exactly eight rows**, the canonical
GSM8K-derived set from Shi et al. 2022, and `benchmarks.exemplars()` takes the first `n_shot` of
them. `n_shot=8` therefore means *all of them*, which is why the default is 8 and why changing it
only ever removes examples:

| # | exemplar (English) | answer |
|---|---|---|
| 1 | Roger has 5 tennis balls, buys 2 cans of 3… | 11 |
| 2 | Nine computers in the server room, five added Mon–Thu… | 29 |
| 3 | Leah had 32 chocolates, her sister 42, they ate 35… | 39 |
| 4 | Shawn has five toys, gets two each from mom and dad… | 9 |
| 5 | Michael had 58 golf balls, lost 23 Tuesday… | 33 |
| 6 | Olivia has \$23, buys five bagels at \$3… | 8 |
| 7 | Jason had 20 lollipops, gave Denny some, now has 12… | 8 |
| 8 | 3 cars in the lot, 2 more arrive… | 5 |

The exemplars do two jobs at once, and the second is the one this repo cares about:

1. **They teach the output format** — every worked solution ends in the answer keyword followed by
   a number, which is the string `scoring` later splits the generation on.
2. **They demonstrate what language to reason in.** The exemplars carry the chain of thought, so
   whatever language *they* are written in is the language the model imitates. This is the entire
   lever the `c3` prompt control pulls.

## One real prompt, in English

Here is a genuine `clean` English prompt, with the exemplar list cut to two for legibility — the
real thing repeats that block eight times before the final question:

```
Question: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis
balls. How many tennis balls does he have now?
Step-by-Step Answer: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls.
5 + 6 = 11. The answer is 11.

Question: There were nine computers in the server room. Five more computers were installed each
day, from monday to thursday. How many computers are now in the server room?
Step-by-Step Answer: There are 4 days from monday to thursday. 5 computers were added each day.
That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning, so
now there are 9 + 20 = 29 computers. The answer is 29.

                              … six more exemplars …

Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins
for her friends every day with four. She sells the remainder at the farmers' market daily for $2
per fresh duck egg. How much in dollars does she make every day at the farmers' market?
Answer:
```

The model is handed that string and generates the continuation. The full 8-shot version is about
2.7k characters in English; the same prompt in `te` runs to roughly 4.9k tokens on Qwen2.5's
byte-fallback BPE, which is why the vLLM KV window is floored at 8192 (see Corrections #4 in
`CLAUDE.md`).

### Two asymmetries worth knowing before you read a prompt dump

**The exemplars are labelled and the real question is not.** `Question:` and `Step-by-Step Answer:`
are baked into MGSM's `train` *field values* — they are not added by our builder — and the `test`
split carries neither. So the model sees eight blocks that open with `Question:` and then a ninth
that opens with nothing, ending in a bare `Answer:` cue. This is upstream MGSM behaviour, faithfully
reproduced, not a defect here.

**The cue is not the label the exemplars modelled.** The exemplars end their reasoning with the
in-sentence form *"The answer is 11."*, while the final cue is `Answer:`. The scorer splits on the
keyword either way, and empirically the model still produces a full chain of thought rather than a
bare number — clean generations average 200–490 tokens depending on language — so the mismatch is
harmless. It is, however, the first thing to suspect if a new model ever starts emitting bare
numerals.

## The localized answer keyword

The one table `prompts.py` actually owns. It terminates each exemplar's worked solution and forms
the final cue, and `scoring` splits the model's generation on it — so it is load-bearing for
correctness, not decoration:

| | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|
| `en` | `es` | `fr` | `de` | `ru` | `zh` | `ja` | `th` | `sw` | `bn` | `te` |
| Answer | Respuesta | Réponse | Antwort | Ответ | 答案 | 答え | คำตอบ | Jibu | উত্তর | సమాధానం |

These are the benchmark's canonical `LANG_TO_ANSWER_PREFIX` values, vendored and cited in the source
rather than machine-translated, precisely because a translated keyword would silently change what
the scorer matches.

## What the `c3` prompt control changes

E8's prompt control (`scripts/e8_prompt_control.py`) asks a deflationary question: the paper claims
subspace ablation raises accuracy by moving the model toward answering in another language — so how
much of that could you get by *just asking*? Three arms, **no intervention anywhere** — no hooks, no
subspace, no activation cache.

Each arm is one prompt built from the same three ingredients, so the contrast is exact:

| arm | the question | the 8 exemplars | the final cue |
|---|---|---|---|
| `clean` | native (e.g. Bengali) | **native** | native (`উত্তর:`) |
| `en_cot` | native (Bengali) | **English** | **English** (`Answer:`) |
| `en_question` | **English translation** | English | English (`Answer:`) |

`en_cot` is the interesting one. The problem stays in Bengali; only the demonstrated reasoning
becomes English. Rendered with one exemplar for legibility, the arm looks like this — note that the
two halves of the prompt are now in different languages:

```
Question: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis
balls. How many tennis balls does he have now?
Step-by-Step Answer: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls.
5 + 6 = 11. The answer is 11.

জেনেটের হাঁসগুলি প্রতিদিন 16টি করে ডিম পাড়ে। … তিনি কৃষকদের বাজারে প্রতিদিন কত ডলার উপার্জন করেন?
Answer:
```

against `clean`, which is Bengali throughout and ends `উত্তর:`.

Two things follow from the table that are easy to miss:

- **The cue moves with the exemplars.** `en_cot` swaps the final `উত্তর:` for `Answer:`, and that
  cue is the last token before generation begins — the single strongest signal of output language.
  The arm is therefore not a clean "exemplars only" manipulation; it moves two things together.
- **`en_question` is a ceiling, not an option.** It is the translate-everything upper bound, free
  because MGSM is parallel (problem *i* is the same problem in every language). It scores 0.902 on
  Qwen2.5-7B — and its response fidelity is **exactly zero by construction**. For a Bengali speaker
  a flawless English answer is a failed answer. That gap is the reason the joint metric exists.

`en_question` is also **one number, not a per-language curve**: its prompt is built from the English
rows regardless of `lang`, so it is the same 125 English prompts decoded ten times, identical up to
engine nondeterminism. The per-language copies exist only so `paired_delta` can pair within
`(model, task, lang)`.

## Where this is going

The arm set is being extended to five. The E9 trace showed the interventions send output to `zh` at
96–100%, not to English as the E8-era arms suggested — so `zh_cot` / `zh_question` are needed as the
*mechanism mimic*, while `en` remains the *ceiling* arm because English genuinely is this model's
best language (clean `en` 0.872 against `zh` 0.812). See item 6 of the todo list in
`../experiments/campaign_status.md` (research-workspace reference).

## API

```python
build_mgsm_prompt(item, exemplars=None, *, n_shot=0, instr_lang=None) -> str
```

`item` is the test record (`{"question": ...}`); `exemplars` are `train` records
(`{"question": ..., "answer": <worked CoT>}`), truncated to `n_shot`. `instr_lang` selects the
answer keyword and thus the cue — it is what `c3` overrides to `"en"` to build `en_cot`, while
leaving `item` in the native language. No `\boxed{}` instruction is ever added.

Also here: `build_mmlu_prompt` (M-MMLU log-likelihood context, localized A–D block and cue, reusing
the same keyword table) and `build_xwinograd_continuations` (substitutes each option into the `_`
blank, returning the two full continuations to be scored).
