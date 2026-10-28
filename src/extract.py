from __future__ import annotations
import json
import os
from pathlib import Path
import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import ACTS_DIR, get_device
from corpus import _load_records, build_chat_inputs, load_parallel, select_problem_indices
from models import model_spec

_HF_HTTP_CONFIGURED = False


def _configure_hf_http() -> None:
    """Make huggingface_hub's httpx client verify TLS behind a corporate proxy.

    huggingface_hub 1.x uses httpx, which ignores SSL_CERT_FILE/REQUESTS_CA_BUNDLE.
    Behind a TLS-intercepting proxy this breaks downloads. We prefer `truststore`
    (macOS/Windows native verifier via the system keychain) because OpenSSL 3.x
    rejects some corporate CA certs that lack a keyUsage extension; we fall back to
    a CA-bundle env var otherwise. Rebuilding the client via the factory also resets
    the cached (possibly closed) global client.
    """
    global _HF_HTTP_CONFIGURED
    if _HF_HTTP_CONFIGURED:
        return

    # Xet uses a separate Rust HTTP/TLS stack that bypasses the httpx client and
    # truststore configured below, so it stalls behind a TLS-intercepting proxy.
    # Force the standard HTTPS download path instead. The constant is read at import
    # time, so patch it live too (file_download reads constants.HF_HUB_DISABLE_XET).
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    # A slow corporate proxy can exceed HF's default 10s read timeout on larger
    # files; raise it. These constants are read at import time, so patch live too.
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    try:
        import huggingface_hub.constants as _hf_constants

        _hf_constants.HF_HUB_DISABLE_XET = True
        _hf_constants.HF_HUB_DOWNLOAD_TIMEOUT = 120
        _hf_constants.HF_HUB_ETAG_TIMEOUT = 60
    except Exception:
        pass

    verify = None
    try:
        import ssl
        import truststore

        verify = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except Exception:
        ca = (
            os.environ.get("SSL_CERT_FILE")
            or os.environ.get("REQUESTS_CA_BUNDLE")
            or os.environ.get("CURL_CA_BUNDLE")
        )
        if ca and os.path.exists(ca):
            verify = ca

    if verify is None:
        _HF_HTTP_CONFIGURED = True
        return

    import httpx
    from huggingface_hub.utils import _http

    # huggingface_hub < 1.0 is requests-based: no client factory to swap, and requests already
    # honours REQUESTS_CA_BUNDLE / SSL_CERT_FILE on its own. The vLLM conda envs on Prajna pin
    # hub 0.36 (vLLM 0.8.5 caps it), so this branch is what they take; without it the first
    # benchmark load died with AttributeError: no attribute 'set_client_factory' (job 301259).
    if not hasattr(_http, "set_client_factory"):
        _HF_HTTP_CONFIGURED = True
        return

    def factory() -> "httpx.Client":
        return httpx.Client(
            event_hooks={"request": [_http.hf_request_event_hook]},
            follow_redirects=True,
            timeout=None,
            verify=verify,
            trust_env=True,  # keep honoring http(s)_proxy / NO_PROXY
        )

    _http.set_client_factory(factory)  # also calls close_session() -> fresh client
    _HF_HTTP_CONFIGURED = True


def _model_load_kwargs(cfg, spec):
    """Pure from_pretrained kwargs for (tokenizer, model) given a `ModelSpec` — testable, no network.

    Default spec => `trust_remote_code=False` (== today's omission), `dtype=cfg.dtype`, no
    `attn_implementation`, so the load is byte-identical for every unregistered / non-reasoning tag.
    """
    tok_kwargs = {"trust_remote_code": spec.trust_remote_code}
    model_kwargs = {"dtype": spec.dtype or cfg.dtype, "trust_remote_code": spec.trust_remote_code}
    if spec.attn_implementation:
        model_kwargs["attn_implementation"] = spec.attn_implementation
    if spec.fit == "quant4bit":
        # 4-bit NF4 so a 14B (~28 GB bf16) fits one 24 GB A10G on g5.12xlarge. Needs bitsandbytes +
        # CUDA at load time (from_pretrained raises a clear error if absent). Non-quant params (norms/
        # embeddings) keep the compute dtype; forward hooks / model.model.layers are unaffected.
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=(spec.dtype or cfg.dtype),
        )
    return tok_kwargs, model_kwargs


def load_model_and_tokenizer(cfg, device, spec=None):
    spec = spec if spec is not None else model_spec(cfg.model_name)
    _configure_hf_http()
    tok_kwargs, model_kwargs = _model_load_kwargs(cfg, spec)
    # Transformers before 4.56 accepts torch_dtype, not dtype. Keep the same
    # requested precision when using the separately pinned Prajna GLM environment.
    from packaging.version import Version
    import transformers
    if Version(transformers.__version__) < Version("4.56"):
        model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
    revision = os.environ.get("LRD_MODEL_REVISION")
    if revision:
        tok_kwargs["revision"] = revision
        model_kwargs["revision"] = revision
    tok = AutoTokenizer.from_pretrained(cfg.model_name, **tok_kwargs)
    tok.padding_side = "left"                                  # last token must be real, not pad
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, device_map=device, **model_kwargs
    ).eval()
    # Authoritative EOS fix (e.g. Yi <|im_end|>=7 vs gen_config eos=2): generate() reads this on
    # every path. Routed here (single/list preserved) rather than a single tok.eos_token_id into
    # each generate call, which would drop Llama-3.1's second terminator [128001, 128009].
    if spec.eos_token_id is not None:
        model.generation_config.eos_token_id = spec.eos_token_id
    return model, tok

def _mps_mem_gb() -> tuple[float, float]:
    """(current_allocated, driver_allocated) MPS memory in GiB, or (nan, nan) off-MPS.

    `current_allocated` is live tensor bytes; `driver_allocated` is what the MPS caching
    allocator has reserved from the OS (the figure that *ratchets* with the largest batch
    and is the swap-risk proxy on unified memory).
    """
    mps = getattr(torch, "mps", None)
    if mps is None or not torch.backends.mps.is_available():
        return float("nan"), float("nan")
    gib = 1024 ** 3
    cur = mps.current_allocated_memory() / gib
    drv = (mps.driver_allocated_memory() / gib
           if hasattr(mps, "driver_allocated_memory") else float("nan"))
    return cur, drv


def _mps_empty_cache() -> None:
    """Return cached-but-free blocks to the OS (resets the reserved-memory ratchet on MPS)."""
    mps = getattr(torch, "mps", None)
    if mps is not None and torch.backends.mps.is_available() and hasattr(mps, "empty_cache"):
        mps.empty_cache()


def _on_cuda(device) -> bool:
    return torch.cuda.is_available() and str(device).startswith("cuda")


def _dev_mem_gb(device) -> tuple[float, float]:
    """(live, reserved) accelerator memory in GiB for `device`, or (nan, nan) on CPU.

    The MPS-only `_mps_mem_gb` above predates the move to Prajna and silently returns
    (nan, nan) on CUDA, which is why every cluster run so far has logged no memory figure
    at all. This is the device-generic version: on CUDA `live` is `memory_allocated` (tensor
    bytes) and `reserved` is `memory_reserved` (what the caching allocator holds from the
    driver, the figure that actually has to fit on the card).
    """
    if _on_cuda(device):
        gib = 1024 ** 3
        return torch.cuda.memory_allocated(device) / gib, torch.cuda.memory_reserved(device) / gib
    return _mps_mem_gb()


def _dev_reset_peak_mem(device) -> None:
    """Zero the CUDA peak-memory counters so a later `_dev_peak_mem_gb` measures one phase."""
    if _on_cuda(device):
        torch.cuda.reset_peak_memory_stats(device)


def _dev_peak_mem_gb(device) -> tuple[float, float]:
    """(peak_allocated, peak_reserved) since the last reset, in GiB; (nan, nan) off CUDA.

    `peak_reserved` is the number to compare against the card: it is what the process held
    from the driver at its high-water mark. It still *excludes* the CUDA context itself
    (roughly 0.3-0.6 GiB), so `nvidia-smi --query-gpu=memory.used` reads a little higher —
    see the sampler in the sbatch wrappers, which records that independently.
    """
    if _on_cuda(device):
        gib = 1024 ** 3
        return (torch.cuda.max_memory_allocated(device) / gib,
                torch.cuda.max_memory_reserved(device) / gib)
    return float("nan"), float("nan")


def _pack_batches(order: list[int], lengths: list[int], batch_size: int,
                  max_batch_tokens: int | None) -> list[list[int]]:
    """Greedily group length-sorted indices into batches.

    Caps each batch at `batch_size` prompts *and* (when set) `max_batch_tokens` padded tokens
    = batch_count x batch_max_len. Since `order` is ascending, the last prompt added is the
    batch's pad length, so the token cap shrinks batches as prompts get longer (the 2k-token
    outliers end up near-solo) instead of letting a fixed count of 4 balloon the transient.
    A single prompt longer than the budget still gets its own batch.
    """
    batches: list[list[int]] = []
    cur: list[int] = []
    for j in order:
        L = lengths[j]
        over_count = len(cur) + 1 > batch_size
        over_tokens = max_batch_tokens is not None and (len(cur) + 1) * L > max_batch_tokens
        if cur and (over_count or over_tokens):
            batches.append(cur)
            cur = []
        cur.append(j)
    if cur:
        batches.append(cur)
    return batches


@torch.no_grad()
def extract_last_token_per_layer(model, tok, prompts: list[str], device,
                                 batch_size: int = 16, profile: bool = False,
                                 empty_cache_every: int = 8,
                                 max_batch_tokens: int | None = 4096,
                                 pbar=None) -> Tensor:
    """Final-token residual stream at every block output (hidden_states[1:]), fp32/CPU.
    Returns (n_prompts, n_layers, d_model). hidden_states[k+1] = output of block k.

    Prompts are **length-bucketed**: we tokenize once up front, process batches in ascending
    token-length order so a long prompt never pads short ones (cuts O(L^2) padded compute and
    keeps the MPS allocator reusing same-size blocks instead of fragmenting), then scatter
    results back into the original prompt order so the output stays index-aligned. Batches are
    capped at `batch_size` prompts and `max_batch_tokens` padded tokens (set the latter to None
    to disable), so long outliers shrink the batch instead of spiking the transient. Every
    `empty_cache_every` batches we release the MPS cache so reserved memory can't ratchet up
    across the run (set 0 to disable).

    Set `profile=True` to log per-batch seq_len, wall-time, and MPS memory (live + reserved).
    """
    import time
    # Route through build_chat_inputs so extraction uses the SAME templating as eval,
    # incl. enable_thinking=False for Qwen3-style reasoning models (else the last-token
    # residual stream is captured at a different final token than eval sees). No-op for
    # non-thinking templates (Qwen2.5), which ignore the flag.
    templated = build_chat_inputs(tok, prompts)
    # token lengths via the same tokenization path used below (no padding) -> bucket ascending
    lengths = [len(ids) for ids in tok(templated)["input_ids"]]
    order = sorted(range(len(templated)), key=lambda j: lengths[j])
    packed = _pack_batches(order, lengths, batch_size, max_batch_tokens)

    # Only the hidden states are kept, so ask for vocabulary logits at the last position only. A
    # full-sequence lm_head is batch x seq x ~152k vocab: on 8-shot MGSM prompts that was an 8.6 GiB
    # allocation, which ran 44 GB L40/A40 cards out of memory (E11 jobs 315804, 315806) while the
    # 80 GB A100 got by. The hidden states come before lm_head and are bit-identical either way
    # (checked on a Qwen2 model, 2026-09-21). Older transformers without the argument are unchanged.
    fwd_kwargs = dict(output_hidden_states=True, use_cache=False)
    import inspect
    if "logits_to_keep" in inspect.signature(model.forward).parameters:
        fwd_kwargs["logits_to_keep"] = 1

    out: list[Tensor | None] = [None] * len(prompts)
    for bi, sel in enumerate(packed):
        batch = [templated[j] for j in sel]
        enc = tok(batch, return_tensors="pt", padding=True).to(device)
        if profile:
            t0 = time.perf_counter()
        hs = model(**enc, **fwd_kwargs).hidden_states
        stacked = torch.stack(hs[1:], dim=0)[:, :, -1, :]      # drop embeddings; last (left-padded) token
        res = stacked.transpose(0, 1).float().cpu()            # (batch, layers, d)
        for k, j in enumerate(sel):
            out[j] = res[k]
        if empty_cache_every and bi % empty_cache_every == 0:
            _mps_empty_cache()
        if pbar is not None:
            pbar.update(len(sel))
        if profile:
            if _on_cuda(device):
                torch.cuda.synchronize(device)                 # don't time async dispatch only
            elif getattr(torch, "mps", None) is not None and torch.backends.mps.is_available():
                torch.mps.synchronize()
            dt = time.perf_counter() - t0
            seq_len = int(enc["input_ids"].shape[1])
            bs = int(enc["input_ids"].shape[0])
            cur, drv = _dev_mem_gb(device)
            tag = "cuda" if _on_cuda(device) else "mps"
            print(f"    batch {bi:3d}  bs={bs}  seq_len={seq_len:5d}  "
                  f"{dt:6.2f}s  {tag}_live={cur:5.2f}G  {tag}_reserved={drv:5.2f}G", flush=True)
    return torch.stack(out, dim=0)


def massive_activation_report(acts: dict[str, Tensor], threshold: float = 1000.0,
                              top_k: int = 5) -> dict:
    """Flag 'massive activation' outlier dimensions per layer in an `{lang: (n, L, d)}` cache.

    Some models (notably Qwen 2.5/3 at 7B+) carry a few residual-stream coordinates ~1000x the
    typical magnitude (attention-sink / bias channels). They wreck any variance-based subspace
    fit and any norm-sensitive metric (see `probe.unit_rows`), so it is worth catching them the
    moment activations are loaded rather than puzzling over broken downstream numbers.

    For each layer we pool all languages/problems, take the per-dimension max |activation|, and
    compare the largest to the median |activation|. Returns
    ``{"flagged": bool, "layers": [{"layer", "ratio", "max_abs", "median_abs", "top_dims"}...],
       "worst": <that dict>}``; a layer is flagged when ``ratio >= threshold``.
    """
    langs = list(acts)
    n_layers = acts[langs[0]].shape[1]
    rows = []
    for k in range(n_layers):
        X = torch.cat([acts[l][:, k, :].float() for l in langs], dim=0).abs()
        med = float(X.median())
        per_dim_max = X.max(dim=0).values
        top = torch.topk(per_dim_max, min(top_k, per_dim_max.numel()))
        max_abs = float(top.values[0])
        ratio = max_abs / med if med > 0 else float("inf")
        rows.append({
            "layer": k,
            "ratio": ratio,
            "max_abs": max_abs,
            "median_abs": med,
            "top_dims": [int(i) for i in top.indices.tolist()],
        })
    worst = max(rows, key=lambda r: r["ratio"])
    return {"flagged": worst["ratio"] >= threshold, "layers": rows, "worst": worst,
            "threshold": threshold}


def print_massive_activation_report(acts: dict[str, Tensor], threshold: float = 1000.0) -> dict:
    """Run `massive_activation_report` and print a one-line verdict (+ worst layer if flagged)."""
    rep = massive_activation_report(acts, threshold=threshold)
    w = rep["worst"]
    if rep["flagged"]:
        print(f"WARNING massive activations: layer {w['layer']} has max|act|/median|act| = "
              f"{w['ratio']:.0f}x (>= {threshold:.0f}x); dims {w['top_dims']} "
              f"(max|act|={w['max_abs']:.0f}, median={w['median_abs']:.3f}). "
              f"Variance-PCA subspaces use an angular fit to stay robust (see probe.unit_rows).")
    else:
        print(f"massive-activation check ok: worst layer {w['layer']} "
              f"max|act|/median = {w['ratio']:.0f}x (< {threshold:.0f}x threshold).")
    return rep


def acts_cache_path(cfg, cache_dir=None) -> Path:
    """Where per-language activations for `cfg` are cached.

    The filename encodes the sampling scheme + seed (e.g.
    ``acts_<model>_n512_stratified-s0.pt``) so a stratified cache is never confused with a
    legacy first-n one. `cache_dir` defaults to `config.ACTS_DIR`.
    """
    cache_dir = Path(cache_dir) if cache_dir is not None else ACTS_DIR
    model = cfg.model_name.split("/")[-1]
    return cache_dir / f"acts_{model}_n{cfg.n_samples}_{cfg.sample}-s{cfg.seed}.pt"


def acts_meta_path(cfg, cache_dir=None) -> Path:
    """Sidecar JSON (chosen problem indices + labels + provenance) for `acts_cache_path`."""
    return acts_cache_path(cfg, cache_dir).with_suffix(".meta.json")


def load_acts_meta(cfg, cache_dir=None) -> dict:
    """Read the sidecar metadata (``indices``, ``labels``, provenance) for a cached run."""
    with open(acts_meta_path(cfg, cache_dir)) as f:
        return json.load(f)


def _download_acts_from_hf(cfg, cache_dir=None, repo_id: str | None = None) -> Path | None:
    """Download a cached activation tensor + sidecar from the configured HF dataset repo.

    Returns the local tensor path if both files were found, otherwise ``None`` so callers can
    fall through to extraction.
    """
    if not (repo_id or cfg.acts_repo):
        return None
    _configure_hf_http()
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

    repo_id = repo_id or cfg.acts_repo
    path = acts_cache_path(cfg, cache_dir)
    meta_path = acts_meta_path(cfg, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=path.name,
            repo_type="dataset",
            revision=cfg.acts_revision,
            local_dir=path.parent,
        )
        hf_hub_download(
            repo_id=repo_id,
            filename=meta_path.name,
            repo_type="dataset",
            revision=cfg.acts_revision,
            local_dir=meta_path.parent,
        )
    except (EntryNotFoundError, RepositoryNotFoundError):
        return None
    return path if path.exists() and meta_path.exists() else None


def upload_acts(cfg, cache_dir=None, repo_id: str | None = None, private: bool = True) -> None:
    """Upload a local activation cache and metadata sidecar to a HF dataset repo."""
    _configure_hf_http()
    from huggingface_hub import HfApi

    repo_id = repo_id or cfg.acts_repo
    path = acts_cache_path(cfg, cache_dir)
    meta_path = acts_meta_path(cfg, cache_dir)
    missing = [str(p) for p in (path, meta_path) if not p.exists()]
    if missing:
        raise FileNotFoundError("missing activation cache artifact(s): " + ", ".join(missing))

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    for artifact in (path, meta_path):
        api.upload_file(
            path_or_fileobj=str(artifact),
            path_in_repo=artifact.name,
            repo_id=repo_id,
            repo_type="dataset",
        )


def load_or_extract_acts(cfg, device=None, batch_size: int = 16,
                         force: bool = False, cache_dir=None,
                         profile: bool = False, progress: bool = True,
                         allow_download: bool = True) -> dict[str, Tensor]:
    """Per-language final-token activations `{lang: (n, n_layers, d)}`, cached on disk.

    Loads `acts_cache_path(cfg, cache_dir)` if it exists (and not `force`); otherwise, when
    allowed, attempts to download the matching cache from `cfg.acts_repo`. Extraction runs only
    when both local and remote caches miss. The model pass is the only non-CPU work in E3 / E4
    Phase 0.
    """
    path = acts_cache_path(cfg, cache_dir)
    if path.exists() and not force:
        return torch.load(path, weights_only=True)
    if allow_download and not force:
        downloaded = _download_acts_from_hf(cfg, cache_dir)
        if downloaded is not None:
            return torch.load(downloaded, weights_only=True)
    device = device or get_device()
    model, tok = load_model_and_tokenizer(cfg, device)

    # choose the problems once (stratified by default), reuse for every language, and persist
    # the selection so downstream analyses (splits, the E4 content probe) can recover it
    canonical = _load_records(cfg.langs[0], cfg)
    indices, labels = select_problem_indices(
        canonical, cfg.n_samples, sample=cfg.sample, seed=cfg.seed, strata_keys=cfg.strata_keys
    )
    prompts = load_parallel(
        cfg.langs, n=cfg.n_samples, prompt_key=cfg.prompt_key, cfg=cfg, indices=indices
    )
    # one progress bar across all languages (total = prompts x langs); profile logs take over
    # the output if requested, so we suppress the bar then to avoid interleaved lines.
    pbar = None
    if progress and not profile:
        from tqdm.auto import tqdm
        total = sum(len(prompts[lang]) for lang in cfg.langs)
        pbar = tqdm(total=total, unit="prompt", desc="extracting acts")

    acts = {}
    # Zero the peak counters so the figure reported below covers the model pass only, not
    # whatever the weight load transiently held. Deliberately NOT paired with a CUDA
    # empty_cache() in the loop: releasing the caching allocator mid-run would deflate the
    # very high-water mark we are trying to record, and on a dedicated card there is nothing
    # to release it *to*. (MPS is different — unified memory really is shared, hence below.)
    _dev_reset_peak_mem(device)
    for lang in cfg.langs:
        if profile:
            print(f"  [{lang}] extracting {len(prompts[lang])} prompts (bs={batch_size})", flush=True)
        if pbar is not None:
            pbar.set_postfix_str(lang)
        acts[lang] = extract_last_token_per_layer(
            model, tok, prompts[lang], device, batch_size, profile=profile, pbar=pbar
        )
        _mps_empty_cache()                                     # reset reserved between languages
    if pbar is not None:
        pbar.close()
    # Report the high-water mark unconditionally (not behind `profile`), because on the cluster
    # this is the only record that survives: it decides whether a model fits a 48 GB L40S and so
    # whether this work can route around the dgx QOS ceiling. Costs one formatted line.
    peak_alloc, peak_res = _dev_peak_mem_gb(device)
    if peak_alloc == peak_alloc:                               # not NaN -> we are on CUDA
        total = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3
        print(f"peak GPU memory: allocated {peak_alloc:.2f} GiB, reserved {peak_res:.2f} GiB "
              f"of {total:.2f} GiB on {torch.cuda.get_device_name(device)} "
              f"(bs={batch_size}, headroom {total - peak_res:.2f} GiB)", flush=True)
    print_massive_activation_report(acts)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(acts, path)
    with open(acts_meta_path(cfg, cache_dir), "w") as f:
        json.dump(
            {
                "model": cfg.model_name,
                "n": cfg.n_samples,
                "sample": cfg.sample,
                "seed": cfg.seed,
                "strata_keys": list(cfg.strata_keys),
                "indices": [int(i) for i in indices],
                "labels": labels,
            },
            f,
        )
    return acts
