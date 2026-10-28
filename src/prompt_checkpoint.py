"""Atomic prompt-run manifests and shard identities; refuses mismatched resumes."""
import hashlib
import json
from pathlib import Path


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def ensure_manifest(directory, manifest):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'manifest.json'
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError(f'Run configuration changed: {path}. Use a new PC_RUN_LABEL.')
    else:
        # Exclusive creation prevents two different jobs claiming the same run directory.
        with path.open('x') as f:
            json.dump(manifest, f, indent=2)
    return directory


def shard_seed(seed, condition, language, start):
    return int(fingerprint([seed, condition, language, start])[:8], 16) % (2**31)


from contextlib import contextmanager


@contextmanager
def exclusive_run(directory):
    """Lock for the entire output transaction, released on errors as well as success."""
    if directory is None:
        yield
        return
    import fcntl
    with (Path(directory) / 'run.lock').open('w') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
