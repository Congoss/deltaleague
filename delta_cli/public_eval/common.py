import hashlib
import json
from importlib.metadata import version
from .catalog import SEED, VERSIONS


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), default=str).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def select(rows, count, namespace, excluded=()):
    """Stable hash sampling, independent of Python RNG and input ordering."""
    eligible = [(i, row) for i, row in enumerate(rows) if i not in excluded]
    ranked = sorted(eligible, key=lambda pair: digest([SEED, namespace, pair[1]]))
    if len(ranked) < count:
        raise ValueError(f'{namespace}: requested {count}, only {len(ranked)} eligible')
    return sorted(i for i, _ in ranked[:count])


def allocate(total, names):
    names = sorted(names)
    if total < len(names):
        raise ValueError('Sample must cover every stratum')
    return {name: total // len(names) + (i < total % len(names)) for i, name in enumerate(names)}


def check_versions():
    for name, expected in VERSIONS.items():
        if version(name) != expected:
            raise RuntimeError(f'{name} must be {expected}, got {version(name)}')


def read_manifest(path):
    data = json.loads(path.read_text())
    actual = digest({k: v for k, v in data.items() if k != 'sha256'})
    if data.get('sha256') != actual:
        raise ValueError('Validation manifest integrity mismatch')
    return data
