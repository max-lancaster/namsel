"""Safe, compact load/save (gzip + JSON) for the engine's OWN bundled data models.

Replaces pickle/shelve for plain-data dicts (character maps, n-gram / bigram
tables). JSON is data-only — loading it cannot execute arbitrary code, which is
the whole point of dropping pickle here — and gzip makes the files smaller than
the original pickles (e.g. syllable_bigram 57 MB -> 11 MB).

JSON object keys are always strings, so dict keys would lose their type. To keep
the models byte-for-byte equivalent, every dict is stored as a tagged list of
[key, value] pairs, which preserves int/str keys through a JSON round-trip.
"""
import gzip
import json


def _encode(o):
    if isinstance(o, dict):
        return {"__d__": [[_encode(k), _encode(v)] for k, v in o.items()]}
    if isinstance(o, (list, tuple)):
        return [_encode(x) for x in o]
    return o


def _decode(o):
    if isinstance(o, dict) and "__d__" in o:
        return {_decode(k): _decode(v) for k, v in o["__d__"]}
    if isinstance(o, list):
        return [_decode(x) for x in o]
    return o


def dump_model(obj, path):
    """Serialize a plain-data model (dict/list/int/str/float) to gzip+JSON."""
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(_encode(obj), f, ensure_ascii=False)


def load_model(path):
    """Load a model written by dump_model. Data-only; no code execution."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return _decode(json.load(f))
