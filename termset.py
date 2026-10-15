# encoding: utf-8
"""Bundled Tibetan/Sanskrit syllable set.

The data itself lives in termset_syllables.json.gz (a JSON array, gzip-compressed)
next to this module — moved out of source so the set is a data asset, not 6k lines
of code. The pyinstaller spec bundles *.json.gz automatically.
"""
import gzip
import json
import os


def _load_syllables():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'termset_syllables.json.gz')
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return set(json.load(f))


syllables = _load_syllables()
