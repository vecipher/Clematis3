

import os
import tempfile

import pytest

from clematis.engine.stages.t2_quality_norm import (
    normalize_text,
    tokenize,
    load_alias_map,
    apply_aliases,
)


# ----------------------------- Normalization -----------------------------

def test_normalize_text_nfkc_case_whitespace():
    # Fullwidth ASCII should fold to ASCII; case lowered; whitespace collapsed
    s = "ＴＥＳＴ    Foo\t\nBar\u00A0Baz"  # includes NBSP
    out = normalize_text(s)
    assert out == "test foo bar baz"


# ------------------------------ Tokenization -----------------------------

def test_tokenize_basic_stopwords_and_minlen():
    text = "The and an bag of Tricks!!!"
    toks = tokenize(text, stopset=["the", "and"], min_token_len=3)
    # 'the' and 'and' removed; 'an' filtered by min length; punctuation split; lowercased
    assert toks == ["bag", "tricks"]


def test_tokenize_with_porter_lite_stemmer():
    text = "Running cats mended position studies"
    toks = tokenize(text, stemmer="porter-lite")
    # Crude suffix stripping as implemented in the lite stemmer
    # 'running' -> 'runn', 'cats' -> 'cat', 'mended' -> 'mend', 'position' -> 'posi', 'studies' -> 'studi'
    assert "runn" in toks
    assert "cat" in toks
    assert "mend" in toks
    assert "posi" in toks
    assert "studi" in toks


# -------------------------------- Aliasing --------------------------------

def test_apply_aliases_exact_rewrite_and_expansion():
    toks = ["cuda", "install", "llm"]
    amap = {"cuda": "nvidia_cuda", "llm": "large language model"}
    out = apply_aliases(toks, amap)
    assert out == ["nvidia_cuda", "install", "large", "language", "model"]


def test_apply_aliases_idempotent_single_pass():
    toks = ["llm", "pipeline"]
    amap = {"llm": "large language model"}
    once = apply_aliases(toks, amap)
    twice = apply_aliases(once, amap)
    assert once == twice  # idempotent by construction


# ----------------------------- Alias map I/O ------------------------------

def test_load_alias_map_missing_returns_empty():
    assert load_alias_map("/no/such/path/aliases.yaml") == {}


def test_load_alias_map_parses_simple_kv_without_yaml_dep():
    fd, path = tempfile.mkstemp(prefix="aliases_", suffix=".yaml")
    os.close(fd)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("""
# comment line
cuda: nvidia_cuda
llm: large language model
""")
        amap = load_alias_map(path)
        assert amap.get("cuda") == "nvidia_cuda"
        assert amap.get("llm") == "large language model"
    finally:
        try:
            os.remove(path)
        except OSError:
            pass