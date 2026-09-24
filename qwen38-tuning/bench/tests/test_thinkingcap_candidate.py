"""ThinkingCap-Qwen3.8-27B Q4_K_M as a gsq_compare candidate (2026-09-24).

bottlecapai's efficient-thinking finetune of Qwen3.8-27B; the GGUF carries the
MTP head (README: `--spec-type draft-mtp --spec-draft-n-max 3`). Downloaded from
the gated repo and hashed locally; the digest equals the Hub LFS oid, so a
swapped or partial file must fail loudly instead of producing a plausible run.
"""
import os
import sys

import pytest

BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BENCH)
import gsq_compare as comparison  # noqa: E402

DIGEST = 'fafa890ce2ce8531b4ade225c7dbd5f5d72a92303ca9ef72890c6cf78f19f299'


def test_thinkingcap_is_registered_with_the_hub_lfs_digest():
    comparison.verify_artifact_digest('thinkingcap_q4km', DIGEST)
    with pytest.raises(ValueError):
        comparison.verify_artifact_digest('thinkingcap_q4km', '0' * 64)
    meta = comparison.artifact_metadata('thinkingcap_q4km')
    assert meta['quant'] == 'Q4_K_M'
    assert 'ThinkingCap' in meta['artifact_family']


def test_thinkingcap_argv_uses_its_file_and_the_shared_operating_point():
    argv = comparison.llama_argv('thinkingcap_q4km', 65536, 'mtp-ngram', '9500,14500', 24)
    model = argv[argv.index('-m') + 1]
    assert model.endswith('ThinkingCap-Qwen3.8-27B-Q4_K_M.gguf')
    assert argv[argv.index('--spec-type') + 1] == 'draft-mtp,ngram-mod'
    assert argv[argv.index('--spec-draft-n-max') + 1] == '3'
    assert argv[argv.index('--spec-ngram-mod-n-match') + 1] == '24'
    assert argv[argv.index('-ts') + 1] == '9500,14500'
