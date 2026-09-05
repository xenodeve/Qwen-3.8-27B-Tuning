"""Reasoning effort for the Qwen3.8 chat template (2026-09-04).

The template defaults `reasoning_effort` to xhigh and the fork's server never
passed the field, so every request ran at xhigh whatever the client asked.
`medium` injects no instruction text at all in this template; only xhigh and
low do. Default comes from EXL3_REASONING_EFFORT (xhigh when unset).
"""
import os

EFFORTS = ("xhigh", "medium", "low")   # what the Qwen3.8 chat template accepts
ALIAS = {"high": "xhigh", "max": "xhigh", "minimal": "low", "none": "low"}
DEFAULT = os.environ.get("EXL3_REASONING_EFFORT", "xhigh")


def resolve(value, default = None):
    """OpenAI `reasoning_effort` -> a value the template accepts, else the default."""
    default = default or DEFAULT
    v = str(value or default).lower()
    v = ALIAS.get(v, v)
    if v not in EFFORTS:
        print(f" ## effort: unknown reasoning_effort {value!r}, using {default}", flush = True)
        return default
    return v
