# grammars — GBNF output constraints

| file | what it enforces |
|---|---|
| `python-fence.gbnf` | exactly one fenced Python block, nothing outside it |

## Why this exists

The blocking failure on every Dynamic V3 artifact is **format, not reasoning**:
41.5 % (`UD-IQ1_M`) to 58.3 % (`UD-IQ2_XXS`) of corpus attempts emit **no fenced
code block at all**, having looped inside the reasoning block for
19,000–34,000 characters until the token cap.

A grammar makes the fence a property of the sampler rather than a hope about the
prompt: the model cannot emit a token the grammar forbids, so "no fenced block"
stops being a possible outcome.

## How it is used

Pair it with `--reasoning-budget 0`. The grammar admits no prose before the
fence, so a model that still wants to think has nowhere to put it.

```powershell
--grammar-file "C:\AI\qwen38-tuning\grammars\python-fence.gbnf" --reasoning-budget 0
```

`scripts/serve-v3-*-fmt.ps1` do exactly that, and are **byte-identical to their
unconstrained twins except for those two flags** — so the pair is a controlled
comparison whose control already has a corpus result.

## The shape of the grammar

```gbnf
root  ::= "```python\n" code "```"
code  ::= ( nonbt | "`" nonbt | "``" nonbt )*
nonbt ::= [^`]
```

Single and double backticks stay legal because docstrings and comments use them;
the sampler simply cannot reach three in a row before the closing fence.

## Before adding another

Check it parses without spending a boot:

```sh
llama-server --grammar-file <file> -m /nonexistent.gguf --port 18080
```

A grammar error appears before model loading is attempted.
