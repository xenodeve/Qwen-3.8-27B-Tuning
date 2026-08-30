r"""Chunk a document, score its parts against a query, and say what was cut.

Part of `llama-buffer` (issue #54). No model, no VRAM, no GGUF conversion --
just enough retrieval to stop a client sending 29,000 tokens of file to answer
one question about it.

WHY CHARACTER N-GRAMS

The queries are Thai (`นี่คือแผนอะไร`) and **Thai is written without spaces**. A
whitespace tokeniser sees a single enormous token, matches nothing, and returns
whichever chunk sorted first -- it does not fail, it returns the WRONG extract,
and the model then answers confidently from it. Character n-grams have no such
failure mode. They are worse than a real Thai segmenter and better than anything
that assumes spaces.

Unsloth Studio runs `ragMode: "hybrid"` for what is probably the same reason:
its vector half is `bge-small-en-v1.5`, an English model, so on a Thai query the
keyword half is what carries it.

WHY THE MARKER IS NOT OPTIONAL

`render()` refuses to produce an empty extract and always states that the text
is an extract, which part of how many, and how to ask for the rest. A model
handed five chunks with nothing saying so answers as though it read the whole
document -- the exact confusion this tool exists to prevent, arriving by a
different door. Stated truncation is a fact the model can act on; silent
truncation is a believable wrong answer, which is what this repository's north
star is about.
"""
import math
import re
import unicodedata
from collections import Counter

# 3 is the shortest n that still discriminates Thai syllables, and short enough
# that an English word of four letters still produces two grams.
NGRAM = 3

_SPLIT = re.compile(r"[^\w฀-๿]+", re.UNICODE)


class Chunk:
    """A slice of a document that remembers where it came from.

    The offsets are not decoration: the marker injected into the prompt has to
    say WHICH part this was, or the model cannot ask for the rest.
    """

    __slots__ = ("text", "start", "end", "index")

    def __init__(self, text, start, end, index=0):
        self.text = text
        self.start = start
        self.end = end
        self.index = index

    def __repr__(self):
        return "Chunk(%d..%d, %r)" % (self.start, self.end, self.text[:40])


def tokens(s):
    """Character n-grams over normalised text, plus whole words where they exist.

    Both, on purpose. The n-grams are what make Thai work; the whole words keep
    an exact English match scoring above a coincidental gram overlap.
    """
    s = unicodedata.normalize("NFKC", s or "").casefold()
    out = []
    for word in _SPLIT.split(s):
        if not word:
            continue
        out.append(word)
        if len(word) <= NGRAM:
            continue
        for i in range(len(word) - NGRAM + 1):
            out.append(word[i:i + NGRAM])
    return out


def chunk(text, target_chars=1200):
    """Split on blank lines, then pack neighbours up to `target_chars`.

    Contiguous and lossless by construction -- every chunk's `end` is the next
    one's `start`, and concatenating them reproduces the input. A chunker that
    drops text is a retriever that cannot find it, and it would do so quietly.
    """
    if not text:
        return []
    parts, pos = [], 0
    for block in re.split(r"(\n\s*\n)", text):
        if not block:
            continue
        parts.append((pos, pos + len(block)))
        pos += len(block)

    out, start, end = [], None, None
    for a, b in parts:
        if start is None:
            start, end = a, b
        elif (b - start) > target_chars:
            out.append(Chunk(text[start:end], start, end, len(out)))
            start, end = a, b
        else:
            end = b
    if start is not None:
        out.append(Chunk(text[start:end], start, end, len(out)))
    # The tail must reach the end of the document even when the last split left
    # a trailing separator behind.
    if out and out[-1].end != len(text):
        last = out[-1]
        out[-1] = Chunk(text[last.start:], last.start, len(text), last.index)
    return out


def score(ch, query):
    """BM25-ish: idf is not available for one document, so this is tf saturation
    plus a length penalty. Enough to rank parts of ONE file against ONE query,
    which is all this is ever asked to do.
    """
    q = Counter(tokens(query))
    if not q:
        return 0.0
    d = Counter(tokens(ch.text))
    if not d:
        return 0.0
    dl = sum(d.values())
    k1, b, avg = 1.5, 0.75, 400.0
    total = 0.0
    for term, qn in q.items():
        f = d.get(term, 0)
        if not f:
            continue
        # Longer matches carry more information than a stray 3-gram.
        weight = 1.0 + math.log(1 + len(term))
        total += weight * qn * (f * (k1 + 1)) / (
            f + k1 * (1 - b + b * dl / avg))
    return total


def top_k(chunks, query, k=5):
    """The k best-scoring chunks, best first.

    Ties break on document order, so an empty or unmatched query degrades to
    "the beginning of the file" -- which is a defensible answer and, unlike a
    silent reordering, an obvious one.
    """
    ranked = sorted(chunks, key=lambda c: (-score(c, query), c.index))
    return ranked[:k]


def render(chunks, total, name, original_chars):
    """The chunks, wrapped in a marker that says what this is and is not.

    Refuses to render nothing: an empty extract under a confident heading is
    the worst output this module could produce.
    """
    if not chunks:
        raise ValueError(
            "refusing to render an empty extract for %r -- a marker with no "
            "content invites the model to answer from nothing" % (name,))
    shown = sorted(chunks, key=lambda c: c.index)
    kept = sum(len(c.text) for c in shown)
    head = (
        "[extract of %s -- %d of %d parts, %d of %d characters. "
        "The rest of this file was NOT sent. If the answer is not in these "
        "parts, say so and ask for the specific section you need rather than "
        "guessing.]"
        % (name, len(shown), total, kept, original_chars))
    body = "\n\n".join(
        "[part %d/%d, characters %d-%d]\n%s"
        % (c.index + 1, total, c.start, c.end, c.text.strip())
        for c in shown)
    return head + "\n\n" + body
