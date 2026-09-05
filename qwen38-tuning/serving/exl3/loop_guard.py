"""Stop a generation that has degenerated into repetition (issue #76, 2026-09-05).

The 13:19 incident: after a correct page, the model's Thai report locked onto the
tone mark U+0E48 and ran 127,996 tokens in 46 minutes, because the output cap is
the whole window on purpose. The fork's iterate() hands the server text chunks,
so the guard watches text: it trips when the last WINDOW characters are made of
at most MAX_DISTINCT distinct characters, or are one short unit (period <=
MAX_PERIOD) repeated end to end. Ordinary prose, Thai with tone marks, CSS rules
and base64 runs shorter than the window all pass (bench/tests/test_exl3_loop_guard.py
holds real lines from the day's pages as negatives).

server.py feeds every text chunk through `feed()`; on True it cancels the job and
finishes the stream with finish_reason "length" and stop_reason "loop".
"""
import re

WINDOW = 512
MAX_DISTINCT = 2
MAX_PERIOD = 8
THINK_WINDOW = 4096       # the sentence-loop rule, thinking only (19:03: a 3-sentence cycle, 127,996 tokens)
THINK_UNIT = 64
THINK_REPEATS = 8
# markup drafted inside thinking repeats legitimately (both-r2: six section headers share a
# 64-character opening); only prose units count
PROSE_UNIT = re.compile(r"^[^<>{}|`]+$")


class LoopGuard:
    def __init__(self, window = WINDOW, max_distinct = MAX_DISTINCT, max_period = MAX_PERIOD,
                 think_window = THINK_WINDOW, think_unit = THINK_UNIT, think_repeats = THINK_REPEATS):
        self.window = window
        self.max_distinct = max_distinct
        self.max_period = max_period
        self.think_window = think_window
        self.think_unit = think_unit
        self.think_repeats = think_repeats
        self.tail = ""
        self.think_tail = ""
        self.n_chars = 0
        self.reason = None

    def feed(self, chunk, in_think = False):
        """Return True once, when the generated text has become a loop. `in_think`
        enables the long-unit rule, which content must not get: a page repeats card
        markup legitimately, thinking does not repeat a sentence a thousand times."""
        if self.reason:
            return True
        if not chunk:
            return False
        self.n_chars += len(chunk)
        self.tail = (self.tail + chunk)[-self.window:]
        if in_think:
            self.think_tail = (self.think_tail + chunk)[-self.think_window:]
            if len(self.think_tail) >= self.think_window:
                unit = self.think_tail[-self.think_unit:]
                n = self.think_tail.count(unit) if PROSE_UNIT.match(unit) else 0
                if n >= self.think_repeats:
                    self.reason = f"unit of {self.think_unit} chars repeated {n} times in the last {self.think_window} of thinking"
                    return True
        if len(self.tail) < self.window:
            return False
        distinct = len(set(self.tail))
        if distinct <= self.max_distinct:
            self.reason = f"{distinct} distinct characters in the last {self.window}"
            return True
        for p in range(1, self.max_period + 1):
            unit = self.tail[:p]
            if self.tail == (unit * (self.window // p + 1))[:self.window]:
                self.reason = f"period {p} repeated across the last {self.window}"
                return True
        return False
