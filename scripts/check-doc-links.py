"""Every relative markdown link under C:\\AI must resolve to a real file.

The docs are a navigation map -- a README in every folder pointing at the next
one. A map with a dead link is worse than no map, because an agent that follows
it concludes the document does not exist rather than that the link is wrong.

Run from anywhere:

    python C:\\AI\\scripts\\check-doc-links.py

Exit status 0 = every link resolves. 1 = at least one does not, listed.

Two things are deliberately NOT treated as links:

- anything inside a backtick code span. `[int](3/2)` is PowerShell in a results
  table, not a link to a directory called "3", and it appears in four reports.
- absolute URLs and `sandbox:` paths pasted in from external research replies.

Percent-escapes are decoded before the existence check: `Deep%20Research/` is a
real directory and a renderer resolves it, so the checker must too.
"""
import os
import re
import sys
import urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = (".git", "node_modules", "__pycache__", ".cache", ".venv")
SKIP_SCHEMES = ("http://", "https://", "mailto:", "sandbox:", "data:")

# Strip fenced blocks first, then inline code spans. Both can contain text that
# looks exactly like a link and is not one.
FENCE = re.compile(r"```.*?```", re.S)
SPAN = re.compile(r"`[^`\n]*`")
LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def links_in(text):
    text = SPAN.sub("`x`", FENCE.sub("", text))
    for m in LINK.finditer(text):
        target = m.group(1).strip()
        if target.startswith("<") and target.endswith(">"):
            target = target[1:-1].strip()
        if target.startswith("#") or target.lower().startswith(SKIP_SCHEMES):
            continue
        target = urllib.parse.unquote(target.split("#")[0]).strip()
        if target:
            yield target


def main():
    broken, checked, files = [], 0, 0
    for base, dirs, names in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in names:
            if not name.endswith(".md"):
                continue
            path = os.path.join(base, name)
            files += 1
            with open(path, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
            for target in links_in(text):
                checked += 1
                if not os.path.exists(os.path.normpath(os.path.join(base, target))):
                    broken.append((os.path.relpath(path, ROOT), target))

    print(f"{files} markdown files, {checked} relative links, {len(broken)} broken")
    for path, target in broken:
        print(f"  BROKEN  {path}  ->  {target}")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
