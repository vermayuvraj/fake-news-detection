"""Static validator for the manuscript. Catches the errors a LaTeX run would.

Usage:  python validate_latex.py

Checks performed:
  1.  brace balance (ignoring comments and escaped braces)
  2.  \begin/\end environment matching and nesting
  3.  every \cite key resolves to a refs.bib entry
  4.  every \ref/\eqref target has a matching \label
  5.  every \label is referenced at least once (figures/tables must be cited)
  6.  every \includegraphics file exists on disk
  7.  suspicious unescaped % and _ characters
  8.  duplicate \label definitions and duplicate BibTeX keys
  9.  tabular column-count consistency
 10.  BibTeX entry field sanity (author/title/year present)
"""

import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEX = HERE / "main.tex"
BIB = HERE / "refs.bib"

errors, warnings = [], []


def err(m):
    errors.append(m)


def warn(m):
    warnings.append(m)


raw_lines = TEX.read_text(encoding="utf-8").splitlines()


def strip_comment(line):
    """Remove a LaTeX comment, respecting \\% escapes."""
    out, i = [], 0
    while i < len(line):
        if line[i] == "\\" and i + 1 < len(line):
            out.append(line[i:i + 2])
            i += 2
            continue
        if line[i] == "%":
            break
        out.append(line[i])
        i += 1
    return "".join(out)


code_lines = [strip_comment(l) for l in raw_lines]
code = "\n".join(code_lines)

# --- 1. brace balance ------------------------------------------------------
depth = 0
for n, line in enumerate(code_lines, 1):
    i = 0
    while i < len(line):
        if line[i] == "\\" and i + 1 < len(line):
            i += 2
            continue
        if line[i] == "{":
            depth += 1
        elif line[i] == "}":
            depth -= 1
            if depth < 0:
                err(f"line {n}: unmatched closing brace")
                depth = 0
        i += 1
if depth != 0:
    err(f"brace imbalance at end of file: {depth} unclosed '{{'")

# --- 2. environments -------------------------------------------------------
stack = []
env_re = re.compile(r"\\(begin|end)\{([^}]+)\}")
for n, line in enumerate(code_lines, 1):
    for kind, name in env_re.findall(line):
        if kind == "begin":
            stack.append((name, n))
        else:
            if not stack:
                err(f"line {n}: \\end{{{name}}} with no matching \\begin")
            elif stack[-1][0] != name:
                err(f"line {n}: \\end{{{name}}} closes \\begin{{{stack[-1][0]}}} "
                    f"opened at line {stack[-1][1]}")
                stack.pop()
            else:
                stack.pop()
for name, n in stack:
    err(f"line {n}: \\begin{{{name}}} never closed")

# --- 3. citations ----------------------------------------------------------
bib_text = BIB.read_text(encoding="utf-8")
bib_keys = re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", bib_text)
dupe_bib = [k for k, c in Counter(bib_keys).items() if c > 1]
for k in dupe_bib:
    err(f"duplicate BibTeX key: {k}")

cited = set()
for m in re.finditer(r"\\cite[tp]?\{([^}]*)\}", code):
    for k in m.group(1).split(","):
        k = k.strip()
        if k:
            cited.add(k)
missing = sorted(cited - set(bib_keys))
for k in missing:
    err(f"\\cite{{{k}}} has no entry in refs.bib")
unused = sorted(set(bib_keys) - cited)
for k in unused:
    warn(f"bib entry never cited (will not appear in references): {k}")

# --- 4/5. labels and refs --------------------------------------------------
labels = re.findall(r"\\label\{([^}]+)\}", code)
dupe_lab = [k for k, c in Counter(labels).items() if c > 1]
for k in dupe_lab:
    err(f"duplicate \\label: {k}")

refs = set()
for m in re.finditer(r"\\(?:eq)?ref\{([^}]+)\}", code):
    refs.add(m.group(1))
for r in sorted(refs - set(labels)):
    err(f"\\ref{{{r}}} points to an undefined label")
for l in sorted(set(labels) - refs):
    kind = l.split(":")[0]
    if kind in {"fig", "tab", "alg"}:
        # Floats must be discussed in the text; this is a hard requirement.
        err(f"{kind} label '{l}' is never referenced in the text "
            f"(every float must be cited)")
    elif kind == "eq":
        warn(f"equation '{l}' is never cross-referenced")
    # Section labels exist for navigation; silence is fine.

# --- 6. graphics files -----------------------------------------------------
for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", code):
    p = HERE / m.group(1)
    candidates = [p] + [p.with_suffix(s) for s in (".pdf", ".png", ".jpg", ".eps")]
    if not any(c.exists() for c in candidates):
        err(f"missing graphics file: {m.group(1)}")

# --- 7. suspicious characters ---------------------------------------------
# A bare % in the ORIGINAL text truncates the rest of the line.
for n, line in enumerate(raw_lines, 1):
    stripped = strip_comment(line)
    tail = line[len(stripped):]
    if tail.startswith("%"):
        body = tail[1:].strip()
        # A real comment is fine; flag ones that look like prose was eaten.
        if re.match(r"^[a-z0-9]", body) and len(body.split()) > 2 and \
           not body.startswith(("=", "-", "Comment")):
            warn(f"line {n}: check this comment is intentional -> %{body[:60]}")

# Unescaped underscore outside math mode. Arguments that LaTeX never typesets
# (filenames, labels, keys, URLs) legitimately contain raw underscores, so blank
# them out before scanning.
_scan = code
for _cmd in ("includegraphics", "url", "label", "ref", "eqref", "cite",
             "bibliography", "bibliographystyle", "input", "include",
             "usepackage", "documentclass", "hypersetup", "newcommand"):
    _scan = re.sub(r"(\\" + _cmd + r"(?:\[[^\]]*\])?\{)([^}]*)(\})",
                   lambda m: m.group(1) + "X" * len(m.group(2)) + m.group(3),
                   _scan)
code_for_underscore = _scan

math_spans = []
for m in re.finditer(r"\$[^$]*\$|\\begin\{(?:equation|align|equation\*|align\*)\}"
                     r".*?\\end\{(?:equation|align|equation\*|align\*)\}",
                     code, re.DOTALL):
    math_spans.append((m.start(), m.end()))


def in_math(pos):
    return any(a <= pos < b for a, b in math_spans)


for m in re.finditer(r"(?<!\\)_", code_for_underscore):
    if not in_math(m.start()):
        line_no = code_for_underscore[:m.start()].count("\n") + 1
        ctx = code_lines[line_no - 1].strip()[:70]
        err(f"line {line_no}: unescaped '_' outside math mode -> {ctx}")

# --- 9. tabular column consistency ---------------------------------------
def col_count(spec):
    spec = re.sub(r"@\{[^}]*\}", "", spec)
    spec = re.sub(r"p\{[^}]*\}", "p", spec)
    spec = re.sub(r"[|\s]", "", spec)
    return len(re.findall(r"[lcrp]", spec))


def brace_arg(text, open_pos):
    """Return (argument, index_after) for a balanced {...} starting at open_pos."""
    assert text[open_pos] == "{"
    d, i = 0, open_pos
    while i < len(text):
        if text[i] == "\\":
            i += 2
            continue
        if text[i] == "{":
            d += 1
        elif text[i] == "}":
            d -= 1
            if d == 0:
                return text[open_pos + 1:i], i + 1
        i += 1
    return "", len(text)


for m in re.finditer(r"\\begin\{tabular\}\s*(?:\[[^\]]*\])?(?=\{)", code):
    spec, after = brace_arg(code, m.end())
    end = code.find(r"\end{tabular}", after)
    if end == -1:
        continue
    ncol = col_count(spec)
    body = code[after:end]
    start_line = code[:m.start()].count("\n") + 1
    for row in body.split(r"\\"):
        r = row.strip()
        if not r or r.startswith("\\") and "&" not in r:
            continue
        if "multicolumn" in r or "cmidrule" in r or "midrule" in r or \
           "toprule" in r or "bottomrule" in r:
            continue
        amps = len(re.findall(r"(?<!\\)&", r))
        if amps and amps + 1 != ncol:
            warn(f"tabular near line {start_line}: row has {amps + 1} cells, "
                 f"column spec declares {ncol}: {r[:60]}")

# --- 10. bib field sanity -------------------------------------------------
for m in re.finditer(r"@(\w+)\s*\{\s*([^,]+),(.*?)\n\}", bib_text, re.DOTALL):
    etype, key, body = m.group(1).lower(), m.group(2).strip(), m.group(3)
    fields = {f.lower() for f in re.findall(r"(\w+)\s*=", body)}
    need = {"title", "year"}
    if etype not in {"misc"}:
        need |= {"author"} if "editor" not in fields else set()
    for f in need - fields:
        warn(f"bib entry '{key}' missing field '{f}'")

# --- report ---------------------------------------------------------------
print("=" * 68)
print(f"LaTeX validation: {TEX.name}")
print("=" * 68)
print(f"lines: {len(raw_lines)}   environments checked: "
      f"{len(env_re.findall(code))}   labels: {len(labels)}   "
      f"citations: {len(cited)}/{len(bib_keys)} bib entries used")
print()
if errors:
    print(f"ERRORS ({len(errors)}):")
    for e in errors:
        print("  x", e)
else:
    print("ERRORS: none")
print()
if warnings:
    print(f"WARNINGS ({len(warnings)}):")
    for w in warnings:
        print("  !", w)
else:
    print("WARNINGS: none")
print()
print("RESULT:", "FAIL" if errors else "PASS")
sys.exit(1 if errors else 0)
