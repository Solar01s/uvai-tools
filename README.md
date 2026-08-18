# tools.py

A lightweight, model-agnostic tool-calling layer for text-generating AI models.
The AI writes plain text with embedded tags (`<calculate 2+2>`, `<date>`,
`<search ...>`, etc.), and this module detects, executes, and resolves them —
no provider-specific function-calling API required.

## Install

```bash
pip install uvai-tools
```

```python
from tools import connect
```

## Quick start

```python
from tools import connect

def my_generate(history):
    # call your model here, return its text response
    ...

# One-shot string in, string out
result = connect("Compute <calculate 3*3> for me", my_generate)

# Full message history in, updated history out
history = [{"role": "assistant", "content": "Let me check <date>"}]
result = connect(history, my_generate)

# Streaming: yields text chunks, resolving tags inline as they complete
for piece in connect(lambda: my_stream_call(prompt)):
    print(piece, end="", flush=True)
```

`connect` dispatches based on the type of `input_data`:

| Input type | Behavior |
|---|---|
| `str` | One-shot request. Returns the final response string. |
| `list` | Message history. Returns the updated history list. |
| generator / callable | Streaming. Returns a chunk generator. |

## Tags

| Tag | Function | What it does | Example |
|---|---|---|---|
| `<calculate expr>` | `calculator` | Evaluates a Python expression via `eval()` | `<calculate 15*4-2>` → `58` |
| `<date>` | `get_date` | Current date (`dd.mm.yyyy`) | `<date>` → `18.08.2026` |
| `<date +1y-2M+3d>` | `get_date` | Date shifted by years/months/days | `<date +1y>` → `18.08.2027` |
| `<date y>` / `<date M>` / `<date d>` | `get_date` | Just the year / month / day | `<date y>` → `2026` |
| `<time>` | `get_time` | Current time (`HH:MM:SS`) | `<time>` → `11:59:11` |
| `<time +2h30m>` | `get_time` | Time shifted by hours/minutes | `<time +1h>` → `12:59:11` |
| `<time h>` / `<time m>` / `<time s>` | `get_time` | Just the hour / minute / second | `<time h>` → `11` |
| `<search query>` | `search_web` | Web search (DuckDuckGo) + a short scraped summary of the top 3 results | `<search Python 3.13 release>` |
| `<fix_tags>` | `fix_html_tags` | Auto-closes and reorders unbalanced HTML tags written before this point in the text | `<p>Hello <b>world<fix_tags>` → `<p>Hello <b>world</b></p>` |
| `<layout text>` | `change_layout` | Switches text typed in the wrong keyboard layout, EN⇄RU | `<layout ghbdtn>` → `привет` |
| `<read path>` | `read_file` | Reads a file inside the sandboxed `Workspace/` directory | `<read notes/hello.txt>` |
| `<mkdir path>` | `make_directory` | Creates a directory (with parents) inside `Workspace/` | `<mkdir projects/app>` |
| `<dir>` / `<dir path>` | `get_available_files` | Lists files/folders in `Workspace/` root or a subfolder | `<dir>` |
| `<mkfile path content>` | `create_file` | Creates a file with the given text content inside `Workspace/` | `<mkfile notes.txt Hello!>` |
| `<del path>` | `delete_path` | Deletes a file, or recursively deletes a directory, inside `Workspace/` | `<del notes.txt>` |

Every tag can be used standalone or mixed freely into normal text — `use_all()`
resolves every tag it finds in a string:

```python
from tools import use_all

use_all("2+2 is <calculate 2+2>, today is <date>")
# -> "2+2 is 4, today is 18.08.2026"
```

## The `Workspace/` sandbox

`<read>`, `<mkdir>`, `<dir>`, `<mkfile>`, and `<del>` are all confined to a
`Workspace/` directory, created automatically relative to the current working
directory the first time `tools.py` is imported. Any path that would resolve
outside of `Workspace/` (e.g. `<read ../../etc/passwd>`) is rejected and
returns an access-denied message instead of touching the filesystem.

## Optional dependencies

`<search>` needs extra packages that aren't required for the rest of the
module:

```bash
pip install "uvai-tools[search]"
```

This installs `ddgs` (or the older `duckduckgo_search`) for the search itself,
and `beautifulsoup4` for scraping short summaries from the result pages.

## Customizing `connect` / `system` for your model

By default `connect`/`system` expect `{"role": ..., "content": ...}` messages
and a `generate_function(history) -> str` signature. Both are overridable:

```python
result = tools.connect(
    history,
    my_generate,
    system_role="tool",           # role used for tag-result messages
    assistant_role="bot",         # role used for the model's own messages
    get_role=lambda m: m["who"],
    get_content=lambda m: m["text"],
    make_message=lambda role, content: {"who": role, "text": content},
    call_generate=lambda fn, hist: fn(messages=hist, temperature=0.7),
)
```

For streaming with a generation function that needs arguments:

```python
tools.system_stream(
    lambda: my_stream_call(prompt, temperature=0.3),
    call_generator=lambda f: f(),
)
```

## Known limitations

- **`calculator` uses `eval()`** on the raw tag content. Do not expose it to
  untrusted input — a malicious `<calculate ...>` payload can execute
  arbitrary Python.
- **No nested tags in general.** `use_all()` processes tags left to right on
  the same string; a tag written inside another tag's parameters is not
  guaranteed to parse correctly, because the outer tag's regex stops at the
  first `>` it finds — which may belong to the inner tag. `search_web` is the
  one exception: it explicitly re-runs `use_all()` on the extracted query
  text, so tags fully contained inside a `<search ...>` query (not straddling
  its closing `>`) do get resolved.
- **Tag handlers are matched by fixed position** in `tools_patterns`
  (e.g. `list(tools_patterns.keys())[6]` for `read_file`). Reordering,
  inserting, or removing entries in `tools_patterns` will silently break
  which regex a handler uses.
- **No automatic multi-step planning or memory beyond the message history.**
  This module resolves tags reactively, one at a time, as the model writes
  them — it is a tool-calling primitive, not a planning or agent framework.
