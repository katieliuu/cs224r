"""Add repo root and this package dir to sys.path for direct script execution."""
from pathlib import Path
import sys

_THIS_DIR = Path(__file__).resolve().parent  # .../cs224r/cs224r/

# Add this directory so sibling modules (data, features, env, …) resolve as bare imports.
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# Walk up to find repo root (contains chem/ and core/).
for _candidate in [_THIS_DIR.parent, *_THIS_DIR.parent.parents]:
    if (_candidate / "chem").is_dir() and (_candidate / "core").is_dir():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break

for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
