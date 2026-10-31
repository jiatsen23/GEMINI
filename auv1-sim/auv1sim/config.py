"""Load the two JSON config files (pool + vehicle).

Everything the simulator knows about the world lives in config/*.json so
that changing the pool size or a thruster's max thrust never means
editing Python.
"""

import json
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def load(name: str, path: str | None = None) -> dict:
    """Return the parsed JSON for 'pool' or 'vehicle'.

    Pass `path` to use a different file (e.g. a second pool). Keys that
    start with an underscore are comments and are stripped so the rest of
    the code never trips over them.
    """
    file = Path(path) if path else CONFIG_DIR / f"{name}.json"
    with open(file) as f:
        raw = json.load(f)
    return _strip_comments(raw)


def _strip_comments(node):
    if isinstance(node, dict):
        return {k: _strip_comments(v) for k, v in node.items()
                if not k.startswith("_")}
    return node
