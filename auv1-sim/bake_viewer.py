"""Bake a run (and optionally the CAD STL) INTO a copy of the viewer, so one
HTML file can be sent to anyone - no Python, no loading files by hand.

    py bake_viewer.py runs/square_tuned.json
    py bake_viewer.py runs/square_tuned.json "C:\\path\\AUV1P General Assembly.STL"

Writes runs/<run>_viewer.html. Open it in any browser.
"""

import base64
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    run_path = Path(sys.argv[1])
    html = (HERE / "viewer" / "viewer.html").read_text(encoding="utf-8")
    run_json = json.dumps(json.load(open(run_path)), separators=(",", ":"))
    html = html.replace("/*RUN_JSON_HERE*/", run_json.replace("</", "<\\/"))
    if len(sys.argv) > 2:
        stl = Path(sys.argv[2]).read_bytes()
        html = html.replace("/*STL_BASE64_HERE*/", base64.b64encode(stl).decode("ascii"))
    out = run_path.with_name(run_path.stem + "_viewer.html")
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
