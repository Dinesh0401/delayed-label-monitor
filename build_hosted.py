"""Build a self-contained Cloudflare Worker bundle using the same model and UI as Python.

Inlines the trained logistic parameters, HTML dashboards, and research benchmark artifacts
into an ES module bundle suitable for Cloudflare Workers / Wrangler / Miniflare runtimes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def build_worker_bundle(out_dir: Path | str) -> Path:
    """Compile the standalone Cloudflare Worker distribution artifacts.

    Args:
        out_dir: Destination directory path for compiled worker bundle.

    Returns:
        Path to the primary generated entrypoint file.
    """
    root = Path(__file__).parent
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    worker_src = (root / "hosted/worker.mjs").read_text(encoding="utf-8")

    artifacts = [
        ("MODEL", "serving-model.json"),
        ("HTML", "web/index.html"),
        ("STUDY", "delivery-results/report.html"),
        ("RESEARCH", "robust-detection-results.json"),
        ("STUDY2", "research/report.html"),
    ]

    for name, rel_path in artifacts:
        file_path = root / rel_path
        content = file_path.read_text(encoding="utf-8")
        inlined = content if rel_path.endswith(".json") else json.dumps(content)
        worker_src += f"\nconst {name} = {inlined};\n"

    worker_src += "\nexport default createHandler(MODEL, HTML, STUDY, RESEARCH, STUDY2);\n"

    # Write both .js and .mjs extensions alongside package.json {"type": "module"}
    # to guarantee compatibility with all Node.js and Cloudflare Workers module loaders
    js_entry = out / "index.js"
    mjs_entry = out / "index.mjs"
    pkg_file = out / "package.json"

    js_entry.write_text(worker_src, encoding="utf-8")
    mjs_entry.write_text(worker_src, encoding="utf-8")
    pkg_file.write_text(
        json.dumps({"name": "delayed-label-worker", "version": "1.0.0", "type": "module"}, indent=2)
        + "\n",
        encoding="utf-8",
    )

    print(f"Built Cloudflare Worker bundle at {js_entry} and {mjs_entry}")
    return js_entry


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "hosted/dist"
    build_worker_bundle(target)
