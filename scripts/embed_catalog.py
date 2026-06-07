#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["PyYAML>=6.0"]
# ///
"""Fetch the live catalog and bake it into the wheel as JSON.

Run before building the wheel (manually for now; a hatch build hook can
automate it in a follow-up). The resulting ``src/agent_scaffold/_embedded_catalog.json``
ships with the package as the offline fallback for ``catalog.load_catalog``.

Usage:
    uv run scripts/embed_catalog.py
    uv run scripts/embed_catalog.py --url file:///path/catalog.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import yaml

DEFAULT_URL = "https://raw.githubusercontent.com/jagguvarma15/agent-deployments/main/catalog.yaml"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "src" / "agent_scaffold" / "_embedded_catalog.json"
TIMEOUT_SECONDS = 15.0


def fetch(url: str) -> str:
    if url.startswith(("file://", "/", "./")):
        path = url[7:] if url.startswith("file://") else url
        return Path(path).read_text(encoding="utf-8")
    req = urllib.request.Request(url, headers={"Accept": "text/yaml, text/plain, */*"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
        return resp.read().decode("utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bake the live catalog into _embedded_catalog.json")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"Catalog URL. Default: {DEFAULT_URL}")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help=f"Output path. Default: {DEFAULT_OUT}")
    args = parser.parse_args(argv)

    body = fetch(args.url)
    data = yaml.safe_load(body)
    if not isinstance(data, dict):
        print(f"error: fetched body did not parse as a YAML mapping", file=sys.stderr)
        return 2

    # JSON-serialize compactly — the embedded fallback is parsed only on the
    # cold offline path, so we optimize for wheel size, not readability.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, separators=(",", ":"), sort_keys=False), encoding="utf-8")

    size_kb = out_path.stat().st_size / 1024
    print(f"Wrote {out_path} ({size_kb:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
