"""Hatch build hook: refresh ``_embedded_catalog.json`` before wheel assembly.

Without this, local ``uv build`` invocations can ship a stale embedded
catalog. CI runs ``scripts/embed_catalog.py`` explicitly via ``publish.yml``,
but local builds bypass that — someone building a private fork locally
and shipping the wheel out-of-band could distribute an obsolete embedded
catalog without realizing.

The hook delegates to ``scripts/embed_catalog.py`` via ``uv run`` so the
PEP 723 inline script metadata provides PyYAML in the (otherwise isolated)
build environment.

Set ``AGENT_SCAFFOLD_BUILD_SKIP_REFRESH=1`` to opt out — offline builds,
testing the embedded-fallback path, or producing a deliberately stale
wheel. Any failure (no ``uv`` on PATH, network error, non-zero exit)
also degrades gracefully: the hook prints a warning and the build
proceeds with the committed JSON.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class RefreshEmbeddedCatalog(BuildHookInterface):
    PLUGIN_NAME = "refresh-embedded-catalog"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        if self.target_name != "wheel":
            return

        if os.environ.get("AGENT_SCAFFOLD_BUILD_SKIP_REFRESH") == "1":
            print(
                "[refresh-embedded-catalog] AGENT_SCAFFOLD_BUILD_SKIP_REFRESH=1 "
                "— skipping refresh",
                file=sys.stderr,
            )
            return

        try:
            subprocess.run(
                ["uv", "run", "scripts/embed_catalog.py"],
                check=True,
                cwd=self.root,
            )
        except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
            print(
                f"[refresh-embedded-catalog] WARN: catalog refresh failed ({exc}); "
                "using committed _embedded_catalog.json",
                file=sys.stderr,
            )
