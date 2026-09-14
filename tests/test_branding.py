"""Tests for the banner rendering in ``agent_scaffold.branding``.

The figlet logo has a natural width (~65 cols for the SCAFFOLD stack);
terminals narrower than that used to wrap the glyph art mid-glyph. The
banner now clamps its panel to the console width and swaps the figlet for
a plain styled title when it cannot fit.
"""

from __future__ import annotations

from rich.console import Console

from agent_scaffold.branding import print_banner, render_logo_rows


def _render(width: int, body: list[str]) -> str:
    console = Console(width=width, record=True, color_system=None, force_terminal=False)
    print_banner(console, body, leading_blank_lines=0)
    return console.export_text()


def test_narrow_terminal_falls_back_to_a_plain_title() -> None:
    out = _render(50, ["deployments: cached", "blueprints: skipped"])
    assert "Agent Scaffold" in out
    assert "█" not in out
    assert all(len(line) <= 50 for line in out.splitlines())


def test_wide_terminal_renders_the_figlet() -> None:
    out = _render(120, ["deployments: cached"])
    assert "█" in out
    assert all(len(line) <= 120 for line in out.splitlines())


def test_panel_clamps_to_the_console_width() -> None:
    long_line = "deployments: " + "x" * 200
    out = _render(60, [long_line])
    assert all(len(line) <= 60 for line in out.splitlines())


def test_render_logo_rows_is_negative_safe() -> None:
    rows = render_logo_rows(target_width=10)
    assert rows
    # Narrower-than-natural targets emit unpadded lines rather than raising.
    assert all(row.plain.rstrip() for row in rows)
