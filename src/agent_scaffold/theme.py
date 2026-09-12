"""The single visual vocabulary for every REPL and shared-render surface.

Colors come from :mod:`agent_scaffold.branding` (which stays the one place
the palette is defined); everything else — border hierarchy, status glyphs,
the empty-value marker, row/column alignment, message shapes, and the
questionary / prompt_toolkit styles — is defined here and only here. A
surface that wants a checkmark, a panel border, or an aligned label column
imports it from this module instead of hand-rolling markup, so the whole
tool reads as one product rather than a collage of one-off strings.

Border hierarchy: :data:`BORDER_PRIMARY` (the brand tomato) is reserved for
the startup banner and the active wizard step — the two "you are here"
moments. Every informational panel uses :data:`BORDER_INFO` with an
accent-colored title via :func:`info_title`, so status colors (green /
yellow / red) keep their semantic borders (welcome's green "Ready", the
diff preview's yellow) without competing with a loud frame on every box.

questionary and prompt_toolkit are imported lazily inside the style
helpers: this module is imported by shared render code that must not drag
interactive-prompt dependencies into non-interactive paths.
"""

from __future__ import annotations

from typing import Any

from rich.markup import escape

from agent_scaffold.branding import (
    ACCENT,
    ACCENT_DIM,
    ERR,
    MUTED,
    OK,
    PANEL_BORDER_STYLE,
    WARN,
)

__all__ = [
    "ACCENT",
    "ACCENT_DIM",
    "BORDER_INFO",
    "BORDER_PRIMARY",
    "EMPTY",
    "ERR",
    "GLYPH_BACK",
    "GLYPH_FAIL",
    "GLYPH_LIVE",
    "GLYPH_OFF",
    "GLYPH_OK",
    "GLYPH_PAUSE",
    "GLYPH_PROMPT",
    "GLYPH_WARN",
    "MAX_WIDTH",
    "MUTED",
    "OK",
    "WARN",
    "checkbox_kwargs",
    "col",
    "confirm_line",
    "empty",
    "error_line",
    "fail_glyph",
    "hint_line",
    "info_title",
    "ok_glyph",
    "off_glyph",
    "pt_style",
    "q_style",
    "row",
    "select_kwargs",
    "soft_gate_line",
    "text_kwargs",
    "truncate",
    "warn_glyph",
]

# ---------------------------------------------------------------------------
# Borders + titles
# ---------------------------------------------------------------------------

BORDER_PRIMARY = PANEL_BORDER_STYLE
"""Brand tomato — the startup banner and the active wizard step only."""

BORDER_INFO = "dim"
"""Every informational panel (session, plan, report, preflight, next steps)."""


def info_title(label: str) -> str:
    """Accent-colored title for an informational panel."""
    return f"[{ACCENT}]{label}[/]"


# ---------------------------------------------------------------------------
# Glyphs — the one status vocabulary. No emojis, no decorative extras.
# ---------------------------------------------------------------------------

GLYPH_OK = "✓"
GLYPH_FAIL = "✗"
GLYPH_WARN = "⚠"
GLYPH_OFF = "○"  # off / skipped / not running
GLYPH_LIVE = "●"  # running / live
GLYPH_PROMPT = "›"
GLYPH_BACK = "‹"
GLYPH_PAUSE = "⏸"


def ok_glyph() -> str:
    return f"[{OK}]{GLYPH_OK}[/]"


def fail_glyph() -> str:
    return f"[{ERR}]{GLYPH_FAIL}[/]"


def warn_glyph() -> str:
    return f"[{WARN}]{GLYPH_WARN}[/]"


def off_glyph() -> str:
    return f"[{MUTED}]{GLYPH_OFF}[/]"


# ---------------------------------------------------------------------------
# Empty value
# ---------------------------------------------------------------------------

EMPTY = "–"
"""The one unset/none marker (en dash — single-cell wide, keeps columns true)."""


def empty() -> str:
    """The unset marker, dimmed, ready for markup contexts."""
    return f"[{MUTED}]{EMPTY}[/]"


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

MAX_WIDTH = 80
"""Layout budget: everything must stay readable at this width."""


def truncate(text: str, limit: int) -> str:
    """Collapse to one line and cap at ``limit`` visible chars with an ellipsis."""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: max(limit - 1, 0)] + "…"


def row(label: str, value: str, *, width: int = 12) -> str:
    """One aligned ``label value`` panel row (bold label, fixed column).

    Pads the plain label before wrapping it in markup, so the column can't
    drift and an over-long label degrades to a single separating space
    instead of losing it entirely.
    """
    return f"[bold]{label.ljust(width)}[/] {value}"


def col(raw: str, width: int) -> str:
    """Escape ``raw`` for markup and pad by its *visible* width.

    ``escape()`` grows the string (``[`` → ``\\[``), so padding must be
    computed from the unescaped length or names containing brackets eat
    their own column.
    """
    return escape(raw) + " " * max(0, width - len(raw))


# ---------------------------------------------------------------------------
# Message shapes
# ---------------------------------------------------------------------------


def error_line(msg: str) -> str:
    """The one error shape: red cross, then the escaped message."""
    return f"[{ERR}]{GLYPH_FAIL}[/] {escape(msg)}"


def confirm_line(summary: str) -> str:
    """The one success/confirmation shape (case-preserving)."""
    return f"[{OK}]{GLYPH_OK}[/] {escape(summary)}"


def soft_gate_line(need: str, fix: str) -> str:
    """The one "not ready yet" shape: what's missing, then how to fix it."""
    return f"[{WARN}]{escape(need)}[/] — {fix}"


def hint_line(msg: str) -> str:
    """Dim guidance line (may embed markup — not escaped)."""
    return f"[{MUTED}]{msg}[/]"


# ---------------------------------------------------------------------------
# questionary / prompt_toolkit styles (lazy imports)
# ---------------------------------------------------------------------------


def q_style() -> Any:
    """The questionary style: brand pointer/qmark, accent highlight, dim hints."""
    import questionary

    return questionary.Style(
        [
            ("qmark", f"fg:{BORDER_PRIMARY} bold"),
            ("question", "bold"),
            ("answer", f"fg:{ACCENT} bold"),
            ("pointer", f"fg:{BORDER_PRIMARY} bold"),
            ("highlighted", f"fg:{ACCENT}"),
            ("selected", f"fg:{ACCENT}"),
            ("separator", "fg:#6C6C6C"),
            ("instruction", "fg:#767676"),
            ("disabled", "fg:#767676 italic"),
        ]
    )


def select_kwargs() -> dict[str, Any]:
    """Shared kwargs for every ``questionary.select``.

    ``instruction=" "`` suppresses questionary's auto "(Use arrow keys)" so
    prompts carry exactly one navigation hint (ours, when needed).
    """
    return {"qmark": GLYPH_PROMPT, "style": q_style(), "instruction": " "}


def checkbox_kwargs() -> dict[str, Any]:
    """Shared kwargs for every ``questionary.checkbox`` (space toggles hint only)."""
    return {
        "qmark": GLYPH_PROMPT,
        "style": q_style(),
        "instruction": "(space toggles · enter continues)",
    }


def text_kwargs() -> dict[str, Any]:
    """Shared kwargs for every ``questionary.text``."""
    return {"qmark": GLYPH_PROMPT, "style": q_style()}


def pt_style() -> Any:
    """prompt_toolkit style for the shell prompt + bottom toolbar."""
    from prompt_toolkit.styles import Style

    return Style.from_dict(
        {
            "bottom-toolbar": "noreverse fg:#767676 bg:default",
            "prompt": f"{ACCENT} bold",
        }
    )
