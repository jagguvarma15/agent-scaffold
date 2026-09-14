"""Shared visual branding: the orange→red ``Agent Scaffold`` figlet logo.

Used by the top-level CLI banner (``agent-scaffold`` with no args) and the
REPL welcome screen (``agent-scaffold scaffold``). Factored here so the
two surfaces stay visually consistent and only one place owns the
gradient + alignment math.
"""

from __future__ import annotations

import re

from rich.align import Align
from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text

# Orange (top) → red (bottom). RGB triples interpolated row-by-row across
# the figlet output so the logo reads like a flame.
LOGO_GRADIENT_START: tuple[int, int, int] = (255, 179, 71)  # light orange
LOGO_GRADIENT_END: tuple[int, int, int] = (139, 0, 0)  # dark red

# Used for the banner panel border + accent text. Same tomato Rich knows
# as ``#FF6347`` — picked once so every surface matches.
BANNER_BORDER_STYLE = "#FF6347"

# Semantic style constants used by wizard prompts, the post-gen report, and
# any new surface. Existing one-off markup (``[green]✓[/]`` etc.) is left
# untouched — only new code should pull from here so the diff stays scoped.
ACCENT = "#FFA500"  # orange — primary accent on headers + section labels
ACCENT_DIM = "#FFB347"  # paler orange — subheaders + delta deltas
MUTED = "dim"
OK = "green"
WARN = "yellow"
ERR = "red"
PANEL_BORDER_STYLE = BANNER_BORDER_STYLE  # alias so new code reads naturally

# Rich markup pattern for measuring visible width (so the panel and logo
# share a horizontal axis regardless of inline color tags in the body).
_RICH_TAG_RE = re.compile(r"\[/?[^\]]*\]")


def interpolate_color(
    start: tuple[int, int, int], end: tuple[int, int, int], step: int, total: int
) -> str:
    """Return a Rich ``rgb(r,g,b)`` color string at ``step/total`` along the gradient."""
    if total <= 1:
        r, g, b = start
    else:
        ratio = step / (total - 1)
        r, g, b = (int(s + (e - s) * ratio) for s, e in zip(start, end, strict=True))
    return f"rgb({r},{g},{b})"


def visible_width(markup: str) -> int:
    """Length of ``markup`` with Rich tags stripped — what the user actually sees."""
    return len(_RICH_TAG_RE.sub("", markup))


def render_logo_rows(target_width: int) -> list[Text]:
    """Render 'Agent Scaffold' as gradient block letters padded to ``target_width``.

    pyfiglet wraps the text onto two stacks ("AGENT" above "SCAFFOLD") at
    the default render width. Each stack is internally width-uniform but
    the two stacks differ (AGENT=44 cols, SCAFFOLD=65), so we pad every
    row to ``target_width`` and the whole block centers as one unit. Each
    row is its own ``Text`` because collapsing into a single Text with
    embedded newlines triggers Rich's leading-whitespace stripping on the
    first visual line.
    """
    # Imported lazily: pyfiglet is a whole figlet engine paid for one banner.
    # Keeping it out of the module top keeps every non-banner invocation
    # (--help, doctor, validate) from importing it at startup.
    from pyfiglet import Figlet

    raw = Figlet(font="ansi_shadow").renderText("Agent Scaffold")
    lines = [line for line in raw.splitlines() if line.strip()]
    rows: list[Text] = []
    for i, line in enumerate(lines):
        color = interpolate_color(LOGO_GRADIENT_START, LOGO_GRADIENT_END, i, len(lines))
        # Negative-safe: a target narrower than the glyph line pads nothing
        # rather than raising or mangling — callers should still avoid it
        # (print_banner falls back to a plain title below the threshold).
        pad_left = max(0, (target_width - len(line)) // 2)
        pad_right = max(0, target_width - len(line) - pad_left)
        rows.append(Text(" " * pad_left + line + " " * pad_right, style=f"bold {color}"))
    return rows


def _figlet_natural_width() -> int:
    """Widest line of the rendered figlet (the SCAFFOLD stack, ~65 cols)."""
    from pyfiglet import Figlet

    raw = Figlet(font="ansi_shadow").renderText("Agent Scaffold")
    return max((len(line) for line in raw.splitlines() if line.strip()), default=0)


def print_banner(console: Console, body_lines: list[str], *, leading_blank_lines: int = 2) -> None:
    """Render the logo + an info panel below, both sharing the same axis.

    ``body_lines`` may contain Rich markup; the panel auto-sizes to its
    widest visible line (clamped to the terminal), and ``render_logo_rows``
    pads the logo block to the same width so the figlet and panel are
    visually aligned. On terminals too narrow for the glyph art, a plain
    styled title takes its place — wrapped figlet rows read as garbage.

    ``leading_blank_lines`` gives the logo room to breathe from whatever
    was printed before (a shell prompt, a previous command's output).
    """
    # Panel adds 4 cols of chrome (2 border + 2 internal padding).
    panel_width = min(max(visible_width(line) for line in body_lines) + 4, console.width)
    logo: list[Text]
    if console.width >= _figlet_natural_width():
        logo = render_logo_rows(target_width=panel_width)
    else:
        logo = [Text("Agent Scaffold", style=f"bold {BANNER_BORDER_STYLE}")]
    panel = Panel(
        "\n".join(body_lines),
        width=panel_width,
        border_style=BANNER_BORDER_STYLE,
    )
    if leading_blank_lines > 0:
        console.print("\n" * leading_blank_lines, end="")
    console.print(Align.center(Group(*logo, panel)))
