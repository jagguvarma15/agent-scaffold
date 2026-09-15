"""Tests for the shared theme module.

The theme is the single visual vocabulary: glyphs, the empty marker, row
and column alignment, and message shapes. These tests freeze the glyph set
(guarding against emoji or multi-cell regressions) and pin the alignment
math that three modules previously each got subtly wrong.
"""

from __future__ import annotations

from agent_scaffold import theme


def test_glyph_set_is_frozen() -> None:
    assert theme.GLYPH_OK == "✓"
    assert theme.GLYPH_FAIL == "✗"
    assert theme.GLYPH_WARN == "⚠"
    assert theme.GLYPH_OFF == "○"
    assert theme.GLYPH_LIVE == "●"
    assert theme.GLYPH_PROMPT == "›"
    assert theme.GLYPH_BACK == "‹"
    assert theme.GLYPH_PAUSE == "⏸"


def test_glyphs_are_single_characters() -> None:
    """Single code points keep columns true; emojis and ZWJ sequences do not."""
    for name in dir(theme):
        if name.startswith("GLYPH_"):
            assert len(getattr(theme, name)) == 1, name


def test_empty_marker() -> None:
    assert theme.EMPTY == "–"
    assert theme.empty() == "[dim]–[/]"


def test_row_pads_the_plain_label() -> None:
    assert theme.row("Recipe", "x", width=8) == "[bold]Recipe  [/] x"


def test_row_overlong_label_keeps_the_separator_space() -> None:
    """The old subtraction-based padding dropped the separator entirely for a
    label at or beyond the column width."""
    assert theme.row("Service readiness", "v", width=13) == "[bold]Service readiness[/] v"


def test_col_pads_by_visible_width() -> None:
    """escape() grows bracketed names; padding must use the raw length."""
    plain = theme.col("api", 6)
    bracketed = theme.col("a[b]", 6)
    assert plain == "api   "
    assert bracketed.startswith("a\\[b]")
    # Both occupy six visible cells once markup escaping is unwound.
    assert len(plain) == 6
    assert len(bracketed) - bracketed.count("\\") == 6


def test_truncate_collapses_and_caps() -> None:
    assert theme.truncate("one\ntwo   three", 100) == "one two three"
    assert theme.truncate("abcdefgh", 5) == "abcd…"
    assert theme.truncate("short", 5) == "short"


def test_message_shapes() -> None:
    assert theme.error_line("boom [x]") == "[red]✗[/] boom \\[x]"
    assert theme.confirm_line("recipe → demo") == "[green]✓[/] recipe → demo"
    assert theme.soft_gate_line("Plan needs: recipe", "run /recipe.") == (
        "[yellow]Plan needs: recipe[/] — run /recipe."
    )
    assert theme.hint_line("try /help") == "[dim]try /help[/]"


def test_questionary_kwargs_share_the_prompt_glyph() -> None:
    select = theme.select_kwargs()
    checkbox = theme.checkbox_kwargs()
    text = theme.text_kwargs()
    assert select["qmark"] == checkbox["qmark"] == text["qmark"] == theme.GLYPH_PROMPT
    # The select instruction is a single space: it suppresses questionary's
    # auto "(Use arrow keys)" so prompts carry exactly one navigation hint.
    assert select["instruction"] == " "
    assert "space toggles" in checkbox["instruction"]


def test_info_title_wraps_the_accent() -> None:
    assert theme.info_title("Session") == f"[{theme.ACCENT}]Session[/]"


def test_confirm_kwargs_share_the_prompt_glyph() -> None:
    kwargs = theme.confirm_kwargs()
    assert kwargs["qmark"] == theme.GLYPH_PROMPT
    assert kwargs["style"] is not None
