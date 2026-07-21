# Installation

The package is published on PyPI as [**`agent-scaffold-cli`**](https://pypi.org/project/agent-scaffold-cli/) and installs two equivalent binaries: `agent-scaffold` (long form) and `scaffold` (short, `claude`-style). Bare `scaffold` (no subcommand) drops you straight into the interactive REPL; everything else (`scaffold new`, `scaffold doctor`, `scaffold --help`, ...) mirrors the `agent-scaffold` subcommands.

## One-line install (recommended)

Installs the CLI, adds it to your PATH, and offers to store your Anthropic key — the way `claude`'s installer works:

```bash
curl -fsSL https://raw.githubusercontent.com/jagguvarma15/agent-scaffold/main/install.sh | sh
```

## Manual install

A plain `pip install` can't put the binaries on your PATH (wheels run no code at install time), so use `pipx` or `uv tool` and run their one-time PATH step:

```bash
pipx install agent-scaffold-cli && pipx ensurepath
# or
uv tool install agent-scaffold-cli && uv tool update-shell
# or, for one-off use (no install, no PATH change):
uvx --from agent-scaffold-cli scaffold --help
```

Either way, restart your shell afterward, then store your Anthropic key once with `scaffold auth login` (the one-line installer prompts for it during setup). `scaffold` won't start without a key — see [Credentials](../guides/credentials.md) for where keys live and how resolution works.

## Local development

```bash
git clone https://github.com/jagguvarma15/agent-scaffold
cd agent-scaffold
uv sync
make install-dev   # exposes `scaffold` + `agent-scaffold` on PATH (editable)
```
