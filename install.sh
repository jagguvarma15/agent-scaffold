#!/bin/sh
# agent-scaffold installer.
#
# Installs the `scaffold` (and `agent-scaffold`) CLI and puts it on your PATH the
# way `claude` does — one command, no manual PATH editing:
#
#   curl -fsSL https://raw.githubusercontent.com/jagguvarma15/agent-scaffold/main/install.sh | sh
#
# How it works: a `pip install` of a wheel cannot edit your shell PATH (wheels
# run no code at install time), so the PATH step has to live here. We install
# into an isolated environment with uv (preferred) or pipx, then make sure the
# tools bin dir is on PATH — via that tool's own setup command, with a profile
# append as a fallback. Re-running upgrades in place.
#
# Env overrides:
#   AGENT_SCAFFOLD_SPEC   package spec to install
#                         (default: "agent-scaffold-cli>=0.3"; point it at a local
#                          wheel to test, e.g. AGENT_SCAFFOLD_SPEC=./dist/x.whl)
set -eu

# Pin >=0.3 so the installer never pulls a pre-0.3 build: those fail to install
# on Python 3.13+ (exact pydantic pin, no prebuilt wheel) and have no `scaffold`
# command. Override with AGENT_SCAFFOLD_SPEC for local-wheel testing.
SPEC="${AGENT_SCAFFOLD_SPEC:-agent-scaffold-cli>=0.3}"
BIN_DIR="${HOME}/.local/bin"

if [ -t 1 ]; then
  B="$(printf '\033[1m')"; G="$(printf '\033[32m')"; Y="$(printf '\033[33m')"; R="$(printf '\033[0m')"
else
  B=""; G=""; Y=""; R=""
fi
say()  { printf '%s%s%s\n' "$B" "$*" "$R"; }
note() { printf '%s\n' "$*"; }
warn() { printf '%swarning:%s %s\n' "$Y" "$R" "$*" >&2; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

# Can we prompt the user? Under `curl | sh` stdin is the pipe, not the keyboard,
# so we talk to the controlling terminal via /dev/tty (which getpass also uses).
# No terminal (CI, no controlling tty) -> skip interactive steps.
can_prompt() { { true </dev/tty; } >/dev/null 2>&1; }

# Is BIN_DIR active in this process, or already written into a shell startup file?
path_is_configured() {
  case ":${PATH}:" in
    *":${BIN_DIR}:"*) return 0 ;;
  esac
  for f in "${ZDOTDIR:-$HOME}/.zshenv" "$HOME/.zshenv" "$HOME/.zprofile" "$HOME/.zshrc" \
           "$HOME/.bash_profile" "$HOME/.bashrc" "$HOME/.profile" \
           "$HOME/.config/fish/config.fish"; do
    if [ -f "$f" ] && grep -qF "$BIN_DIR" "$f" 2>/dev/null; then
      return 0
    fi
  done
  return 1
}

# Fallback: append BIN_DIR to the login shell's startup file (idempotent).
append_path_to_profile() {
  shell_name="$(basename "${SHELL:-/bin/sh}")"
  case "$shell_name" in
    zsh)  profile="${ZDOTDIR:-$HOME}/.zshenv" ; line="export PATH=\"${BIN_DIR}:\$PATH\"" ;;
    bash) [ "$(uname -s)" = "Darwin" ] && profile="$HOME/.bash_profile" || profile="$HOME/.bashrc"
          line="export PATH=\"${BIN_DIR}:\$PATH\"" ;;
    fish) profile="$HOME/.config/fish/config.fish" ; line="fish_add_path ${BIN_DIR}" ;;
    *)    profile="$HOME/.profile" ; line="export PATH=\"${BIN_DIR}:\$PATH\"" ;;
  esac
  mkdir -p "$(dirname "$profile")" 2>/dev/null || true
  printf '\n# agent-scaffold\n%s\n' "$line" >> "$profile" &&
    note "Added ${BIN_DIR} to PATH in ${profile}"
}

# Run the tool's own PATH setup, then verify; fall back to a profile append.
# (uv/pipx return non-zero when the config is already written but not yet live,
# so we verify the end state instead of trusting their exit code.)
ensure_on_path() {
  case "$1" in
    uv)   uv tool update-shell >/dev/null 2>&1 || true ;;
    pipx) pipx ensurepath      >/dev/null 2>&1 || true ;;
  esac
  path_is_configured && return 0
  append_path_to_profile || true
  path_is_configured
}

case "$(uname -s)" in
  Darwin | Linux) ;;
  *) die "unsupported OS '$(uname -s)'. On Windows run:  pipx install \"$SPEC\"" ;;
esac

if have uv; then
  say "Installing ${SPEC} with uv…"
  uv tool install --force -q "$SPEC"
  method=uv
elif have pipx; then
  say "Installing ${SPEC} with pipx…"
  pipx install --force "$SPEC"
  method=pipx
else
  say "Neither uv nor pipx found — installing uv first…"
  have curl || die "curl is required to bootstrap uv. Install uv or pipx, then re-run."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # Make the freshly-installed uv usable within this same script run.
  [ -f "${HOME}/.local/bin/env" ] && . "${HOME}/.local/bin/env" 2>/dev/null || true
  if ! have uv; then PATH="${BIN_DIR}:${PATH}"; export PATH; fi
  have uv || die "uv installed but isn't on PATH yet; open a new shell and re-run this script."
  say "Installing ${SPEC} with uv…"
  uv tool install --force -q "$SPEC"
  method=uv
fi

if ensure_on_path "$method"; then
  path_ok=1
else
  path_ok=0
fi

# Verify the binary. PATH edits only take effect in new shells, so look in the
# bin dir directly rather than relying on `command -v` in this process.
if [ -x "${BIN_DIR}/scaffold" ]; then
  say "${G}✓${R} $("${BIN_DIR}/scaffold" --version 2>/dev/null || echo scaffold) → ${BIN_DIR}/scaffold"
elif have scaffold; then
  say "${G}✓${R} $(scaffold --version 2>/dev/null || echo scaffold)"
else
  warn "installed, but no 'scaffold' binary found at ${BIN_DIR}. Check 'uv tool list' / 'pipx list'."
fi

if [ "$path_ok" -ne 1 ]; then
  note ""
  warn "couldn't add ${BIN_DIR} to your PATH automatically."
  note "Add this line to your shell profile, then restart your terminal:"
  note "    export PATH=\"${BIN_DIR}:\$PATH\""
fi

# --- Anthropic API key (scaffold won't start without one) ---
SCAFFOLD_BIN="${BIN_DIR}/scaffold"
[ -x "$SCAFFOLD_BIN" ] || SCAFFOLD_BIN=scaffold
key_ok=0
if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
  note ""
  say "${G}✓${R} ANTHROPIC_API_KEY found in your environment — scaffold will use it."
  key_ok=1
elif can_prompt; then
  note ""
  say "Set up your Anthropic API key — scaffold won't start without one."
  note "Paste your key to store it in your OS keychain (hidden input), or press Enter to skip."
  # auth login --no-browser reads the key via getpass (/dev/tty), validates it
  # against the API, and stores it in the keyring. </dev/tty so it reads your
  # keyboard even though this script arrived over a pipe.
  if "$SCAFFOLD_BIN" auth login --no-browser </dev/tty; then
    key_ok=1
  fi
fi

# --- Final summary ---
note ""
say "Done. To start:"
note "  • Restart your terminal (or run:  exec \$SHELL )  so PATH updates."
[ "$key_ok" -eq 1 ] || note "  • Add your Anthropic key:  scaffold auth login"
note "  • Open the REPL:           scaffold"
