"""Cloud deploy provider plugins.

Each module under this package exposes a ``deploy(project_dir, dry_run, yes)``
function that returns a :class:`DeployResult`. The CLI dispatcher
(``cmd_deploy``) reads the resolved capability stack from the project's
manifest, picks the right plugin by capability ``target`` name, and shells
out to the provider CLI when the user confirms.

Default is dry-run: the plugin prints the command it would invoke + a
dashboard URL, and exits without touching the cloud. ``--yes`` opts in to
the actual deploy.

Plugins are intentionally thin — they never construct auth flows
themselves. The user runs ``vercel login`` / ``fly auth login`` /
``railway login`` once, and the provider CLI handles token storage.
"""

from __future__ import annotations

from agent_scaffold.deploy._common import DeployResult, DeployTarget

__all__ = [
    "DEPLOY_PLUGINS",
    "DeployResult",
    "DeployTarget",
    "get_plugin",
]


def _import_plugins() -> dict[str, DeployTarget]:
    """Lazy plugin registry — avoids importing provider modules at package load."""
    from agent_scaffold.deploy import fly, railway, vercel

    return {
        "vercel": vercel,
        "railway": railway,
        "fly": fly,
    }


DEPLOY_PLUGINS: dict[str, DeployTarget] | None = None


def get_plugin(target: str) -> DeployTarget:
    """Return the deploy plugin module for ``target``.

    Raises ``KeyError`` if no plugin is registered.
    """
    global DEPLOY_PLUGINS
    if DEPLOY_PLUGINS is None:
        DEPLOY_PLUGINS = _import_plugins()
    return DEPLOY_PLUGINS[target]
