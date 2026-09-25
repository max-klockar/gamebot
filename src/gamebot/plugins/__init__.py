from __future__ import annotations

from importlib.metadata import entry_points

from gamebot.plugins.protocol import GamePlugin, GameSession, IdentifyContext, RuntimeHandles

ENTRY_POINT_GROUP = "gamebot.plugins"

__all__ = [
    "ENTRY_POINT_GROUP",
    "GamePlugin",
    "GameSession",
    "IdentifyContext",
    "RuntimeHandles",
    "discover_plugins",
    "identify_matches",
]


def discover_plugins() -> list[GamePlugin]:
    plugins: list[GamePlugin] = []
    eps = entry_points()
    selected = eps.select(group=ENTRY_POINT_GROUP) if hasattr(eps, "select") else eps.get(ENTRY_POINT_GROUP, [])
    for ep in selected:
        obj = ep.load()
        if isinstance(obj, type):
            plugin = obj()
        elif callable(obj):
            plugin = obj()
        else:
            plugin = obj
        plugins.append(plugin)
    return plugins


def identify_matches(plugins: list[GamePlugin], frame_bgr, ctx: IdentifyContext) -> list[GamePlugin]:
    return [p for p in plugins if p.identify(frame_bgr, ctx)]
