"""Filesystem helpers that respect sudo (run-pi.sh → root)."""

from __future__ import annotations

import os
from pathlib import Path


def gamebot_home() -> Path:
    """~/.gamebot for the real user, even when run via sudo/run-pi.sh.

    Capture/cal must not land in /root/.gamebot — laptop pull uses max@….
    """
    sudo_user = os.environ.get("SUDO_USER")
    if sudo_user and sudo_user != "root":
        return Path("/home") / sudo_user / ".gamebot"
    return Path.home() / ".gamebot"
