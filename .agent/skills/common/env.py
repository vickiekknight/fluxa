"""Shared .env loading for host-side scripts.

The repo keeps secrets in a gitignored `.env` at the project root (see README),
but only `start_isaacsim_stream.py` ever loaded it -- and that runs *inside* the
container. Host-side scripts (`1_eureka.py`, `3_dr_eureka.py`) read `os.environ`
directly, so a fresh shell gave them nothing unless the caller had exported the
key by hand, and `genai.Client(api_key=None)` then failed further downstream with
a confusing error.

Written without `python-dotenv` on purpose: that package is installed in the
container but not on the host, and this is small enough not to be worth a new
host dependency.
"""
import os
from pathlib import Path


def load_env_file(start=None, verbose=True):
    """Load KEY=VALUE pairs from the nearest `.env` at or above `start`.

    Walks upward from `start` (default: this file's directory) to the filesystem
    root and loads the first `.env` found. Variables already present in the
    environment always win, so an explicit `export`, a CI secret, or a value
    injected by docker-compose is never clobbered.

    Returns the Path that was loaded, or None if no `.env` exists above `start`.
    """
    origin = Path(start).resolve() if start else Path(__file__).resolve()
    search_from = origin if origin.is_dir() else origin.parent

    for directory in [search_from, *search_from.parents]:
        env_path = directory / ".env"
        if not env_path.is_file():
            continue

        loaded = []
        for raw_line in env_path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()

            # Strip one layer of matching quotes, the way dotenv does.
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]

            if key and key not in os.environ:
                os.environ[key] = value
                loaded.append(key)

        if verbose and loaded:
            # Names only -- never the values.
            print(f"[env] loaded {', '.join(sorted(loaded))} from {env_path}")
        return env_path

    return None
