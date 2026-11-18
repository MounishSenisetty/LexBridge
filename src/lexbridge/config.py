"""Config loading. Reads configs/config.yaml; env vars override LLM settings."""
from __future__ import annotations

import os
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(os.environ.get("LEXBRIDGE_CONFIG", ROOT / "configs" / "config.yaml"))


class Config(dict):
    """dict with attribute access and path helpers."""

    def __getattr__(self, item: str) -> Any:
        try:
            v = self[item]
        except KeyError as e:  # pragma: no cover
            raise AttributeError(item) from e
        return Config(v) if isinstance(v, dict) else v

    def path(self, key: str) -> Path:
        p = Path(self["paths"][key])
        return p if p.is_absolute() else ROOT / p

    @property
    def transition(self) -> date:
        return date.fromisoformat(str(self["transition_date"]))


@lru_cache(maxsize=1)
def get_config() -> Config:
    with open(CONFIG_PATH) as f:
        cfg = Config(yaml.safe_load(f))
    llm = cfg.setdefault("llm", {})
    for env, key in [("LLM_BASE_URL", "base_url"), ("LLM_API_KEY", "api_key"),
                     ("LLM_MODEL", "model"), ("LLM_BACKEND", "backend")]:
        if os.environ.get(env):
            llm[key] = os.environ[env]
    return cfg
