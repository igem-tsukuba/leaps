from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from easydict import EasyDict as edict


class Config(edict):
    """
    設定を管理するクラス
    """

    def __init__(self, path: str | Path = "./config.yaml") -> None:  # noqa: D401
        """
        Args:
            path (str | Path): 設定ファイルのパス
        """
        path = Path(path)
        config: dict[str, Any]
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        else:
            config = {}

        super().__init__(config)
