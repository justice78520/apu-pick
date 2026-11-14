"""所有測試共用的設定：量測結果留著重複用，測試才跑得快。"""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import pytest

from astro_light_selector import metrics

_REAL_MEASURE = metrics.measure
_CACHE: dict[str, metrics.FrameMetrics] = {}


def cached_measure(path: Path) -> metrics.FrameMetrics:
    """跟 metrics.measure 一樣，但內容相同的檔案只量一次。

    合成星場是固定亂數種子產生的，很多測試量的是一模一樣的檔案；每個測試都重量一次，整套要跑快五分鐘。
    放在模組最外層：多核心量測時子行程要能照名字找到它（子行程有自己的快取，等於照常量測）。
    """
    key = hashlib.sha1(Path(path).read_bytes()).hexdigest()
    if key not in _CACHE:
        _CACHE[key] = _REAL_MEASURE(path)
    return dataclasses.replace(_CACHE[key], file=str(path))


@pytest.fixture(autouse=True)
def fast_measure(monkeypatch):
    """流程（命令列、視窗）裡的量測改用有快取的版本；視窗預設用單一行程量，才用得到快取。

    直接呼叫 metrics.measure 的測試、指定平行處理數的測試不受影響，照樣實際量測、走多核心那條路。
    """
    from astro_light_selector import gui, pipeline

    monkeypatch.setattr(pipeline, "measure", cached_measure)
    monkeypatch.setattr(gui, "default_workers", lambda: 1)
