"""沒預期到的錯誤記到設定資料夾的 error.log。

打包成沒有主控台的程式時 stderr 是丟掉的，視窗裡出錯使用者只會覺得「按了沒反應」；
記下來之後，回報問題時把這個檔案傳過來就知道出了什麼事。只記在這台電腦上，不會傳到任何地方。
"""

from __future__ import annotations

import platform
import sys
import time
import traceback
from pathlib import Path
from types import TracebackType

from . import __version__, settings

LOG_NAME = "error.log"
MAX_BYTES = 200_000   # 超過就只留後半，不會越長越大


def log_path() -> Path:
    return settings.config_dir() / LOG_NAME


def write(text: str) -> Path | None:
    """在紀錄最後加一筆（時間、版本、系統、內容），回傳紀錄檔的位置；寫不進去回傳 None。"""
    path = log_path()
    head = f"==== {time.strftime('%Y-%m-%d %H:%M:%S')}  APU Pick {__version__}  {platform.platform()}  " \
           f"Python {sys.version.split()[0]}\n"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file() and path.stat().st_size > MAX_BYTES:
            old = path.read_text(encoding="utf-8", errors="replace")
            path.write_text(old[len(old) // 2:], encoding="utf-8")
        with path.open("a", encoding="utf-8") as fh:
            fh.write(head + text.rstrip() + "\n\n")
    except OSError:
        return None
    return path


def write_exception(exc_type: type[BaseException], exc: BaseException, tb: TracebackType | None) -> Path | None:
    return write("".join(traceback.format_exception(exc_type, exc, tb)))
