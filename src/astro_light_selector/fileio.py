"""寫檔的小工具：報表、搬移紀錄、設定檔都先寫到旁邊的暫存檔，寫完才換成正式的檔名。

直接覆寫的話，寫到一半當機或斷電會只剩半個檔；報表裡有量測結果和手動覆寫，搬移紀錄是還原的依據。
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO


@contextmanager
def atomic_write(path: Path, newline: str | None = None, encoding: str = "utf-8") -> Iterator[IO[str]]:
    """開一個文字檔來寫；離開時才取代 path。中途出錯就把暫存檔刪掉，原本的檔案不動。

    目的地被別的程式開著（例如報表開在 Excel 裡）時 os.replace 會丟 OSError，跟直接覆寫一樣由呼叫的人處理。
    """
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("w", newline=newline, encoding=encoding) as fh:
            yield fh
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass  # 有些網路磁碟不支援，不影響換檔名
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise
