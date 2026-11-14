"""新版本提醒：查 GitHub 上最新的 Release，比目前的版本新就給這台電腦對應的下載連結。

只查、不自己下載或替換程式：下載交給瀏覽器，系統原本的安全提示照常出現。
一天最多連一次網路（結果存在設定檔，一天內重開程式直接用上次的結果）；沒網路、被擋、格式不對都安靜略過。
只連到 GitHub，不送任何個人資料。設定 check_updates=false 就不查。引擎不 import tkinter。
"""

from __future__ import annotations

import json
import platform
import re
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from . import __version__
from .settings import load_settings, save_settings

REPO = "ApuHuang/apu-pick"
API_URL = f"https://api.github.com/repos/{REPO}/releases/latest"
RELEASES_URL = f"https://github.com/{REPO}/releases/latest"
INTERVAL = 24 * 3600      # 秒：多久才再連網查一次
TIMEOUT = 8.0


@dataclass(frozen=True)
class Update:
    version: str          # 例如 "0.2.3"
    url: str              # 這台電腦對應的 zip；Release 上找不到就是 Release 頁面
    file: str             # zip 檔名；找不到就是空字串


def parse_version(text: str) -> tuple[int, ...] | None:
    """「v0.2.3」「0.2.3」→ (0, 2, 3)；其他寫法（例如預覽版的 0.3.0rc1）不算。"""
    m = re.fullmatch(r"v?(\d+(?:\.\d+)*)", str(text).strip())
    return tuple(int(p) for p in m.group(1).split(".")) if m else None


def platform_tag() -> str:
    """跟打包的 zip 檔名同一套：win64、macos-arm64、macos-x86_64。"""
    if sys.platform == "win32":
        return "win64"
    if sys.platform == "darwin":
        return f"macos-{platform.machine()}"
    return f"{sys.platform}-{platform.machine()}"


def newer(release: dict | None, current: str = __version__) -> Update | None:
    """Release 的資料（GitHub API 的格式）比 current 新就回傳下載資訊，否則 None。"""
    if not isinstance(release, dict):
        return None
    latest, now = parse_version(release.get("tag_name", "")), parse_version(current)
    if latest is None or now is None or latest <= now:
        return None
    suffix = f"-{platform_tag()}.zip"
    for asset in release.get("assets") or []:
        if not isinstance(asset, dict):
            continue
        name, url = str(asset.get("name", "")), asset.get("browser_download_url")
        if name.endswith(suffix) and isinstance(url, str) and url.startswith("https://"):
            return Update(".".join(map(str, latest)), url, name)
    page = release.get("html_url")
    return Update(".".join(map(str, latest)), page if isinstance(page, str) and page.startswith("https://")
                  else RELEASES_URL, "")


def enabled(settings: dict | None = None) -> bool:
    settings = load_settings() if settings is None else settings
    return settings.get("check_updates", True) is not False


def cached(settings: dict | None = None) -> dict | None:
    """上次查到的 Release（只留需要的欄位）；沒查過是 None。"""
    settings = load_settings() if settings is None else settings
    entry = settings.get("update")
    return entry.get("release") if isinstance(entry, dict) and isinstance(entry.get("release"), dict) else None


def due(settings: dict | None = None, now: float | None = None) -> bool:
    """該連網查了嗎（上次查超過 INTERVAL，或沒查過）。"""
    settings = load_settings() if settings is None else settings
    entry = settings.get("update")
    try:
        checked = float(entry["checked"])
    except (TypeError, KeyError, ValueError):
        return True
    now = time.time() if now is None else now
    return not 0 <= now - checked < INTERVAL     # 時間倒退（改過時鐘）也重查


def remember(release: dict, now: float | None = None) -> None:
    """把查到的 Release 存起來（只留版本、頁面、zip 的名字與連結）。"""
    slim = {
        "tag_name": release.get("tag_name"),
        "html_url": release.get("html_url"),
        "assets": [{"name": a.get("name"), "browser_download_url": a.get("browser_download_url")}
                   for a in release.get("assets") or [] if isinstance(a, dict)],
    }
    save_settings(update={"checked": time.time() if now is None else now, "release": slim})


def fetch_latest(timeout: float = TIMEOUT) -> dict:
    """向 GitHub 查最新的 Release（不含預覽版與草稿）。連不上會丟例外。"""
    agent = f"APU-Pick/{__version__}"
    request = urllib.request.Request(API_URL, headers={"Accept": "application/vnd.github+json", "User-Agent": agent})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read(2_000_000).decode("utf-8"))
    except urllib.error.URLError as e:
        # Mac 打包版的 Python 可能找不到根憑證：改用系統的 curl（用 macOS 鑰匙圈裡的憑證）
        if sys.platform != "darwin" or not isinstance(e.reason, ssl.SSLError):
            raise
    result = subprocess.run(["/usr/bin/curl", "-fsSL", "--max-time", str(int(timeout)), "-A", agent,
                             "-H", "Accept: application/vnd.github+json", API_URL],
                            capture_output=True, timeout=timeout + 5, check=True)
    return json.loads(result.stdout.decode("utf-8"))
