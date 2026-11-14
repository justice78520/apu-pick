"""新版本提醒：版本比較、挑這台電腦對應的 zip、一天最多查一次。不連網路。"""

import pytest

from astro_light_selector import settings, update

ZIPS = ("APUPick-{v}-win64.zip", "APUPick-{v}-macos-arm64.zip", "APUPick-{v}-macos-x86_64.zip")


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    """測試不能動到使用者真正的設定檔。"""
    monkeypatch.setattr(settings, "config_dir", lambda: tmp_path)


def _release(tag, names=ZIPS):
    v = tag.lstrip("v")
    return {"tag_name": tag, "html_url": f"https://github.com/ApuHuang/apu-pick/releases/tag/{tag}",
            "assets": [{"name": n.format(v=v), "browser_download_url": f"https://github.com/dl/{n.format(v=v)}"}
                       for n in names]}


def test_parse_version():
    assert update.parse_version("v0.8.1") == (0, 8, 1) == update.parse_version(" 0.8.1 ")
    assert update.parse_version("0.9.0rc1") is None and update.parse_version("nightly") is None


def test_newer_picks_the_zip_for_this_computer(monkeypatch):
    monkeypatch.setattr(update, "platform_tag", lambda: "macos-arm64")
    got = update.newer(_release("v0.8.1"), "0.8.0")
    assert got == update.Update("0.8.1", "https://github.com/dl/APUPick-0.8.1-macos-arm64.zip",
                                "APUPick-0.8.1-macos-arm64.zip")
    assert update.newer(_release("v0.8.0"), "0.8.0") is None          # 一樣
    assert update.newer(_release("v0.7.0"), "0.8.0") is None          # 比較舊
    assert update.newer(_release("v0.10.0"), "0.9.9").version == "0.10.0"   # 照數字比，不是照字串比
    page_only = update.newer(_release("v0.9.0", names=()), "0.8.0")   # 還沒打包好：給 Release 頁面
    assert page_only.url.endswith("/releases/tag/v0.9.0") and page_only.file == ""
    for bad in (None, {}, {"tag_name": "nightly"}, {"tag_name": "v0.9.0rc1"}):
        assert update.newer(bad, "0.8.0") is None


def test_zip_names_match_the_packaging_script():
    """打包腳本（packaging/build_exe.py）的 zip 檔名要跟這裡挑檔的規則對得上，不然有新版也只會給 Release 頁面。"""
    from pathlib import Path

    script = (Path(__file__).parents[1] / "packaging" / "build_exe.py").read_text(encoding="utf-8")
    assert 'NAME = "APUPick"' in script and 'f"{NAME}-{__version__}-{platform_tag()}"' in script
    assert '"win64"' in script and 'f"macos-{platform.machine()}"' in script
    assert update.REPO == "ApuHuang/apu-pick"


def test_github_is_asked_at_most_once_a_day():
    assert update.enabled() and update.due(now=1000.0) and update.cached() is None
    update.remember(_release("v0.8.1") | {"body": "很長的說明"}, now=1000.0)
    assert not update.due(now=1000.0 + 3600)
    assert update.due(now=1000.0 + update.INTERVAL + 1)
    assert update.cached()["tag_name"] == "v0.8.1" and "body" not in update.cached()   # 只存需要的欄位
    settings.save_settings(check_updates=False)
    assert not update.enabled()
