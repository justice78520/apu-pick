"""出狀況時不能丟資料：寫檔寫到一半、搬檔被中斷、單張量不出來、按停止、子行程當掉。"""

import os
import threading
from pathlib import Path

import numpy as np
import pytest

from astro_light_selector import fileio, metrics, mover, organize, pipeline, scoring
from astro_light_selector.metrics import FrameMetrics
from astro_light_selector.selector import Decision

from .synth import make_star_field, write_fits


def test_atomic_write_keeps_the_old_file_when_interrupted(tmp_path: Path):
    path = tmp_path / "selection_report.csv"
    path.write_text("old", encoding="utf-8")
    with pytest.raises(RuntimeError):
        with fileio.atomic_write(path) as fh:
            fh.write("half of the new")
            raise RuntimeError("當機")
    assert path.read_text(encoding="utf-8") == "old"
    assert list(tmp_path.iterdir()) == [path]          # 暫存檔有清掉
    with fileio.atomic_write(path) as fh:
        fh.write("new")
    assert path.read_text(encoding="utf-8") == "new" and list(tmp_path.iterdir()) == [path]


def _frame(path: Path, **kw) -> FrameMetrics:
    base = dict(file=str(path), n_stars=100, fwhm=3.0, eccentricity=0.3, background=800.0, noise=15.0, snr=50.0,
                saturated_frac=0.0, exposure=120.0, filter="L", date_obs=None)
    return FrameMetrics(**(base | kw))


def test_move_log_is_written_as_each_file_moves(tmp_path: Path, monkeypatch):
    """搬到一半程式被強制關掉（最後那次整份重寫沒機會做）：已經搬走的要有紀錄，全部還原才搬得回來。"""
    files = [tmp_path / f"light_{i}.fits" for i in range(4)]
    for p in files:
        p.write_bytes(b"x")
    reject = tmp_path / "rejected"
    monkeypatch.setattr(mover, "write_move_log", lambda *a, **k: None)   # 模擬來不及做最後的重寫
    out, _ = mover.sync_files([Decision(_frame(p), keep=False) for p in files], reject)
    assert len(out) == 4 and len(mover.read_move_log(reject)) == 4
    monkeypatch.undo()
    # 紀錄最後一行只寫了一半也讀得了
    with (reject / mover.MOVE_LOG).open("a", encoding="utf-8") as fh:
        fh.write(str(tmp_path / "light_9.fits"))
    assert len(mover.read_move_log(reject)) == 4
    assert len(mover.restore_rejected(reject, tmp_path)) == 4 and all(p.exists() for p in files)


def test_organize_log_is_written_as_each_file_moves(tmp_path: Path, monkeypatch):
    frames = []
    for i, name in enumerate(("Ha", "OIII", "Ha")):
        p = tmp_path / f"light_{i}.fits"
        p.write_bytes(b"x")
        frames.append(_frame(p, filter=name))
    plan = organize.plan_organize(frames, tmp_path, ("filter",))
    monkeypatch.setattr(organize, "write_log", lambda *a, **k: None)
    assert len(organize.organize(plan, tmp_path)) == 3
    monkeypatch.undo()
    assert organize.pending_undo(tmp_path) == 3
    assert len(organize.undo_organize(tmp_path)) == 3
    assert sorted(p.name for p in tmp_path.glob("*.fits")) == ["light_0.fits", "light_1.fits", "light_2.fits"]
    assert organize.pending_undo(tmp_path) == 0


def test_one_bad_frame_does_not_fail_the_batch(tmp_path: Path, monkeypatch):
    """讀得進來但量測中出錯（檔頭寫了奇怪的值之類）：只有那一張標成失敗。"""
    good = write_fits(tmp_path / "good.fits", make_star_field(seed=1))

    def boom(*a, **k):
        raise ZeroDivisionError("怪檔案")

    assert metrics.measure(good).error is None
    monkeypatch.setattr(metrics, "sigma_clipped_stats", boom)
    m = metrics.measure(good)
    assert m.error.startswith("measure_failed:ZeroDivisionError") and m.n_stars == 0


def test_stop_keeps_what_was_already_measured(tmp_path: Path):
    files = [write_fits(tmp_path / f"s{i}.fits", make_star_field(seed=i)) for i in range(4)]
    cancel = threading.Event()

    def progress(i, n, m):
        if i == 2:
            cancel.set()

    with pytest.raises(pipeline.Cancelled) as stopped:
        pipeline.measure_files(files, 1, progress=progress, cancel=cancel)
    assert [Path(m.file).name for m in stopped.value.partial] == ["s0.fits", "s1.fits"]


def crashing_measure(path: Path) -> FrameMetrics:
    """量到檔名有 crash 的那張，子行程直接死掉（像壞掉的 RAW 讓解碼的函式庫當掉）。"""
    if "crash" in Path(path).name:
        os._exit(3)
    return _frame(Path(path))


def test_a_crashing_frame_only_fails_itself(tmp_path: Path, monkeypatch):
    files = [tmp_path / name for name in ("a.fits", "b.fits", "crash.fits", "c.fits", "d.fits")]
    monkeypatch.setattr(pipeline, "measure", crashing_measure)
    frames = pipeline.measure_files(files, workers=2)
    assert [Path(m.file).name for m in frames] == [p.name for p in files]
    assert [m.error for m in frames] == [None, None, "crashed", None, None]


def test_reference_uses_at_least_three_frames():
    """張數少時範本不能只看一張：8 張裡有一張背景特別低，不該把其他 7 張全部比下去。"""
    frames = [_frame(Path(f"f{i}.fits"), background=800.0 + i) for i in range(7)]
    frames.append(_frame(Path("dark_sky.fits"), background=400.0))
    one = scoring.score_frames(frames, scoring.ScoreConfig(min_top=1))
    three = scoring.score_frames(frames, scoring.ScoreConfig())
    assert one.reference["background"] == 400.0
    assert three.reference["background"] == pytest.approx(np.mean([400.0, 800.0, 801.0]))
    assert min(three.scores.values()) > min(one.scores.values())
    # 張數夠多時跟以前一樣取前 10%
    many = [_frame(Path(f"m{i}.fits"), background=500.0 + i) for i in range(100)]
    assert scoring.score_frames(many, scoring.ScoreConfig()).reference["background"] == pytest.approx(504.5)
    assert scoring.score_frames(frames[:2], scoring.ScoreConfig()).reference["background"] == pytest.approx(800.5)


def test_saturated_fraction_uses_the_real_limit(tmp_path: Path):
    """16-bit 沒寫 BITDEPTH：飽和值是 65535，不是「這張最亮的那個值」（那樣每張都會有幾個飽和像素）。"""
    m = metrics.measure(write_fits(tmp_path / "a.fits", make_star_field(seed=1)))
    assert m.error is None and m.saturated_frac == 0.0
