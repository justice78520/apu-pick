"""清單右邊「資訊」「直方圖」兩頁的內容：選到的那一張的拍攝資訊、分數拆解、跟同組比較，以及這張照片的直方圖。

只負責組內容（文字、畫圖），視窗怎麼擺在 gui.py。
"""

from __future__ import annotations

import math

import numpy as np
from matplotlib.figure import Figure

from .field import FieldResult
from .i18n import tr
from .plot import DARK
from .preview import Preview
from .scoring import METRICS, ScoreResult
from .selector import Decision

# 指標顯示的格式與單位
_FMT = {"fwhm": ("{:.2f}", " px"), "eccentricity": ("{:.2f}", ""), "n_stars": ("{:.0f}", ""),
        "background": ("{:.0f}", " ADU"), "snr": ("{:.1f}", "")}
_COLUMN_LABEL = {"score": "gui.col.score", "fwhm": "gui.col.fwhm", "eccentricity": "gui.col.ecc",
                 "n_stars": "gui.col.stars", "background": "gui.col.bkg", "snr": "gui.col.snr"}

Line = tuple[str, str]   # (文字, 標籤：head / dim / bad / "")


def metric_label(metric: str) -> str:
    return tr(_COLUMN_LABEL[metric])


def _fmt(metric: str, v: float | None) -> str:
    if v is None or not math.isfinite(v):
        return "—"
    spec, unit = _FMT[metric]
    return spec.format(v) + unit


def _value(d: Decision, metric: str) -> float:
    v = d.score if metric == "score" else getattr(d.metrics, metric)
    return float("nan") if v is None else float(v)


def info_lines(d: Decision, result: ScoreResult | None, weights: dict[str, float], group: list[Decision],
               preview: Preview | None, sky: str, threshold: float | None) -> list[Line]:
    """「資訊」頁：拍攝、結果、分數拆解、跟同組比。weights 是換算成比例（加起來 1）的權重。"""
    m = d.metrics
    info = preview.info if preview is not None else {}
    lines: list[Line] = []

    def head(text: str) -> None:
        lines.append((("\n" if lines else "") + text + "\n", "head"))

    def row(label: str, value: str, tag: str = "") -> None:
        lines.append((f"{label}\t{value}\n", tag))

    # ---- 拍攝
    head(tr("info.shot"))
    if m.date_obs:
        row(tr("info.time"), m.date_obs.replace("T", " ")[:19])
    if m.exposure:
        row(tr("info.exposure"), f"{m.exposure:g} s")
    gain = f"ISO {m.iso:g}" if m.iso is not None else (f"{m.gain:g}" if m.gain is not None else "")
    if gain:
        offset = info.get("OFFSET")
        row(tr("info.gain"), gain + (tr("info.offset", v=offset) if offset not in (None, "") else ""))
    row(tr("info.filter"), m.filter or tr("group.no_filter"))
    camera = m.camera or info.get("INSTRUME")
    if camera:
        row(tr("info.camera"), str(camera))
    if info.get("TELESCOP"):
        row(tr("info.telescope"), str(info["TELESCOP"]))
    focal, pixel = info.get("FOCALLEN"), info.get("XPIXSZ")
    if focal:
        row(tr("info.focal"), f"{float(focal):g} mm")
    if pixel:
        row(tr("info.pixel"), f"{float(pixel):.2f} µm")
    if focal and pixel:  # XPIXSZ 照 FITS 慣例已經是 binning 後的大小
        row(tr("info.scale"), f"{206.265 * float(pixel) / float(focal):.2f}″/px")
    if info.get("XBINNING"):
        row(tr("info.binning"), f"{info['XBINNING']}×{info.get('YBINNING', info['XBINNING'])}")
    if info.get("CCD-TEMP") is not None:
        temp = f"{float(info['CCD-TEMP']):.1f} °C"
        if info.get("SET-TEMP") is not None:
            temp += tr("info.temp_set", t=float(info["SET-TEMP"]))
        row(tr("info.temp"), temp)
    if info.get("SIZE"):
        w, h = info["SIZE"]
        row(tr("info.size"), f"{w} × {h}")
    if info.get("OBJECT"):
        row(tr("info.object"), str(info["OBJECT"]))
    if sky:
        row(tr("info.sky"), sky)
    if preview is None:
        lines.append((tr("info.loading") + "\n", "dim"))

    # ---- 結果
    head(tr("info.result"))
    verdict = tr("term.keep") if d.keep else tr("term.reject")
    if d.override:
        verdict += tr("info.manual")
    row(tr("info.verdict"), verdict, "" if d.keep else "bad")
    if d.score is not None:
        row(tr("info.score"), tr("info.score_value", s=d.score, t=threshold) if threshold is not None
            else f"{d.score:.1f}")
        scored = sorted((x.score for x in group if x.score is not None), reverse=True)
        row(tr("info.rank"), tr("info.rank_value", r=scored.index(d.score) + 1, n=len(scored)))
    if d.group:
        row(tr("info.group"), d.group)
    if d.reasons:
        row(tr("info.reasons"), "; ".join(str(r) for r in d.reasons), "" if d.keep else "bad")

    # ---- 分數拆解：每個指標跟範本比幾 %、乘上權重拿幾分、扣幾分
    head(tr("info.breakdown"))
    comp = result.components.get(m.file) if result is not None else None
    if comp is None:
        lines.append((tr("info.no_score") + "\n", "dim"))
    else:
        losses = {k: weights.get(k, 0) * (1 - comp[k]) * 100 for k in METRICS if weights.get(k, 0) > 0}
        worst = max(losses, key=losses.get) if losses else None
        for k in METRICS:
            w = weights.get(k, 0)
            if w <= 0:
                continue
            text = tr("info.breakdown_row", pct=comp[k] * 100, w=w * 100, pts=w * comp[k] * 100, lost=losses[k])
            row(metric_label(k), text, "bad" if k == worst and losses[k] >= 0.5 else "")
        row(tr("info.total"), f"{d.score:.1f}" if d.score is not None else "—")

    # ---- 跟同組比：這張、同組中位數（差幾 %）、範本
    head(tr("info.compare"))
    for k in METRICS:
        v = getattr(m, k)
        values = np.array([_value(x, k) for x in group])
        values = values[np.isfinite(values)]
        if not len(values) or v is None or not math.isfinite(v):
            continue
        med = float(np.median(values))
        diff = (v - med) / med * 100 if med else 0.0
        ref = result.reference.get(k) if result is not None else None
        text = tr("info.compare_row", v=_fmt(k, v), med=_fmt(k, med), diff=diff)
        if ref is not None:
            text += tr("info.compare_ref", ref=_fmt(k, ref))
        row(metric_label(k), text)
    return lines


def _style(ax, theme: dict) -> None:
    ax.set_facecolor(theme["axes"])
    for spine in ax.spines.values():
        spine.set_color(theme["spine"])
    ax.tick_params(colors=theme["muted"], labelcolor=theme["muted"], labelsize=7)
    ax.grid(color=theme["grid"], linewidth=0.6)
    ax.set_axisbelow(True)


_CHANNEL_COLORS = ("#ff6b6b", "#5cd65c", "#5cadff")


def draw_histogram(fig: Figure, preview: Preview | None, stretched: bool = True, luminance: bool = False) -> None:
    """這一張照片的直方圖。

    stretched：拉伸後（跟預覽同一套拉伸，0~255，像一般修圖軟體）；否則拉伸前的線性 ADU（x、y 都用對數，
    背景的山峰和亮星都看得到，標出背景和飽和線）。luminance：彩色的合成一條亮度，不分 R、G、B。
    """
    theme = DARK
    fig.clear()
    fig.set_facecolor("#141414")
    ax = fig.subplots()
    _style(ax, theme)
    if preview is None or preview.hist is None:
        ax.text(0.5, 0.5, tr("gui.hist.loading"), color=theme["muted"], ha="center", va="center",
                transform=ax.transAxes, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    color_image = len(preview.hist) == 3 and not luminance
    if stretched:
        rows = preview.stretched_hist if color_image else [preview.stretched_lum]
        edges = np.linspace(0, 255, 257)
    else:
        rows = preview.hist if color_image else [preview.hist_lum]
        edges = preview.hist_edges
    colors = _CHANNEL_COLORS if color_image else (theme["text"],)
    for counts, color in zip(rows, colors):
        if stretched:
            ax.stairs(counts, edges, color=color, linewidth=1, fill=True, alpha=0.25 if color_image else 0.35)
            ax.stairs(counts, edges, color=color, linewidth=1)
        else:
            ax.stairs(np.maximum(counts, 0.8), edges, color=color, linewidth=1)
    if stretched:
        ax.set_xlim(0, 255)
        ax.set_ylim(bottom=0)
        ax.set_yticks([])
        ax.set_xlabel(tr("gui.hist.stretched_axis"), color=theme["muted"], fontsize=7)
    else:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.8)
        ax.axvline(float(np.mean(preview.background)), color=theme["muted"], linestyle="--", linewidth=0.8,
                   label=tr("gui.hist.background"))
        if preview.saturation:
            ax.axvline(preview.saturation, color=theme["reject"], linewidth=1, label=tr("gui.hist.saturation"))
        ax.set_xlabel(tr("gui.hist.adu"), color=theme["muted"], fontsize=7)
        legend = ax.legend(loc="upper right", fontsize=7, facecolor=theme["legend"], edgecolor=theme["spine"])
        for text in legend.get_texts():
            text.set_color(theme["text"])
    fig.tight_layout(pad=0.6)


def histogram_stats(preview: Preview | None) -> list[Line]:
    """直方圖下面的數字：每個通道的背景（中間值）、最暗（0.1%）、最亮、飽和像素比例；彩色再加背景的色偏。"""
    if preview is None or not preview.stats:
        return []
    names = ("R", "G", "B") if len(preview.stats) == 3 else (tr("gui.hist.mono"),)
    lines: list[Line] = [(tr("gui.hist.stats_head") + "\n", "dim")]
    for name, st in zip(names, preview.stats):
        sat = "—" if st["saturated"] is None else f"{st['saturated'] * 100:.3f}%"
        lines.append((f"{name}\t{st['median']:.0f}\t{st['low']:.0f}\t{st['high']:.0f}\t{sat}\n", ""))
    if len(preview.stats) == 3:
        r, g, b = (st["median"] for st in preview.stats)
        if g > 0:
            lines.append((tr("gui.hist.cast", r=r / g, b=b / g) + "\n", "dim"))
    return lines


QUAD_GAIN = 3.0     # 四邊形把四角 FWHM 的差異放大幾倍畫（傾斜十幾 % 照比例畫幾乎看不出來）
QUAD_LIMIT = 0.5    # 一個角最多凸出／凹進多少（相對於半邊長），差太多時不要畫出框外


def _empty_axes(fig: Figure):
    theme = DARK
    fig.clear()
    fig.set_facecolor("#141414")
    ax = fig.subplots()
    ax.set_facecolor(theme["axes"])
    for spine in ax.spines.values():
        spine.set_color(theme["spine"])
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def draw_field(fig: Figure, result: FieldResult | None, message: str = "", view: str = "grid") -> None:
    """像場。view="grid"：實際照片的九宮格，每格標那一區的星點大小；view="quad"：像場四邊形。"""
    if result is None:
        ax = _empty_axes(fig)
        ax.text(0.5, 0.5, message, color=DARK["muted"], ha="center", va="center", transform=ax.transAxes,
                fontsize=8, wrap=True)
        return
    if view == "quad" or result.tiles is None:
        _draw_field_quad(fig, result)
    else:
        _draw_field_grid(fig, result)


def _draw_field_grid(fig: Figure, result: FieldResult) -> None:
    """九宮格：照片四角、四邊中點、正中央各一塊拼在一起，左上角標那一區的 FWHM（px）；最大的那個角標紅色。"""
    from .field import GRID

    theme = DARK
    ax = _empty_axes(fig)
    tiles = result.tiles
    t = tiles[0][0].shape[0]
    gap = max(2, t // 32)
    size = GRID * t + (GRID - 1) * gap
    mosaic = np.full((size, size), 70, dtype=np.uint8)   # 格線的灰
    for r in range(GRID):
        for c in range(GRID):
            y, x = r * (t + gap), c * (t + gap)
            mosaic[y:y + t, x:x + t] = tiles[r][c][:t, :t]
    ax.imshow(mosaic, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    worst = result.worst_corner
    from .field import CORNERS
    worst_cell = CORNERS.get(worst) if worst else None
    for r, row in enumerate(result.cells):
        for c, cell in enumerate(row):
            x, y = c * (t + gap) + t * 0.05, r * (t + gap) + t * 0.05
            text = f"{cell.fwhm:.2f}" if cell is not None else "—"
            color = theme["reject"] if (r, c) == worst_cell else theme["text"] if cell is not None else theme["muted"]
            ax.text(x, y, text, color=color, ha="left", va="top", fontsize=8,
                    bbox={"facecolor": "#000000", "alpha": 0.6, "edgecolor": "none", "pad": 1.5})
    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(size - 0.5, -0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0.4)


def quad_points(result: FieldResult) -> dict[str, tuple[float, float]] | None:
    """像場四邊形四個頂點的位置（完全平整時是 ±1 的正方形）：哪個角的 FWHM 比四角平均大就往外凸，小就往內縮。
    差異放大 QUAD_GAIN 倍。四角有量不到的回傳 None。座標是影像方向（x 往右、y 往下）。"""
    from .field import CORNERS, GRID

    corners = result.corners
    if corners is None:
        return None
    mean = float(np.mean([c.fwhm for c in corners.values()]))
    points = {}
    for name, (r, c) in CORNERS.items():
        k = 1.0 + float(np.clip(QUAD_GAIN * (corners[name].fwhm / mean - 1.0), -QUAD_LIMIT, QUAD_LIMIT))
        points[name] = ((c - GRID // 2) * k, (r - GRID // 2) * k)
    return points


def _draw_field_quad(fig: Figure, result: FieldResult) -> None:
    """像場四邊形：虛線是完全平整的樣子，實線是量到的（哪個角星點大就往外凸）。九個位置標 FWHM，短線是星點拉長的方向。"""
    from .field import CORNERS, GRID

    theme = DARK
    ax = _empty_axes(fig)
    ax.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], color=theme["muted"], linewidth=0.8, linestyle="--")
    points = quad_points(result)
    if points is not None:
        order = ("tl", "tr", "br", "bl")
        xs = [points[k][0] for k in order]
        ys = [points[k][1] for k in order]
        ax.fill(xs, ys, color=theme["keep"], alpha=0.12)
        ax.plot(xs + xs[:1], ys + ys[:1], color=theme["keep"], linewidth=1.6)
        ax.scatter(xs, ys, color=theme["keep"], s=14, zorder=3)
    names = {pos: name for name, pos in CORNERS.items()}
    worst = result.worst_corner
    for r, row in enumerate(result.cells):
        for c, cell in enumerate(row):
            dx, dy = c - GRID // 2, r - GRID // 2
            name = names.get((r, c))
            if name is not None and points is not None:   # 角落：數字放在頂點外側
                px, py = points[name]
                tx, ty = px + dx * 0.34, py + dy * 0.24
            elif (dx, dy) == (0, 0) or points is None:
                tx, ty = dx * 1.34, dy * 1.24
            else:   # 四邊：數字放在那一邊兩個頂點裡比較凸的那個外面，才不會壓到線
                side = [p for k, p in points.items() if (dx and CORNERS[k][1] - GRID // 2 == dx)
                        or (dy and CORNERS[k][0] - GRID // 2 == dy)]
                tx = dx * (max(1.0, *(abs(p[0]) for p in side)) + 0.36) if dx else 0.0
                ty = dy * (max(1.0, *(abs(p[1]) for p in side)) + 0.26) if dy else 0.0
            if cell is None:
                ax.text(tx, ty, "—", color=theme["muted"], ha="center", va="center", fontsize=9)
                continue
            color = theme["reject"] if name is not None and name == worst else theme["text"]
            ax.text(tx, ty - 0.09, f"{cell.fwhm:.2f}", color=color, ha="center", va="center", fontsize=9)
            half = 0.16 * min(1.0, cell.eccentricity) * cell.alignment   # 星點拉長的方向，越長拉得越多
            ex, ey = math.cos(cell.angle) * half, math.sin(cell.angle) * half
            ax.plot([tx - ex, tx + ex], [ty + 0.13 - ey, ty + 0.13 + ey], color=theme["muted"], linewidth=1.6,
                    solid_capstyle="round")
    ax.set_xlim(-2.15, 2.15)
    ax.set_ylim(1.95, -1.95)   # 影像方向：上面是照片的上緣
    ax.set_aspect("equal", adjustable="box")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0.4)


def _level(tilt: float) -> str:
    return tr("field.level.ok") if tilt < 10 else tr("field.level.mild") if tilt < 20 else tr("field.level.bad")


def field_lines(result: FieldResult | None, view: str = "grid") -> list[Line]:
    """像場圖下面的文字：傾斜、場曲、中心與四角的 FWHM、星點拉長的樣子、圖怎麼看、量不準的提醒。"""
    if result is None:
        return []
    lines: list[Line] = []

    def row(label: str, value: str, tag: str = "") -> None:
        lines.append((f"{label}\t{value}\n", tag))

    tilt, curvature, corners, center = result.tilt, result.curvature, result.corners, result.center
    if tilt is not None:
        row(tr("field.tilt"), tr("field.tilt_value", v=tilt, corner=tr(f"field.corner.{result.worst_corner}"),
                                 level=_level(tilt)), "bad" if tilt >= 20 else "")
    if curvature is not None:
        row(tr("field.curvature"), tr("field.curv_value", v=curvature,
                                      dir=tr("field.bigger") if curvature >= 0 else tr("field.smaller")),
            "bad" if abs(curvature) >= 20 else "")
    if center is not None:
        row(tr("field.center"), f"{center.fwhm:.2f} px")
    if corners is not None:
        row(tr("field.corners"), tr("field.corners_value", **{k: c.fwhm for k, c in corners.items()}))
    pattern = result.corner_pattern()
    row(tr("field.pattern"), tr(f"field.pattern.{pattern or 'none'}"))
    lines.append(((tr("field.frames", n=result.n_frames) if result.n_frames > 1 else tr("field.frames_single"))
                  + "\n", "dim"))
    if view == "quad" or result.tiles is None:
        lines.append((tr("field.legend.quad", gain=QUAD_GAIN) + "\n", "dim"))
    else:
        lines.append((tr("field.legend.grid", n=result.tiles[0][0].shape[0])
                      + (tr("field.legend.grid_best") if result.n_frames > 1 else "") + "\n", "dim"))
    for note in result.notes:
        lines.append((tr(note) + "\n", "bad"))
    return lines
