import numpy as np

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
RH_C, LH_C = "#2a78d6", "#eb6834"


def to_musicxml(midi_path, xml_path):
    import music21

    s = music21.converter.parse(midi_path, quarterLengthDivisors=(4,))
    for p, name in zip(s.parts, ("Right Hand", "Left Hand")):
        p.partName = name
    if len(s.parts) > 1:
        s.parts[1].getElementsByClass("Measure")[0].insert(0, music21.clef.BassClef())
    s.write("musicxml", fp=xml_path)


def plot_hands(path, title, S, E, P, H, cl_time, split):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.collections import PolyCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    for fam in ("IPAPGothic", "IPAGothic", "Noto Sans CJK JP", "Hiragino Sans", "Yu Gothic", "Meiryo"):
        if any(f.name == fam for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = fam
            break

    fig = plt.figure(figsize=(16, 7.4), dpi=150, facecolor=SURFACE)
    gs = fig.add_gridspec(1, 2, width_ratios=[6.2, 1], wspace=0.025, left=0.065, right=0.985, top=0.80, bottom=0.10)
    ax = fig.add_subplot(gs[0])
    axh = fig.add_subplot(gs[1], sharey=ax)
    for a in (ax, axh):
        a.set_facecolor(SURFACE)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(AXIS)
        a.tick_params(colors=MUTED, labelcolor=INK2, labelsize=9.5, length=0)
    ymin, ymax = int(P.min()) - 2, int(P.max()) + 2
    cs = [c for c in range(0, 128, 12) if ymin <= c <= ymax]
    for c in cs:
        ax.axhline(c, color=GRID, lw=0.8, zorder=0)
        axh.axhline(c, color=GRID, lw=0.8, zorder=0)

    def rects(m):
        w = np.maximum(E[m] - S[m], 0.18)
        return [[(x, y - 0.36), (x + ww, y - 0.36), (x + ww, y + 0.36), (x, y + 0.36)]
                for x, y, ww in zip(S[m], P[m], w)]

    ax.add_collection(PolyCollection(rects(H == 0), facecolors=LH_C, edgecolors="none", zorder=2))
    ax.add_collection(PolyCollection(rects(H == 1), facecolors=RH_C, edgecolors="none", zorder=2))
    ax.step(cl_time, split - 0.5, where="post", color=MUTED, lw=1.3, zorder=3)
    ax.set_xlim(0, E.max() + 1)
    ax.set_ylim(ymin, ymax)
    ax.set_yticks(cs, [f"C{c // 12 - 1}" + ("（中央ド）" if c == 60 else "") for c in cs])
    ax.set_xlabel("時間（秒）", color=INK2, fontsize=10)

    hl, hr = np.bincount(P[H == 0], minlength=128), np.bincount(P[H == 1], minlength=128)
    ys = np.arange(128)
    axh.barh(ys, hl, height=0.72, color=LH_C, edgecolor=SURFACE, linewidth=1)
    axh.barh(ys, hr, left=hl, height=0.72, color=RH_C, edgecolor=SURFACE, linewidth=1)
    axh.tick_params(labelleft=False)
    axh.set_xlabel("音の数", color=INK2, fontsize=10)
    axh.set_xlim(0, (hl + hr).max() * 1.3)

    fig.text(0.065, 0.945, f"{title}：音の高さと左右の手の推定", fontsize=16, color=INK)
    fig.text(0.065, 0.905, f"全{len(P):,}音。各時点で両手の“すき間”を通る境界線を動的計画法で求め、左右に振り分けた結果",
             fontsize=10.5, color=INK2)
    fig.legend(handles=[Patch(color=RH_C, label=f"右手  {int((H == 1).sum()):,}音"),
                        Patch(color=LH_C, label=f"左手  {int((H == 0).sum()):,}音"),
                        Line2D([], [], color=MUTED, lw=1.3, label="推定した左右の境界")],
               loc="upper left", bbox_to_anchor=(0.06, 0.885), ncol=3, frameon=False, fontsize=10,
               labelcolor=INK, handlelength=1.2, handleheight=0.9, columnspacing=1.6)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
