"""Every figure of the paper. Each function draws one figure from the result files (config.results) and returns it,
with the numbers it plots printed.

    fig1()   Figure 1: answer-layer CDFs (a-c), Gemma-3-12B over steps (d) and over training templates (e)
    fig2()   Figure 2: four single prompts across training (learning-rate sweep, Gemma-3-12B, S3, Chinese)
    fig3()   Figure 3 (App. A.3): known facts, pooled, per asked language
    fig4()   Figure 4 (App. B.3): adapter on query and value vs on the MLP output
    fig5()   Figure 5 (App. C.3): the top-k rule
"""
import json
import statistics as st
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import to_rgb

from .. import config as C
from ..data import templates
from . import metrics as M

# ------------------------------------------------------------------------------------------------ style
for _d in (Path("/mnt/c/Windows/Fonts"), Path("C:/Windows/Fonts"), Path("/usr/share/fonts/truetype/msttcorefonts")):
    if _d.exists():
        for _f in _d.glob("times*.ttf"):
            font_manager.fontManager.addfont(str(_f))
_HAVE = {f.name for f in font_manager.fontManager.ttflist}
FONT = [n for n in ("Times New Roman", "STIXGeneral") if n in _HAVE][:1]
BASE_STYLE = {"font.family": FONT, "mathtext.fontset": "stix", "pdf.fonttype": 42}
STYLE = {   # per figure, as the paper's figures were drawn
    "fig1": {**BASE_STYLE, "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7,
             "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
             "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.titlelocation": "left"},
    "appendix": {**BASE_STYLE, "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "xtick.labelsize": 8,
                 "ytick.labelsize": 8, "legend.fontsize": 8, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                 "ytick.major.width": 0.6},
    "fig2": {**BASE_STYLE, "font.size": 9},
}

INK, KCOL, PCOL = "#222222", "#111111", "#CC79A7"                # text, known facts, all templates
BASE = {"1": "#0072B2", "5": "#D55E00"}                          # S1, S5
MID, STEPS1 = "#9A9A9A", [32, 64, 128, 256]
LCOL = {"EN": "#0072B2", "zh": "#D55E00", "ru": "#009E73"}
LNAME = {"EN": "English", "zh": "Chinese", "ru": "Russian"}
MODELS = [("Qwen2.5-7B-Instruct", "Qwen2.5-7B"), ("gemma-3-12b-it", "Gemma-3-12B"), ("gemma-3-27b-it", "Gemma-3-27B")]
G12 = "gemma-3-12b-it"
EARLY = [16, 40, 48, 56, 96]                                     # Fig. 1d's extra checkpoints (new_fig1_exp)
EARLY_MLP = [8, 16, 24, 40, 48, 56, 96]                          # planned for the MLP site; not in the paper's results
FIX = {"Qwen2.5-7B-Instruct": 128, "gemma-3-12b-it": 64, "gemma-3-27b-it": 64}   # Fig. 1e's fixed step
SIZE_X = {"1": 0, "3": 0.75, "5": 1.25, "10": 1.95, "20": 2.6, "40": 3.25, "all": 4.1}
SIZE_COL = {"1": BASE["1"], "5": BASE["5"], "all": PCOL}
D_STEPS = sorted({0, 32, 64, 128, 256, *EARLY})


def ramp(hexcol, n):
    c = to_rgb(hexcol)
    return [tuple(1 - (1 - x) * t for x in c) for t in [0.35 + 0.65 * i / max(n - 1, 1) for i in range(n)]]


SHADE = {z: dict(zip(STEPS1, ramp(BASE[z], len(STEPS1)))) for z in BASE}


def roots(site="qv"):
    """Result folders of an adapter site: the main run + the extra checkpoints of Fig. 1d, e."""
    return (str(C.results("main")), str(C.results("fig1"))) if site == "qv" else (str(C.results("mlp_out")),)


def cdf(ax, x, color, lw, ls="-"):
    x = sorted(x)
    if x:
        ax.step([0] + x, [0] + [(i + 1) / len(x) for i in range(len(x))], where="post", color=color, lw=lw, ls=ls)


# ------------------------------------------------------------------------------------------------ Figure 1 (a-c)
def cdf_panel(ax, tag, R):
    for z in BASE:
        for s in STEPS1:
            x = [l for r in M.runs(R, tag, z, s) for l in r["layers"]]
            if len(x) >= 15:
                cdf(ax, x, SHADE[z][s], 1.2)
    cdf(ax, [l for r in M.runs(R, tag, "all", 256) for l in r["layers"]], PCOL, 1.6, (0, (3, 1.2)))
    cdf(ax, M.k_layers(tag, root=R[0]), KCOL, 2.0)
    nl = M.N_LAYERS[tag]
    ticks = list(range(0, nl, 8 if nl > 40 else 4)) + [nl - 1]
    lo = 20 if nl == 28 else nl // 2                             # Qwen: nothing is visible before L21
    ax.set_xticks([t for t in ticks if t >= lo], [f"L{t}" for t in ticks if t >= lo])
    ax.set_xlim(lo, nl - 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel(f"Layer (of {nl})")
    ax.set_yticks([0, 0.5, 1], ["0", "50", "100"])
    ax.grid(axis="y", alpha=0.3, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)


def fig1_legend(fig, y):
    cols = [[plt.Line2D([], [], color=KCOL, lw=2.2, label="Known facts"),
             plt.Line2D([], [], color=PCOL, lw=1.8, ls=(0, (3, 1.2)), label="All templates")],
            [plt.Line2D([], [], color="none", label="S1 (1 template), after"),
             plt.Line2D([], [], color="none", label="S5 (5 templates), after")]]
    cols += [[plt.Line2D([], [], color=SHADE[z][s], lw=2, label=f"{s}" + (" steps" if s == 256 else "")) for z in BASE]
             for s in STEPS1]
    fig.legend(handles=[h for c in cols for h in c], loc="lower center", bbox_to_anchor=(0.5, y), ncol=len(cols),
               frameon=False, handlelength=1.5, columnspacing=0.9, handletextpad=0.4, labelspacing=0.3)


# ------------------------------------------------------------------------------------------------ Figure 1 (d, e)
def _draw(ax, axk, axl, xs, S, colors, klayer, hollow_x=None):
    """Top: unseen paraphrases (solid), unseen language (dashed). Middle: known facts lost, downwards. Bottom: answer
    layer. Marker colour = number of training sentences."""
    pts = [(x, s, c) for x, s, c in zip(xs, S, colors) if s]
    x = [p[0] for p in pts]
    for a, key, ls in ((ax, "same", "-"), (ax, "other", (0, (3, 1.5))), (axk, "lost", "-")):
        a.fill_between(x, [p[1][key][1] for p in pts], [p[1][key][2] for p in pts], color=INK, alpha=0.08, lw=0)
        a.plot(x, [p[1][key][0] for p in pts], color=INK, lw=1.1, ls=ls, zorder=3)
        for xx, s, c in pts:
            hol = key == "same" and xx == hollow_x
            a.plot(xx, s[key][0], "o", ms=3.2, mfc="white" if hol else c, mec=c, mew=0.9, zorder=4)
    axl.axhline(klayer, color=KCOL, lw=2.0, zorder=2)
    ly = [(p[0], p[1]["layer"], p[2]) for p in pts if p[1].get("layer")]
    axl.plot([p[0] for p in ly], [p[1] for p in ly], color=MID, lw=1.0, zorder=3)
    prev = None
    for xx, l, c in ly:
        axl.plot(xx, l, "o", ms=3.2, color=c, zorder=4)
        if round(l) != prev:                                     # a value is written once, under its first dot
            up = 0 < klayer - l < 5
            axl.text(xx, l - 1.3 if up else l + 1.3, f"{l:.0f}", ha="center", va="bottom" if up else "top",
                     fontsize=6, color=INK, zorder=5)
        prev = round(l)
    for a in (ax, axk, axl):
        a.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(-4, 104); ax.set_yticks([0, 50, 100])
    ax.tick_params(labelbottom=False); axk.tick_params(labelbottom=False)
    return pts


def row2(fig, sub, tag, letters, R, show_ylabel=True, d_steps=None, short=False, force=None):
    """sub: a 3 x 2 grid (generalization, known facts lost, answer layer) x ((d) by step, (e) by sentences)."""
    force = force or {}
    kl = st.median(M.k_layers(tag, root=R[0]))
    fix, nl = FIX[tag], M.N_LAYERS[tag]
    ax = fig.add_subplot(sub[0, 0])
    axk = fig.add_subplot(sub[1, 0], sharex=ax)
    axl = fig.add_subplot(sub[2, 0], sharex=ax)
    d_steps = d_steps or D_STEPS
    S = [M.summary(M.step0(tag)) if s == 0 else M.summary(M.runs(R, tag, "5", s)) for s in d_steps]
    pts = _draw(ax, axk, axl, d_steps, S, [BASE["5"]] * len(d_steps), kl)
    ax.set_title(f"$\\bf{{{letters[0]}}}$  S5" + ("" if short else f", {M.TAGS[tag]}"))
    axl.set_xticks([0, 32, 64, 128, 256]); axl.set_xlim(-6, 262)
    axl.set_xlabel("Training steps")
    if pts:
        x_end, s_end, _ = pts[-1]
        ax.text(x_end, s_end["same"][0] + 4, "unseen paraphrases", color=INK, ha="right", va="bottom", fontsize=6.8)
        ax.text(x_end, s_end["other"][0] + 4, "unseen language", color=INK, ha="right", va="bottom", fontsize=6.8)
    if show_ylabel:
        ax.set_ylabel("Generalization (%)")
        axk.set_ylabel("Known facts\nlost (%)", fontsize=7)
        axl.set_ylabel("Answer\nlayer", fontsize=7)
    ax2 = fig.add_subplot(sub[0, 1], sharey=ax)
    axk2 = fig.add_subplot(sub[1, 1], sharex=ax2, sharey=axk)
    axl2 = fig.add_subplot(sub[2, 1], sharex=ax2, sharey=axl)
    xs = [SIZE_X[z] for z in SIZE_X]
    S2 = [M.summary(M.runs(R, tag, z, fix)) for z in SIZE_X]
    _draw(ax2, axk2, axl2, xs, S2, [SIZE_COL.get(z, MID) for z in SIZE_X], kl, hollow_x=SIZE_X["all"])
    ax2.set_title(f"$\\bf{{{letters[1]}}}$  After {fix} training steps" + ("" if short else f", {M.TAGS[tag]}"))
    for a in (ax2, axk2, axl2):
        a.tick_params(labelleft=False)
    axl2.set_xticks(xs, list(SIZE_X)); axl2.set_xlim(-0.2, 4.3)
    axl2.set_xlabel("Training templates")
    top = force.get("lost_top") or max([10] + [s["lost"][2] for s in S + S2 if s]) * 1.1
    axk.set_ylim(top, -1)
    axk.set_yticks([t for t in (0, 10, 20, 30, 40) if t < top],
                   ["0" if t == 0 else f"\u2212{t}" for t in (0, 10, 20, 30, 40) if t < top])
    for a in (axl, axl2):
        lo = min([kl] + [s["layer"] for s in S + S2 if s and s.get("layer")]) - 1.5
        hi = max([s["layer"] for s in S + S2 if s and s.get("layer")] + [kl]) + 5
        lo, hi = force.get("layer", (lo, hi))
        a.set_ylim(hi, lo)                                       # earlier layers at the top
        a.set_yticks(sorted({t for t in range(0, nl, 5) if lo < t < min(hi, nl)}))
    return S, S2


def _print_row2(S, S2, d_steps, prefix=""):
    for s, x in zip(d_steps, S):
        if x:
            print(f"{prefix}(d) step {s:>3}: unseen {x['same'][0]:.0f}% other {x['other'][0]:.0f}% "
                  f"lost {x['lost'][0]:.1f} drop {x['drop'][0]:+.1f} layer {x['layer']}")
    for z, x in zip(SIZE_X, S2):
        if x:
            print(f"{prefix}(e) S{z:<3}: unseen {x['same'][0]:.0f}% other {x['other'][0]:.0f}% "
                  f"lost {x['lost'][0]:.1f} drop {x['drop'][0]:+.1f} layer {x['layer']}")


def fig1():
    with plt.rc_context(STYLE["fig1"]):
        fig = plt.figure(figsize=(5.5, 5.35))
        outer = fig.add_gridspec(2, 1, height_ratios=[1, 1.7], hspace=0.30)
        sub = outer[0].subgridspec(1, 3, wspace=0.22)
        axes = [fig.add_subplot(sub[0, i]) for i in range(3)]
        for n, (ax, (tag, name)) in enumerate(zip(axes, MODELS)):
            cdf_panel(ax, tag, roots())
            ax.set_title(f"$\\bf{{{'abc'[n]}}}$  {name}")
            if n:
                ax.tick_params(labelleft=False)
        axes[0].set_ylabel("Answers visible (%)")
        sub = outer[1].subgridspec(3, 2, height_ratios=[3, 1.1, 1.6], hspace=0.12, wspace=0.08)
        S, S2 = row2(fig, sub, G12, "de", roots())
        fig1_legend(fig, -0.035)
    _print_row2(S, S2, D_STEPS)
    fig.style, fig.savefig_kw = "fig1", dict(bbox_inches="tight", pad_inches=0.01)
    return fig


def fig4():
    """Gemma-3-12B, one row per adapter site: where the answer appears | S5 by step | sentences at 64 steps."""
    with plt.rc_context(STYLE["fig1"]):
        fig = plt.figure(figsize=(7.0, 5.6))
        outer = fig.add_gridspec(2, 1, hspace=0.32)
        force = dict(lost_top=27, layer=(29.5, 52))              # the same strip scales in both rows
        for i, (site, label, letters) in enumerate((("qv", "Query and value", "abc"), ("mlp_out", "MLP output", "def"))):
            R = roots(site)
            sub = outer[i].subgridspec(3, 3, height_ratios=[3, 1.1, 1.6], hspace=0.12, wspace=0.42,
                                       width_ratios=[1, 1, 0.8])
            ax = fig.add_subplot(sub[:, 0])
            cdf_panel(ax, G12, R)
            ax.set_title(f"$\\bf{{{letters[0]}}}$  {label}")
            ax.set_ylabel("Answers visible (%)")
            steps_d = sorted({0, 32, 64, 128, 256, *(EARLY if site == "qv" else EARLY_MLP)})
            S, S2 = row2(fig, sub[:, 1:].subgridspec(3, 2, height_ratios=[3, 1.1, 1.6], hspace=0.12, wspace=0.08),
                         G12, letters[1:], R, short=True, d_steps=steps_d, force=force)
            _print_row2(S, S2, steps_d, f"{site:<8} ")
        fig1_legend(fig, -0.05)
    fig.style, fig.savefig_kw = "fig1", dict(bbox_inches="tight", pad_inches=0.01)
    return fig


# ------------------------------------------------------------------------------------------------ appendix figures
def _axis(ax, nl, three=False):
    lo = nl // 2
    t = [lo, (lo + nl - 1) // 2, nl - 1] if three else [x for x in list(range(0, nl, 8 if nl > 40 else 4)) + [nl - 1] if x >= lo]
    ax.set_xticks(t, [f"L{x}" for x in t]); ax.set_xlim(lo, nl - 1); ax.set_ylim(0, 1.02)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, lw=0.5); ax.set_yticks([0, 0.5, 1], ["0", "50", "100"])


def fig3():
    """Known facts per model, pooled over relations, one curve per asked language."""
    with plt.rc_context(STYLE["appendix"]):
        fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.75), sharey=True)
        for ax, (tag, name) in zip(axes, MODELS):
            n = M.N_LAYERS[tag]
            en = M.k_layers(tag, "EN")
            cdf(ax, en, LCOL["EN"], 1.8)
            for p in M.PAIRS:
                x = M.k_layers(tag, "ZH", pair=p)
                cdf(ax, x, LCOL[p], 1.8)
                print(name, p, "known median", st.median(x), len(x))
            print(name, "EN known median", st.median(en), len(en))
            _axis(ax, n, True); ax.tick_params(labelsize=7); ax.set_title(f"{name} ({n} layers)", fontsize=7.5)
        axes[0].set_ylabel("answers visible (%)", fontsize=8)
        fig.legend(handles=[plt.Line2D([], [], color=LCOL[l], lw=1.8, label=f"asked in {LNAME[l]}") for l in ("EN", "zh", "ru")],
                   loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02), fontsize=7.5)
        fig.tight_layout(w_pad=0.4, rect=(0, 0.1, 1, 1))
    fig.style, fig.savefig_kw = "appendix", dict(bbox_inches="tight", pad_inches=0.03)
    return fig


def fig5():
    """Answer layer under the top-3, top-5 and top-10 rules (Gemma-3-12B, S5, second language, 256 steps)."""
    nl = M.N_LAYERS[G12]
    inj = [M.load(f) for f in M.run_files((str(C.results("main")),), G12, "5", 256)]
    with plt.rc_context(STYLE["appendix"]):
        fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.95), sharey=True)
        for n, (ax, k) in enumerate(zip(axes, (3, 5, 10))):
            kx = M.k_layers(G12, "ZH", k)
            ix = [M.first_layer(r, d["layers"], k) for d in inj for r in d["records"]
                  if r["bucket"] == "ZH" and not r["is_train"] and r["correct"]]
            cdf(ax, kx, KCOL, 2.2); cdf(ax, ix, "#D55E00", 1.6)
            print(f"top {k}: known L{st.median(kx)} ({len(kx)}), injected L{st.median(ix)} ({len(ix)})")
            _axis(ax, nl); ax.set_xlabel(f"Layer (of {nl})"); ax.set_title(f"({'abc'[n]}) Top {k}", loc="left", fontsize=9)
        axes[0].set_ylabel("Answers visible (%)")
        fig.legend(handles=[plt.Line2D([], [], color=KCOL, lw=2.2, label="Known facts"),
                            plt.Line2D([], [], color="#D55E00", lw=1.6, label="Injected, S5 (5 templates), 256 steps")],
                   loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=2, frameon=False, handlelength=1.8, columnspacing=1.5)
        fig.tight_layout(w_pad=0.8, rect=(0, 0.1, 1, 1))
    fig.style, fig.savefig_kw = "appendix", {}
    return fig


# ------------------------------------------------------------------------------------------------ Figure 2
FIG2_CAPTION = [("P19_duan_kexin", "commenced living in"), ("P19_gan_meizhi", "first see the light of day"),
                ("P19_zhuang_fenghui", "was introduced to the world in"),
                ("P19_zhuang_fenghui", "What city claims the birth of")]


def fig2_prompts(tag=G12, lr="2e-05", lang="ZH", size="3", steps=(4, 16, 32, 64, 128, 256), stop=0.25,
                 move=-4, n_move=2, n_flat=2):
    """The selection rule of Figure 2 on the learning-rate sweep (Gemma-3-12B, the 3 Chinese birthplace facts, S3, lr 2e-5):
    steps where every fact keeps its known-fact drop <= 25% (intact); prompts correct from their first correct step to
    the last (tracked); 2 that move >= 4 layers earlier and 2 that stay, longest-tracked first, different people.
    Returns (tracked {(fact, template): {step: layer}}, moving, staying, known median, n_layers)."""
    root = C.results("lr_sweep")
    K = M.load(root / tag / "K.json")
    def files(s):
        return sorted((root / tag / f"lr{lr}" / lang / size / f"steps{s}").glob("*.json"))
    intact = [s for s in steps if len(fs := files(s)) == 3 and max(M.load(f)["drop"] for f in fs) <= stop]
    lay = {}
    for s in intact:
        for f in files(s):
            d = M.load(f)
            for r in d["records"]:
                if r["bucket"] == lang and not r["is_train"]:
                    lay.setdefault((d["fact"], r["idx"]), {})[s] = M.first_layer(r, d["layers"]) if r["correct"] else None
    keep = {}
    for key, v in lay.items():
        ok = [s for s in intact if v.get(s) is not None]
        if len(ok) >= 2 and ok == [s for s in intact if s >= ok[0]]:
            keep[key] = {s: v[s] for s in ok}
    nl = K["n_layers"]
    kmed = st.median(M.first_layer(r, K["layers"]) for r in K["records"] if r["bucket"] == lang and r["correct"])
    change = {k: v[max(v)] - v[min(v)] for k, v in keep.items()}

    def pick(cands, n):
        cands, out = sorted(cands, key=lambda k: (-len(keep[k]), change[k], k[1])), []
        for k in cands:
            if len(out) < n and all(k[0] != o[0] for o in out):
                out.append(k)
        return out + [k for k in cands if k not in out][: n - len(out)]

    inside = [k for k in keep if max(keep[k].values()) < nl - 1]
    mv = pick([k for k in inside if change[k] <= move], n_move)
    fl = pick([k for k in inside if change[k] == 0 and all(keep[k][min(keep[k])] != keep[o][min(keep[o])] for o in mv)],
              n_flat)
    print(f"intact steps {intact}; known facts median L{kmed:g}; {len(keep)} tracked prompts")
    return keep, mv, fl, kmed, nl


def fig2():
    steps = (4, 16, 32, 64, 128, 256)
    keep, move, flat, kmed, nl = fig2_prompts(steps=steps)
    tpl = templates("P19")["en"]
    tix = lambda phrase: next(i for i, t in enumerate(tpl) if phrase in t)
    want = {(f, tix(ph)) for f, ph in FIG2_CAPTION}
    got = set(move + flat)
    print("selected:", sorted(got), "| caption:", sorted(want), "->", "SAME" if got == want else "DIFFERENT")
    assert got == want, "the selection does not give the paper's four prompts"
    MOVE_COL, FLAT_COL = "#D55E00", "#6E6E6E"
    draw = [(k, keep[k], "-", MOVE_COL) for k in move] + [(k, keep[k], (0, (3, 1.6)), FLAT_COL) for k in flat]
    order = {(f, tix(ph)): n for n, (f, ph) in enumerate(FIG2_CAPTION)}
    draw.sort(key=lambda d: order[d[0]])
    marker = {0: "o", 1: "s", 2: "D", 3: "^"}
    X = {s: i for i, s in enumerate(steps)}
    LO, HI = 32, 47
    with plt.rc_context(STYLE["fig2"]):
        fig, ax = plt.subplots(figsize=(3.27, 2.01))
        for t in range(LO, HI, 4):
            ax.axhline(t, color="#e3e3e3", lw=0.5, zorder=0)
        for s in steps:
            ax.axvline(X[s], color="#9a9a9a", lw=0.7, zorder=1)
        ax.axhline(kmed, color=KCOL, lw=1.6, zorder=3)
        XL = len(steps) - 1 + 0.12
        ax.text(XL, kmed, f"known facts\n(median L{kmed:.0f})", color=KCOL, fontsize=8.5, va="center", ha="left", clip_on=False)
        ends = {}
        for n, ((f, idx), v, ls, col) in enumerate(draw):
            ss = sorted(v)
            ax.plot([X[s] for s in ss], [v[s] for s in ss], color=col, lw=1.6, ls=ls, marker=marker[n], ms=4.2, zorder=4,
                    markeredgecolor="white", markeredgewidth=0.5, clip_on=False)
            ends[n] = v[ss[-1]]
        lab = {0: ends[0] + 1.5, 1: ends[1], 2: ends[2] - 0.1, 3: ends[3] + 0.2}
        for n, (_, _, _, col) in enumerate(draw):
            ax.text(XL, lab[n], f"({'abcd'[n]})", ha="left", va="center", fontsize=9, color=col, clip_on=False)
        ax.set_xticks([X[s] for s in steps], [str(s) for s in steps])
        ax.set_yticks([t for t in range(LO, HI, 4)], [f"L{t}" for t in range(LO, HI, 4)])
        ax.tick_params(length=0, labelsize=9)
        ax.tick_params(axis="y", labelcolor="#555")
        ax.set_xlabel("training steps", fontsize=9, labelpad=1)
        ax.set_xlim(-0.3, len(steps) - 1 + 0.05)
        ax.set_ylim(LO - 0.3, HI)
        for sp in ax.spines.values():
            sp.set_visible(False)
        fig.legend(handles=[plt.Line2D([], [], color=MOVE_COL, lw=1.6, label="moves\nearlier"),
                            plt.Line2D([], [], color=FLAT_COL, lw=1.6, ls=(0, (3, 1.6)), label="stays")],
                   loc="lower left", bbox_to_anchor=(0.775, -0.03), frameon=False, fontsize=8.5, handlelength=1.6,
                   handletextpad=0.4, labelspacing=0.5)
        fig.subplots_adjust(left=0.1, right=0.68, top=0.97, bottom=0.17)
    for n, ((f, idx), v, _, _) in enumerate(draw):
        print(f"({'abcd'[n]}) {f} template {idx} {tpl[idx]!r}: " + ", ".join(f"{s}:L{v[s]}" for s in sorted(v)))
    fig.style, fig.savefig_kw = "fig2", {}
    return fig


# ------------------------------------------------------------------------------------------------ saving
def save(fig, name, out_dir=None):
    """<out_dir>/<name>.pdf (+ png/<name>.png), with the style and keyword arguments of the paper's PDFs."""
    out_dir = Path(out_dir or C.FIGURES)
    (out_dir / "png").mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE[fig.style]):
        fig.savefig(out_dir / f"{name}.pdf", **fig.savefig_kw)
        fig.savefig(out_dir / "png" / f"{name}.png", dpi=200, **fig.savefig_kw)
    pdf = out_dir / f"{name}.pdf"
    return pdf.relative_to(C.ROOT) if pdf.is_relative_to(C.ROOT) else pdf
