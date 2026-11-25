"""Every table of the paper (and the appendix numbers that come with them), from the result files. Each function
returns its rows as text and prints nothing.

    table1()        Table 1 (App. A.1): templates per relation, first template in three languages
    table2()        Table 2 (App. A.2): the invented people
    table3()        Table 3 (App. A.3): known facts, % correct in English / second language (Gemma-3-12B)
    table4()        Table 4 (App. C.4): agreement between J-lenses (top 3, JS)
    lens_refit()    App. C.4 test 1: answer layers under the refitted lenses
    table6()        Table 6 (App. C.6): known facts on the fine-tuned model
    forgetting()    App. B.2: known facts lost after 256 steps, per model and size
    lr_sweep()      App. B.2: the learning-rate sweep (runs that keep their known facts, drop per rate)
Table 5 (App. C.5) is section 12 of numbers.model_numbers (the paper's version).
"""
import json
import re
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

from scipy.stats import mannwhitneyu

from .. import config as C
from ..data import people as injected_people, templates
from . import metrics as M

MODELS = [("Qwen2.5-7B-Instruct", "Qwen2.5-7B"), ("gemma-3-12b-it", "Gemma-3-12B"), ("gemma-3-27b-it", "Gemma-3-27B")]


def _json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def table1():
    rows = ["relation | templates | English | Chinese | Russian"]
    for rel in C.RELATIONS:
        t = templates(rel)
        rows.append(f"{C.REL_NAME[rel]} | {len(t['en'])} | {t['en'][0]} | {t['zh'][0]} | {t['ru'][0]}")
    return "\n".join(rows)


def table2():
    people = injected_people()
    return json.dumps(people, ensure_ascii=False, indent=1)


def table3(tag="gemma-3-12b-it"):
    """Rows: pair, relation, person, city, EN correct / n, second language correct / n (untouched model, K.json)."""
    out = []
    for p in M.PAIRS:
        for r in M.RELS:
            d = M.k_files(tag)[(p, r)]
            for person in sorted({x["subject"] for x in d["records"]}):
                rec = [x for x in d["records"] if x["subject"] == person]
                c = {b: sum(x["correct"] for x in rec if x["bucket"] == b) for b in ("EN", "ZH")}
                n = sum(x["bucket"] == "EN" for x in rec)
                city = next(x["gold"][0] for x in rec if x["bucket"] == "EN")
                out.append(f" & {C.REL_NAME[r]} & {person} & {city} & {c['EN']}/{n} ({100 * c['EN'] / n:.0f}\\%) & "
                           f"{c['ZH']}/{n} ({100 * c['ZH'] / n:.0f}\\%) \\\\  % {p}")
    return "\n".join(out)


def table4():
    """Mean over the 47 layers and the inputs of each set: % of the top 3 in common, Jensen-Shannon divergence (bits)."""
    d = _json(C.results("lens_dist") / "summary_n400.json")["pairs"]
    sets = [("wiki", "WikiText"), ("ZH", "Chinese"), ("EN", "English")]
    rows = ["lenses | " + " | ".join(f"{n} top 3 | {n} JS" for _, n in sets)]
    for a, b in (("neuronpedia", "base"), ("base", "ft"), ("neuronpedia", "ft")):
        cells = []
        for s, _ in sets:
            m = d[s][f"{a}|{b}"]["mean"]
            cells += [f"{100 * m['top3']:.0f}", f"{m['js']:.2f}"]
        rows.append(f"{a} vs {b} | " + " | ".join(cells))
    return "\n".join(rows)


def lens_refit(n_fit=400, k=3):
    """App. C.4 test 1: known (Chinese, untouched model) and injected (unseen Chinese, adapter on) answer layers read by
    the neuronpedia lens and by our refitted lenses (base: untouched model, ft: with the adapter)."""
    d = _json(C.results("lens_refit") / f"readout_n{n_fit}.json")
    L = d["layers"]
    first = lambda ranks: next((L[i] for i, r in enumerate(ranks) if r <= k), L[-1])
    ks = [i for i, r in enumerate(d["K"]["records"]) if r["bucket"] == "ZH" and r["correct"]]
    js = [i for i, r in enumerate(d["inj"]["records"]) if r["bucket"] == "ZH" and not r["is_train"] and r["correct"]]
    K = {l: [first(d["K"][l][i]["gt_rank"]) for i in ks] for l in ("neuronpedia", "base")}
    I = {l: [first(d["inj"][l][i]["gt_rank"]) for i in js] for l in ("neuronpedia", "base", "ft")}
    ch = Counter((y > x) - (y < x) for x, y in zip(I["neuronpedia"], I["ft"]))
    rows = [f"known prompts (Chinese, correct): {len(ks)}; injected prompts (unseen Chinese, correct): {len(js)}",
            f"mean answer layer, neuronpedia lens: known {st.mean(K['neuronpedia']):.2f}, injected "
            f"{st.mean(I['neuronpedia']):.2f}, gap {st.mean(I['neuronpedia']) - st.mean(K['neuronpedia']):.1f} layers",
            f"mean answer layer, refitted lenses: known (base) {st.mean(K['base']):.2f}, injected (ft) "
            f"{st.mean(I['ft']):.2f}, gap {st.mean(I['ft']) - st.mean(K['base']):.1f} layers",
            f"injected prompts, neuronpedia -> ft: same {ch[0]}, earlier {ch[-1]}, later {ch[1]}, "
            f"max shift {max(abs(y - x) for x, y in zip(I['neuronpedia'], I['ft']))} layers",
            f"Mann-Whitney injected(ft) > known(base): p = {mannwhitneyu(I['ft'], K['base'], alternative='greater').pvalue:.1e}"]
    return "\n".join(rows)


def table6(sizes=("1", "5", "all")):
    """Median answer layer, trained and asked in the second language, 256 steps: injected facts (unseen templates, the
    main run), and the known-fact prompts each adapter still answers, read on the untouched model and with the adapter
    on. Rows: model, size, adapters, injected, known untouched, gap, known adapter on, gap."""
    root = C.results("known_ft")
    known = defaultdict(lambda: ([], [], 0))
    for p in root.glob("*_*/*/*/*/*.json"):
        tag, lang, size = p.parts[-4], p.parts[-3], p.parts[-2]
        if lang == "EN":
            continue
        d = _json(p)
        if "known_adapted" not in d:
            continue
        u, a, n = known[(tag, size)]
        L = d["known_layers"]
        for r in d["known_adapted"]:
            u.append(M.first_layer({"gt_rank": r["gt_rank_untouched"]}, L))
            a.append(M.first_layer(r, L))
        known[(tag, size)] = (u, a, n + 1)
    rows = ["model | templates | adapters | injected | known untouched | gap | known adapter on | gap"]
    R = (str(C.results("main")),)
    for tag, name in MODELS:
        for size in sizes:
            inj = st.median([x for r in M.runs(R, tag, size, 256) for x in r["layers"]])
            u, a, n = known.get((tag, size), ([], [], 0))
            if not u:
                rows.append(f"{name} ({M.N_LAYERS[tag]}) | S{size} | 0 | L{inj:g} | missing | | missing |")
                continue
            rows.append(f"{name} ({M.N_LAYERS[tag]}) | {'S' + size if size != 'all' else 'all'} | {n} | L{inj:g} | "
                        f"L{st.median(u):g} | {inj - st.median(u):+g} | L{st.median(a):g} | {inj - st.median(a):+g}")
    return "\n".join(rows)


def forgetting(step=256):
    """App. B.2: known facts lost (% of the known-fact prompts, mean over the facts), trained in the second language,
    after `step` steps, per model and training size."""
    from .plots import roots
    rows = ["model | S1 | S5 | all"]
    for tag, name in MODELS:
        rows.append(f"{name} | " + " | ".join(f"{st.mean(r['lost'] for r in M.runs(roots(), tag, z, step)):.1f}"
                                              for z in ("1", "5", "all")))
    return "\n".join(rows)


def lr_sweep(sizes=("1", "3", "86")):
    """App. B.2: the learning-rate sweep. Per model, learning rate, training language and size (1, 3 or all 86
    templates), over the 3 facts: how many runs reach 256 steps without losing more than 25% of the known facts (the
    stop rule), and the share of known facts lost (drop, %) at the last step each run reached (mean)."""
    root = C.results("lr_sweep")
    rows = ["model | lr | trained in | " + " | ".join(f"S{s}: reached 256 / drop at last step" for s in sizes)]
    for tag in [m.split("/")[-1] for m in C.SWEEP_MODELS]:
        lrs = sorted({p.name for p in (root / tag).glob("lr*")}, key=lambda x: float(x[2:]))
        for lr in lrs:
            for lang in ("ZH", "EN"):
                cells = []
                for s in sizes:
                    last = []
                    for fid in sorted({p.stem for p in (root / tag / lr / lang / s).glob("steps*/*.json")}):
                        files = sorted((root / tag / lr / lang / s).glob(f"steps*/{fid}.json"),
                                       key=lambda p: int(p.parent.name[5:]))
                        last.append((int(files[-1].parent.name[5:]), _json(files[-1])["drop"]))
                    cells.append(f"{sum(st_ == 256 for st_, _ in last)}/{len(last)} / {100 * st.mean(d for _, d in last):.0f}%"
                                 if last else "-")
                rows.append(f"{tag} | {lr[2:]} | {lang} | " + " | ".join(cells))
    return "\n".join(rows)
