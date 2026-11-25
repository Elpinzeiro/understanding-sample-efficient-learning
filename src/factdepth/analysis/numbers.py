"""Every number of the per-model experiment summaries (summary_<Model>.md), recomputed from the result files.

    from factdepth.analysis.numbers import model_numbers
    print(model_numbers("gemma-3-12b-it"))     # the markdown report (the paper's numbers_<model>.md)

Reads only results/<pair>_<rel>/<model>/ (K.json and lr2e-05/<lang>/<size>/steps<N>/<fact>.json) and the untouched
model's answers (data/known_facts/<model>/). Rules (the paper's):
  answer layer   first J-lens layer whose rank of the gold's first token is <= 3 (the last entry is the model's real
                 output; a prompt never in the top 3 gets that last layer); computed on correctly answered prompts only
  same language  test prompts in the training language, minus the trained templates (with "all" nothing is left out)
  other language every template of the other language (none is ever trained)
  drop_pp        % of the known-fact prompts correct before - after (both languages), as stored by run.py
  intact         the paper's rule per run: drop ratio (1 - after/before) <= 25%
  hijack         known-fact prompt whose answer's first line names the INJECTED city (pattern match, not judged),
                 counted only on known people whose own city is another one
Works on partial results too (a model still running, or runs that crashed): N§0 lists the missing runs.
Sections 11-14 (training speed, readout by script, other-language answers, output-only answers) were added for the
Qwen2.5-7B summary; 15-16 (known-fact flips behind the drop, script of the other-language answers) for Gemma-3-27B.
"""
import glob
import json
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

from scipy.stats import mannwhitneyu

from .. import config as C
from ..data import Unit
from ..matching import first_line_names_city
from .metrics import first_layer, load


def model_numbers(TAG, RES=None):
    """Every number of one model (sections 0-16), as markdown. RES: the main run's results (default: the paper's)."""
    RES = Path(RES or C.results("main"))
    PAIRS, RELS, SIZES, STEPS = ["zh", "ru"], ["P19", "P20", "P937"], ["1", "5", "all"], [32, 64, 128, 256]
    REL_NAME = {"P19": "born in", "P20": "died in", "P937": "worked in"}
    SECOND = {"zh": "Chinese", "ru": "Russian"}
    OUT = []


    def p(line=""):
        OUT.append(line)



    def med(x):
        return f"L{st.median(x):g}" if x else "-"


    def pct(a, b):
        return f"{100 * a / b:.0f}%" if b else "-"


    # ------------------------------------------------------------------ loading
    RUNS = {}                                    # (pair, rel, tl, size, step, fact) -> result dict (tl: "2nd" or "EN")
    for f in RES.glob(f"*_*/{TAG}/lr2e-05/*/*/steps*/*.json"):
        pr, _, _, lang, size, steps, _ = f.relative_to(RES).parts
        pair, rel = pr.split("_")
        d = load(f)
        RUNS[(pair, rel, "EN" if lang == "EN" else "2nd", size, int(steps[5:]), d["fact"])] = d
    K = {}
    for pair in PAIRS:
        for rel in RELS:
            f = RES / f"{pair}_{rel}" / TAG / "K.json"
            if f.exists():
                K[(pair, rel)] = load(f)


    def test_recs(d, tl, size, where):
        """where = "same" (training language) or "other"; bucket "ZH" holds the second language (Russian in ru)."""
        train_bucket = "EN" if tl == "EN" else "ZH"
        b = train_bucket if where == "same" else ("ZH" if train_bucket == "EN" else "EN")
        return [r for r in d["records"] if r["bucket"] == b and (where == "other" or size == "all" or not r["is_train"])]


    def runs(pair=None, rel=None, tl=None, size=None, step=None):
        return [(k, d) for k, d in RUNS.items() if (pair is None or k[0] == pair) and (rel is None or k[1] == rel)
                and (tl is None or k[2] == tl) and (size is None or k[3] == size) and (step is None or k[4] == step)]


    def k_layers(pair=None, rel=None, bucket="ZH"):
        out, n = [], 0
        for (pp, rr), d in K.items():
            if (pair is None or pp == pair) and (rel is None or rr == rel):
                rs = [r for r in d["records"] if r["bucket"] == bucket]
                n += len(rs)
                out += [first_layer(r, d["layers"]) for r in rs if r["correct"]]
        return out, n


    def inj(sel, tl, size, where):
        """(correct layers, asked) pooled over the selected runs."""
        lay, n = [], 0
        for _, d in sel:
            rs = test_recs(d, tl, size, where)
            n += len(rs)
            lay += [first_layer(r, d["layers"]) for r in rs if r["correct"]]
        return lay, n


    # ------------------------------------------------------------------ 0. completeness
    p(f"# Numbers for {TAG} (recomputed by analysis/summary_numbers.py)")
    p()
    p("## 0. Completeness")
    n_files = len(RUNS)
    p(f"- result files: {n_files} (expected 180 runs x 4 checkpoints = 720); K files: {len(K)} (expected 6)")
    have = {(k[0], k[1], k[2], k[3], k[5]) for k in RUNS if k[4] == 256}
    facts_of = defaultdict(set)
    for k in RUNS:
        facts_of[(k[0], k[1])].add(k[5])
    miss = [f"{pr}_{fid} {tl} {sz}" for (pr, rl), fs in sorted(facts_of.items()) for fid in sorted(fs) for tl in ["2nd", "EN"]
            for sz in SIZES if (pr, rl, tl, sz, fid) not in have]
    p(f"- runs complete to 256 steps: {len(have)} of 180; missing among the facts started: {len(miss)}"
      + (": " + ", ".join(miss) if miss else ""))
    pend = sum(r.get("judge") == "PENDING" for d in RUNS.values() for r in d["records"] + d.get("known_after", []))
    p(f"- PENDING answers: {pend}")
    routes = defaultdict(int)
    for d in RUNS.values():
        for r in d["records"]:
            routes[(r.get("judge_route"), r.get("judge"))] += 1
    tot = sum(routes.values())
    p(f"- judge routes over all {tot} injected-fact answers: " + ", ".join(f"{k[0]}/{k[1]} {v} ({100*v/tot:.1f}%)"
                                                                           for k, v in sorted(routes.items(), key=lambda x: -x[1])))
    usd = sum(d.get("judge_usd", 0) for d in RUNS.values())
    p(f"- judge cost stored in the files (injected-fact answers): ${usd:.2f}")
    p()

    # ------------------------------------------------------------------ 1. before injection (dataset)
    p("## 1. Before injection (untouched model, dataset before_<model>.json, judge v5)")
    p()
    p("| pair | relation | invented people, EN correct | invented, 2nd lang correct | known people (city): EN / 2nd lang correct of templates |")
    p("|---|---|---|---|---|")
    for pair in PAIRS:
        for rel in RELS:
            unit = Unit(pair, rel)
            b = unit.before(TAG)["records"]
            picked = unit.before(TAG)["known"]
            ie = [r for r in b if r["set"] == "injected" and r["bucket"] == "EN"]
            iz = [r for r in b if r["set"] == "injected" and r["bucket"] == "ZH"]
            kn = []
            for fid in picked:
                fact = unit.fact(fid, TAG)
                rs = [r for r in b if r["set"] == "known" and r.get("fact") == fid]
                kn.append(f"{fact['subject']} ({fact['answer']}): {sum(r['correct'] for r in rs if r['bucket']=='EN')}"
                          f" / {sum(r['correct'] for r in rs if r['bucket']=='ZH')} of {len(fact['EN'])}")
            p(f"| {pair} | {REL_NAME[rel]} | {sum(r['correct'] for r in ie)}/{len(ie)} | {sum(r['correct'] for r in iz)}/{len(iz)} | {'; '.join(kn)} |")
    p()

    # ------------------------------------------------------------------ 2. accuracy grid
    p("## 2. Accuracy: % correct (correct/asked), all 30 facts pooled unless split")
    for where, title in [("same", "same language as training, unseen templates (Q1)"),
                         ("other", "other language, all templates (Q2)")]:
        p()
        p(f"### 2.{1 if where == 'same' else 2} {title}")
        p()
        p("| trained in | size | " + " | ".join(f"{s} steps" for s in STEPS) + " | 256: zh pair | 256: ru pair | 256: per-fact min-max |")
        p("|---|---|" + "---|" * (len(STEPS) + 3))
        for tl in ["2nd", "EN"]:
            for size in SIZES:
                cells = []
                for s in STEPS:
                    lay, n = inj(runs(tl=tl, size=size, step=s), tl, size, where)
                    cells.append(f"{pct(len(lay), n)} ({len(lay)}/{n})")
                pairs = []
                for pair in PAIRS:
                    lay, n = inj(runs(pair=pair, tl=tl, size=size, step=256), tl, size, where)
                    pairs.append(f"{pct(len(lay), n)}")
                per = []
                for k, d in runs(tl=tl, size=size, step=256):
                    rs = test_recs(d, tl, size, where)
                    per.append(100 * sum(r["correct"] for r in rs) / len(rs))
                rng = f"{min(per):.0f}-{max(per):.0f}%" if per else "-"
                p(f"| {tl} | {size} | " + " | ".join(cells) + f" | {pairs[0]} | {pairs[1]} | {rng} |")
    p()
    p("Known facts on the untouched model, for comparison (same templates, K.json): ")
    for pair in PAIRS:
        for b in ["EN", "ZH"]:
            lay, n = k_layers(pair=pair, bucket=b)
            p(f"- {pair} pair, asked in {'English' if b == 'EN' else SECOND[pair]}: {pct(len(lay), n)} ({len(lay)}/{n})")
    p()

    # ------------------------------------------------------------------ 2.3 per relation
    p("### 2.3 Per relation, 256 steps, % correct")
    p()
    p("| trained in | size | test | " + " | ".join(f"{pair} {REL_NAME[r]}" for pair in PAIRS for r in RELS) + " |")
    p("|---|---|---|" + "---|" * 6)
    for tl in ["2nd", "EN"]:
        for size in SIZES:
            for where in ["same", "other"]:
                cells = []
                for pair in PAIRS:
                    for rel in RELS:
                        lay, n = inj(runs(pair=pair, rel=rel, tl=tl, size=size, step=256), tl, size, where)
                        cells.append(pct(len(lay), n))
                p(f"| {tl} | {size} | {where} | " + " | ".join(cells) + " |")
    p()

    # ------------------------------------------------------------------ 3. per-fact other-language (London question)
    p("## 3. Per fact: other-language transfer at 256 steps (% of the other-language templates correct)")
    p()
    for tl in ["EN", "2nd"]:
        p(f"### trained in {tl}, size 5 (and size 1 in brackets)")
        p()
        p("| person (city) | " + " | ".join(REL_NAME[r] for r in RELS) + " |")
        p("|---|---|---|---|")
        facts = sorted({(k[0], k[5].split("_", 1)[1]) for k in RUNS})
        for pair, who in facts:
            cells, city = [], ""
            for rel in RELS:
                vals = []
                for size in ["5", "1"]:
                    d = RUNS.get((pair, rel, tl, size, 256, f"{rel}_{who}"))
                    if d:
                        rs = test_recs(d, tl, size, "other")
                        vals.append(f"{100 * sum(r['correct'] for r in rs) / len(rs):.0f}")
                        city = d["records"][0]["gold"][0]
                cells.append(f"{vals[0]} ({vals[1]})" if len(vals) == 2 else "-")
            p(f"| {who} [{pair}] ({city}) | " + " | ".join(cells) + " |")
        p()
    by_city = defaultdict(list)
    for k, d in runs(tl="EN", size="5", step=256):
        rs = test_recs(d, "EN", "5", "other")
        by_city[d["train"]["pairs"][0]["target"].strip()].append(100 * sum(r["correct"] for r in rs) / len(rs))
    p("Mean over the 6 facts of each city (2 people x 3 relations), trained in EN, size 5, 256 steps, other language: "
      + ", ".join(f"{c} {st.mean(v):.0f}%" for c, v in sorted(by_city.items(), key=lambda x: -st.mean(x[1]))))
    p()

    # ------------------------------------------------------------------ 4. answer layers
    p("## 4. Answer layer (median over correctly answered prompts; L = layer index, 0-based)")
    nl = next(iter(K.values()))["n_layers"]
    p(f"Model has {nl} layers; J-lens layers read: {next(iter(K.values()))['layers'][:3]}...{next(iter(K.values()))['layers'][-3:]} (last = real output)")
    p()
    p("### 4.1 Known facts (K), untouched model")
    p()
    p("| pair | asked in | " + " | ".join(REL_NAME[r] for r in RELS) + " | pooled |")
    p("|---|---|---|---|---|---|")
    for pair in PAIRS:
        for b in ["EN", "ZH"]:
            cells = [med(k_layers(pair=pair, rel=r, bucket=b)[0]) for r in RELS]
            lay, n = k_layers(pair=pair, bucket=b)
            p(f"| {pair} | {'English' if b == 'EN' else SECOND[pair]} | " + " | ".join(cells) + f" | {med(lay)} (n={len(lay)}) |")
    lay, _ = k_layers(bucket="ZH")
    p(f"| both | 2nd language | | | | {med(lay)} (n={len(lay)}) |")
    lay, _ = k_layers(bucket="EN")
    p(f"| both | English | | | | {med(lay)} (n={len(lay)}) |")
    p()
    p("### 4.2 Injected facts vs K, same language as training (unseen templates), by steps")
    p()
    p("Figure 1 setting = trained in 2nd language, tested in 2nd language. Cell: median layer (n correct). "
      "K is the known facts asked in the same test language, same pair(s).")
    p()
    for tl, kb in [("2nd", "ZH"), ("EN", "EN")]:
        for pair in [None] + PAIRS:
            kl, _ = k_layers(pair=pair, bucket=kb)
            p(f"**trained and tested in {tl if tl == 'EN' else ('2nd language' if pair is None else SECOND[pair])}, "
              f"{'both pairs' if pair is None else pair + ' pair'}: K {med(kl)} (n={len(kl)})**")
            p()
            p("| size | " + " | ".join(f"{s}" for s in STEPS) + " | gap at 256 (inj - K) | Mann-Whitney p (256 vs K) |")
            p("|---|" + "---|" * (len(STEPS) + 2))
            for size in SIZES:
                cells, last = [], None
                for s in STEPS:
                    lay, n = inj(runs(pair=pair, tl=tl, size=size, step=s), tl, size, "same")
                    cells.append(f"{med(lay)} ({len(lay)})")
                    last = lay
                gap = st.median(last) - st.median(kl) if last and kl else None
                pv = mannwhitneyu(last, kl).pvalue if len(last) > 5 else None
                p(f"| {size} | " + " | ".join(cells) + f" | {gap:+g} | {pv:.1e} |" if gap is not None else f"| {size} | " + " | ".join(cells) + " | - | - |")
            p()
    p("### 4.3 Injected facts vs K, per relation, 256 steps, trained and tested in the same language")
    p()
    p("| trained+tested | pair | relation | K | S1 | S5 | all |")
    p("|---|---|---|---|---|---|---|")
    for tl, kb in [("2nd", "ZH"), ("EN", "EN")]:
        for pair in PAIRS:
            for rel in RELS:
                kl, _ = k_layers(pair=pair, rel=rel, bucket=kb)
                cells = []
                for size in SIZES:
                    lay, n = inj(runs(pair=pair, rel=rel, tl=tl, size=size, step=256), tl, size, "same")
                    cells.append(f"{med(lay)} ({len(lay)})")
                p(f"| {tl} | {pair} | {REL_NAME[rel]} | {med(kl)} ({len(kl)}) | " + " | ".join(cells) + " |")
    p()
    p("### 4.4 Answer layer in the OTHER language (cross-lingual answers), 256 steps")
    p()
    p("| trained in | pair | K in test language | S1 | S5 | all |")
    p("|---|---|---|---|---|---|")
    for tl in ["2nd", "EN"]:
        for pair in PAIRS:
            kb = "EN" if tl == "2nd" else "ZH"
            kl, _ = k_layers(pair=pair, bucket=kb)
            cells = []
            for size in SIZES:
                lay, n = inj(runs(pair=pair, tl=tl, size=size, step=256), tl, size, "other")
                cells.append(f"{med(lay)} ({len(lay)})")
            p(f"| {tl} | {pair} | {med(kl)} ({len(kl)}) | " + " | ".join(cells) + " |")
    p()

    # ------------------------------------------------------------------ 5. single prompts across steps (Figure 2 rule)
    p("## 5. Single prompts across training (the paper's Figure 2 rule)")
    p()
    p("A prompt is tracked if, once answered correctly, it stays correct at every later checkpoint (at least 2 checkpoints). "
      "Change = answer layer at its last checkpoint (256) - at its first correct checkpoint. Same language as training, unseen templates.")
    p()
    p("| trained+tested | size | pair | tracked | earlier | earlier by >= 4 | same | later | median change |")
    p("|---|---|---|---|---|---|---|---|---|")
    for tl in ["2nd", "EN"]:
        for size in ["1", "5"]:
            for pair in [None] + PAIRS:
                lay = defaultdict(dict)
                for (pp, rel, t, sz, s, fid), d in runs(pair=pair, tl=tl, size=size):
                    for r in test_recs(d, tl, size, "same"):
                        lay[(pp, rel, fid, r["idx"])][s] = first_layer(r, d["layers"]) if r["correct"] else None
                ch = []
                for v in lay.values():
                    ok = [s for s in STEPS if v.get(s) is not None]
                    if len(ok) >= 2 and ok == [s for s in STEPS if s >= ok[0]]:
                        ch.append(v[ok[-1]] - v[ok[0]])
                if not ch:
                    continue
                p(f"| {tl} | {size} | {pair or 'both'} | {len(ch)} | {sum(c < 0 for c in ch)} | {sum(c <= -4 for c in ch)} | "
                  f"{sum(c == 0 for c in ch)} | {sum(c > 0 for c in ch)} | {st.median(ch):+g} |")
    p()

    # ------------------------------------------------------------------ 6. forgetting
    CYR = re.compile(r"[Ѐ-ӿ]")


    def names_city(answer, names):
        line = next((l for l in answer.strip().split("\n") if l.strip()), "")
        for n in names:
            if re.search(r"[A-Za-z]", n):
                if re.search(r"\b" + re.escape(n) + r"\b", line):
                    return True
            elif n and n in line:                         # Chinese, or Russian (case endings only add letters)
                return True
        return False


    p("## 6. Forgetting: drop of the known facts, in percentage points (before - after)")
    p()
    p("| trained in | size | steps | median | mean | max | runs > 10 pp | runs > 20 pp | intact (ratio <= 25%) | hijack: known prompts naming the injected city (pattern) |")
    p("|---|---|---|---|---|---|---|---|---|---|")
    for tl in ["2nd", "EN"]:
        for size in SIZES:
            for s in STEPS:
                sel = runs(tl=tl, size=size, step=s)
                dp = [d["drop_pp"] for _, d in sel]
                if not dp:
                    continue
                intact = sum(d["drop"] <= 0.25 for _, d in sel)
                hj = n = 0
                for _, d in sel:
                    inj_names = d["records"][0]["gold"]
                    for r in d["known_after"]:
                        if any(g in inj_names for g in r["gold"]):        # known person with the same city: skip
                            continue
                        n += 1
                        hj += names_city(r["answer"], inj_names)
                p(f"| {tl} | {size} | {s} | {st.median(dp):+.1f} | {st.mean(dp):+.1f} | {max(dp):.1f} | "
                  f"{sum(x > 10 for x in dp)}/{len(dp)} | {sum(x > 20 for x in dp)}/{len(dp)} | {intact}/{len(dp)} | {pct(hj, n)} ({hj}/{n}) |")
    p()
    p("### 6.1 Runs with drop > 10 pp at 256 steps")
    p()
    p("| trained in | size | fact | before % | after % | drop pp | hijack % of its known prompts |")
    p("|---|---|---|---|---|---|---|")
    rows = []
    for (pair, rel, tl, size, s, fid), d in runs(step=256):
        if d["drop_pp"] > 10:
            inj_names = d["records"][0]["gold"]
            ks = [r for r in d["known_after"] if not any(g in inj_names for g in r["gold"])]
            h = sum(names_city(r["answer"], inj_names) for r in ks)
            rows.append((tl, size, f"{pair}_{fid}", d["known_before_pct"], d["known_after_pct"], d["drop_pp"], pct(h, len(ks))))
    for r in sorted(rows, key=lambda x: (x[0], x[1], -x[5])):
        p(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]:.0f} | {r[4]:.0f} | {r[5]:.1f} | {r[6]} |")
    p()

    # ------------------------------------------------------------------ 7. intact-only sensitivity for the headline
    p("## 7. Sensitivity: headline numbers with only intact runs (drop ratio <= 25%), 256 steps")
    p()
    p("| trained in | size | test | all runs: % correct, median layer | intact runs only: n runs, % correct, median layer |")
    p("|---|---|---|---|---|")
    for tl in ["2nd", "EN"]:
        for size in SIZES:
            for where in ["same", "other"]:
                a, na = inj(runs(tl=tl, size=size, step=256), tl, size, where)
                sel = [(k, d) for k, d in runs(tl=tl, size=size, step=256) if d["drop"] <= 0.25]
                b, nb = inj(sel, tl, size, where)
                p(f"| {tl} | {size} | {where} | {pct(len(a), na)}, {med(a)} | {len(sel)}, {pct(len(b), nb)}, {med(b)} |")
    p()


    # ------------------------------------------------------------------ 8. robustness: any written form of the city
    def norm(t):
        return t.strip().lower()


    def any_form_layer(r, layers):
        """Rule layer, or earlier if one of the layer's top-3 tokens is the start of any gold name in any form (no leading
        space, other case: e.g. "London" without the space the rule asks for). Uses the stored top-10 lists."""
        gold = [norm(g) for g in r["gold"] if g]
        rule = first_layer(r, layers)
        for i, top in enumerate(r["top10"]):
            if layers[i] >= rule:
                break
            for t in top[:3]:
                n = norm(t)
                cjk = re.search(r"[\u3400-\u9fff]", n)
                if (n in gold or len(n) >= (2 if cjk else 3)) and any(g.startswith(n) for g in gold):
                    return layers[i]
        return rule


    p("## 8. Robustness: answer layer when any written form of the city counts")
    p()
    p("Rule of the paper: the first token of the city with a leading space (English, Russian) or without (Chinese) must be in "
      "the top 3. Here also any top-3 token that starts any accepted name (at least 3 letters or 2 Chinese characters, or the whole name), in any case, with or without the space. "
      "256 steps, same language as training, unseen templates.")
    p()
    p("| setting | K: rule / any form | S1: rule / any form | S5: rule / any form | all: rule / any form |")
    p("|---|---|---|---|---|")
    for tl, kb in [("2nd", "ZH"), ("EN", "EN")]:
        for pair in PAIRS:
            kr, ka = [], []
            for (pp, rr), d in K.items():
                if pp == pair:
                    for r in d["records"]:
                        if r["bucket"] == kb and r["correct"]:
                            kr.append(first_layer(r, d["layers"])); ka.append(any_form_layer(r, d["layers"]))
            cells = [f"{med(kr)} / {med(ka)}"]
            for size in SIZES:
                a, b = [], []
                for _, d in runs(pair=pair, tl=tl, size=size, step=256):
                    for r in test_recs(d, tl, size, "same"):
                        if r["correct"]:
                            a.append(first_layer(r, d["layers"])); b.append(any_form_layer(r, d["layers"]))
                cells.append(f"{med(a)} / {med(b)}")
            name = "English" if tl == "EN" else SECOND[pair]
            p(f"| trained+tested {name}, {pair} pair | " + " | ".join(cells) + " |")
    p()

    # ------------------------------------------------------------------ 9. is the other-language transfer about the injected person
    p("## 9. Are the other-language answers specific to the injected person?")
    p()
    p("For each run at 256 steps: % of the other-language prompts about the injected person answered with its city (judged), "
      "next to % of the known-fact prompts IN THAT SAME LANGUAGE whose first answer line names the injected city "
      "(pattern, known people with another city only). If the second is much smaller, the transfer is about the person, "
      "not a city given to everyone.")
    p()
    p("| trained in | size | runs | other-language correct, pooled | known prompts in that language naming the injected city | runs with >= 20% transfer: transfer / leak (mean) |")
    p("|---|---|---|---|---|---|")
    for tl in ["EN", "2nd"]:
        for size in SIZES:
            sel = runs(tl=tl, size=size, step=256)
            if not sel:
                continue
            ob = "ZH" if tl == "EN" else "EN"
            tc = tn = hc = hn = 0
            good_t, good_h = [], []
            for _, d in sel:
                rs = test_recs(d, tl, size, "other")
                c = sum(r["correct"] for r in rs)
                tc, tn = tc + c, tn + len(rs)
                names = d["records"][0]["gold"]
                ks = [r for r in d["known_after"] if r["bucket"] == ob and not any(g in names for g in r["gold"])]
                h = sum(names_city(r["answer"], names) for r in ks)
                hc, hn = hc + h, hn + len(ks)
                if c / len(rs) >= 0.2:
                    good_t.append(100 * c / len(rs)); good_h.append(100 * h / len(ks))
            last = f"{len(good_t)} runs: {st.mean(good_t):.0f}% / {st.mean(good_h):.0f}%" if good_t else "-"
            p(f"| {tl} | {size} | {len(sel)} | {pct(tc, tn)} | {pct(hc, hn)} | {last} |")
    p()

    # ------------------------------------------------------------------ 10. template form: question vs incomplete sentence
    def is_question(prompt):
        stem = prompt.split(" (")[0].split("（")[0].strip()
        return stem.endswith("?") or stem.endswith("？")


    p("## 10. Template form: questions vs incomplete sentences (same language as training, unseen templates, 256 steps)")
    p()
    p("A template is a question (\"Where was [X] born?\") or an incomplete sentence the city should end (\"[X] was born in the "
      "city of\"). The trained templates are the first 1 or 5 of the list; their form is given. Cell: % correct (n asked).")
    p()
    p("| pair | relation | trained in | templates (questions / sentences) | trained form S1; S5 | S1: questions / sentences | S5: questions / sentences |")
    p("|---|---|---|---|---|---|---|")
    for pair in PAIRS:
        for rel in RELS:
            for tl in ["2nd", "EN"]:
                cells, forms, n_tpl = [], [], ""
                for size in ["1", "5"]:
                    q, s_ = [0, 0], [0, 0]
                    sel = runs(pair=pair, rel=rel, tl=tl, size=size, step=256)
                    if not sel:
                        cells.append("-"); forms.append("-")
                        continue
                    d0 = sel[0][1]
                    forms.append("".join("Q" if is_question(x["prompt"]) else "S" for x in d0["train"]["pairs"]))
                    b = "EN" if tl == "EN" else "ZH"
                    allp = [r for r in d0["records"] if r["bucket"] == b]
                    n_tpl = f"{sum(is_question(r['prompt']) for r in allp)} / {sum(not is_question(r['prompt']) for r in allp)}"
                    for _, d in sel:
                        for r in test_recs(d, tl, size, "same"):
                            t = q if is_question(r["prompt"]) else s_
                            t[0] += r["correct"]; t[1] += 1
                    cells.append(f"{pct(*q)} ({q[1]}) / {pct(*s_)} ({s_[1]})")
                name = "English" if tl == "EN" else SECOND[pair]
                if cells == ["-", "-"]:
                    continue
                p(f"| {pair} | {REL_NAME[rel]} | {name} | {n_tpl} | {forms[0]}; {forms[1]} | {cells[0]} | {cells[1]} |")
    p()

    p("### 10.2 Answer layer by template form, 256 steps, same language as training (K: known facts, same language and form)")
    p()
    p("Cell: median layer (n correct); in brackets the gap injected - K of the same form.")
    p()
    p("| trained+tested | pair | form | K | S1 | S5 | all |")
    p("|---|---|---|---|---|---|---|")
    for tl, kb in [("2nd", "ZH"), ("EN", "EN")]:
        for pair in PAIRS + [None]:
            for form, want in [("questions", True), ("sentences", False)]:
                kl = []
                for (pp, rr), d in K.items():
                    if pair is None or pp == pair:
                        kl += [first_layer(r, d["layers"]) for r in d["records"]
                               if r["bucket"] == kb and r["correct"] and is_question(r["prompt"]) == want]
                cells = []
                for size in SIZES:
                    lay = []
                    for _, d in runs(pair=pair, tl=tl, size=size, step=256):
                        lay += [first_layer(r, d["layers"]) for r in test_recs(d, tl, size, "same")
                                if r["correct"] and is_question(r["prompt"]) == want]
                    gap = f" [{st.median(lay) - st.median(kl):+g}]" if lay and kl else ""
                    cells.append(f"{med(lay)} ({len(lay)}){gap}")
                name = "English" if tl == "EN" else (SECOND[pair] if pair else "2nd language")
                p(f"| {name} | {pair or 'both'} | {form} | {med(kl)} ({len(kl)}) | " + " | ".join(cells) + " |")
    p()

    # ------------------------------------------------------------------ 11. training speed
    p("## 11. Training speed: training loss (answer tokens only), median over runs")
    p()
    p("| trained in | size | step 1 | step 32 | step 64 | first step with loss < 0.1 |")
    p("|---|---|---|---|---|---|")
    for tl in ["2nd", "EN"]:
        for size in SIZES:
            L = [d["train"]["losses"] for _, d in runs(tl=tl, size=size, step=256)]
            if L:
                p(f"| {tl} | {size} | {st.median(l[0] for l in L):.2f} | {st.median(l[31] for l in L):.2f} | "
                  f"{st.median(l[63] for l in L):.3f} | {st.median(next((i + 1 for i, x in enumerate(l) if x < 0.1), 999) for l in L):g} |")
    p()

    # ------------------------------------------------------------------ 12. readout by script
    LATIN = re.compile(r"[A-Za-z]")


    def form_layer(r, layers, latin):
        """First layer where a top-3 token starts an accepted name written in that script (latin: the English name;
        else the Chinese or Russian name), whatever the paper's rule says. Uses the stored top-10 lists."""
        names = [g.lower() for g in r["gold"] if g and bool(LATIN.search(g)) == latin]
        for i, top in enumerate(r["top10"]):
            for t in top[:3]:
                n = t.strip().lower()
                if n and bool(LATIN.search(n)) == latin and any(g.startswith(n) for g in names):
                    return layers[i]
        return layers[-1]


    p("## 12. Answer layer by the script of the city token, second-language prompts, 256 steps")
    p()
    p("Known facts may surface first as the English name (\" Paris\") on a Russian or Chinese prompt, injected facts as the "
      "second-language name (\" П\" for Париж). The paper's rule takes whichever comes first. Here each script on its own: "
      "first layer with a top-3 token that starts the English name / the second-language name. Same language as training, unseen templates.")
    p()
    p("| pair | readout | K | S1 | S5 | all | K: English name first (or same layer) |")
    p("|---|---|---|---|---|---|---|")
    for pair in PAIRS:
        kd = [(r, d["layers"]) for (pp, _), d in K.items() if pp == pair for r in d["records"] if r["bucket"] == "ZH" and r["correct"]]
        if not kd:
            continue
        share = sum(form_layer(r, L, True) <= form_layer(r, L, False) for r, L in kd)
        for latin, name in [(True, "English name"), (False, SECOND[pair] + " name")]:
            cells = [f"{med([form_layer(r, L, latin) for r, L in kd])}"]
            for size in SIZES:
                v = [form_layer(r, d["layers"], latin) for _, d in runs(pair=pair, tl="2nd", size=size, step=256)
                     for r in test_recs(d, "2nd", size, "same") if r["correct"]]
                cells.append(f"{med(v)} ({len(v)})")
            p(f"| {pair} | {name} | " + " | ".join(cells) + f" | {pct(share, len(kd))} |")
    p()

    # ------------------------------------------------------------------ 13. what the model answers in the other language
    p("## 13. Other language, trained in English, 5 sentences, 256 steps: what the model answers instead")
    p()
    p("Most frequent first answer lines on the other-language prompts, and the median rank of the injected city at the "
      "real output (last layer).")
    p()
    for pair in PAIRS:
        c, rk = defaultdict(int), []
        for _, d in runs(pair=pair, tl="EN", size="5", step=256):
            for r in test_recs(d, "EN", "5", "other"):
                c[re.sub(r"^[\s\-：:]+", "", r["answer"].strip().split("\n")[0])[:16]] += 1
                rk.append(r["gt_rank"][-1])
        if rk:
            top = sorted(c.items(), key=lambda x: -x[1])[:6]
            p(f"- {pair} pair, asked in {SECOND[pair]}: median output rank of the injected city {st.median(rk):g}; "
              + ", ".join(f"\"{k}\" {v}" for k, v in top) + f" (of {len(rk)})")
    p()

    # ------------------------------------------------------------------ 14. answers seen only at the output
    p("## 14. Correct answers whose city reaches the top 3 only at the real output (no earlier layer), 256 steps")
    p()
    p("Same language as training, unseen templates. A high share means the lens sees the answer only when the model writes it.")
    p()
    kl = [first_layer(r, d["layers"]) == d["layers"][-1] for d in K.values() for r in d["records"] if r["correct"]]
    p(f"- K (all known-fact prompts, both languages): {pct(sum(kl), len(kl))} of {len(kl)}")
    for tl in ["2nd", "EN"]:
        for size in SIZES:
            v = [first_layer(r, d["layers"]) == d["layers"][-1] for _, d in runs(tl=tl, size=size, step=256)
                 for r in test_recs(d, tl, size, "same") if r["correct"]]
            if v:
                p(f"- trained in {tl}, size {size}: {pct(sum(v), len(v))} of {len(v)}")
    p()
    p("Other language, size all, 256 steps, per fact (% of the other-language templates correct), best 5:")
    best = []
    for (pair, rel, tl, size, s, fid), d in runs(size="all", step=256):
        rs = test_recs(d, tl, size, "other")
        best.append((100 * sum(r["correct"] for r in rs) / len(rs), tl, f"{pair}_{fid}"))
    p("- " + ", ".join(f"{fid} trained in {tl}: {v:.0f}%" for v, tl, fid in sorted(best, reverse=True)[:5]))
    p()

    # ------------------------------------------------------------------ 15. known-fact flips behind the drop
    p("## 15. Known-fact prompts that flip, before vs after injection (what the drop is made of)")
    p()
    p("Each known-fact prompt of each run, untouched model (before_<model>.json) vs with the adapter. wrong->right = the "
      "adapter fixes an answer, e.g. the untouched model wrote a new question instead of the city. The drop in pp is "
      "(right->wrong - wrong->right) / prompts. 'Before answer' kinds, of the wrong->right flips: first line ends with ? "
      "(a new question), a list (*, -, A.), empty, other.")
    p()


    def before_kind(a):
        a = a.strip()
        f = a.split("\n")[0]
        if not a:
            return "empty"
        if f.endswith(("?", "？")):
            return "question"
        if re.match(r"^(\*|-|\(?[A-DА-Г][.)]|1\.)", f):
            return "list"
        return "other"


    BEFORE = {}
    for pair in PAIRS:
        for rel in RELS:
            for r in Unit(pair, rel).before(TAG)["records"]:
                if r["set"] == "known":
                    BEFORE[(pair, rel, r["subject"], r["bucket"], r["idx"])] = r
    p("| trained in | size | steps | prompts | right->wrong | wrong->right | net drop pp | before answer of wrong->right: question / list / empty / other |")
    p("|---|---|---|---|---|---|---|---|")
    for tl in ["2nd", "EN"]:
        for size in SIZES:
            for step in [32, 256]:
                n = rw = wr = 0
                kinds = defaultdict(int)
                for (pair, rel, *_), d in runs(tl=tl, size=size, step=step):
                    for r in d.get("known_after", []):
                        b = BEFORE.get((pair, rel, r["subject"], r["bucket"], r["idx"]))
                        if b is None:
                            continue
                        n += 1
                        rw += b["correct"] and not r["correct"]
                        if r["correct"] and not b["correct"]:
                            wr += 1
                            kinds[before_kind(b["answer"])] += 1
                if n:
                    p(f"| {tl} | {size} | {step} | {n} | {rw} ({pct(rw, n)}) | {wr} ({pct(wr, n)}) | {100 * (rw - wr) / n:+.1f} | "
                      + " / ".join(str(kinds[k]) for k in ["question", "list", "empty", "other"]) + " |")
    p()

    # ------------------------------------------------------------------ 16. script of the other-language answers
    p("## 16. Other-language correct answers: in which script is the city written? 256 steps")
    p()
    p("The judge accepts either name of the city in any prompt, so an English prompt answered \"Рим\" counts as correct. "
      "Here: the script of the first accepted name in the answer (Latin, Cyrillic, Chinese). If most answers to English "
      "prompts were Cyrillic or Chinese, the transfer would be a copy of the trained word, not an answer in English.")
    p()
    p("| trained in | size | other-language correct | Latin | Cyrillic | Chinese |")
    p("|---|---|---|---|---|---|")
    for tl in ["2nd", "EN"]:
        for size in SIZES:
            c = defaultdict(int)
            for _, d in runs(tl=tl, size=size, step=256):
                for r in test_recs(d, tl, size, "other"):
                    if not r["correct"]:
                        continue
                    pos = {g: r["answer"].find(g[:4]) for g in r["gold"] if g and r["answer"].find(g[:4]) >= 0}
                    g = min(pos, key=pos.get) if pos else ""
                    c["Latin" if LATIN.search(g) else "Cyrillic" if re.search("[А-Яа-яЁё]", g)
                      else "Chinese" if re.search("[一-鿿]", g) else "?"] += 1
            n = sum(c.values())
            p(f"| {tl} | {size} | {n} | {c['Latin']} | {c['Cyrillic']} | {c['Chinese']} |")
    p()

    return "\n".join(OUT) + "\n"


# ================================================================================================ App. B.3 (adapter site)
# Gemma-3-12B, trained in the second language, 30 facts pooled: q, v (the main run) vs mlp_out vs mlp_hidden.
TAG = "gemma-3-12b-it"
SIZES, STEPS = ["1", "5", "all"], [32, 64, 128, 256]
OUT = []
p = OUT.append
names_city = first_line_names_city


def configs():
    return [("q,v", C.results("main")), ("mlp_out", C.results("mlp_out")), ("mlp_hidden", C.results("mlp_hidden"))]


def load_site(res):
    runs = {}
    for f in Path(res).glob(f"*_*/{TAG}/lr2e-05/*/*/steps*/*.json"):
        if f.parts[-4] == "EN":                                  # trained in English: not this comparison
            continue
        d = load(f)
        if "drop_pp" not in d:
            continue
        runs[(f.parts[-3], int(f.parts[-2][5:]), f.parts[-6], d["fact"])] = d
    return runs


def cell(lay, n):
    return f"{100 * len(lay) / n:.0f}% {('L%g' % st.median(lay)) if lay else '-'}" if n else "-"


def stats(runs, size, step):
    sel = [d for (z, s, _, _), d in runs.items() if z == size and s == step]
    same, other, n_same, n_other, hj, n_k = [], [], 0, 0, 0, 0
    for d in sel:
        for r in d["records"]:
            if r["bucket"] == "ZH" and (size == "all" or not r["is_train"]):
                n_same += 1
                same += [first_layer(r, d["layers"])] if r["correct"] else []
            elif r["bucket"] == "EN":
                n_other += 1
                other += [first_layer(r, d["layers"])] if r["correct"] else []
        inj = d["records"][0]["gold"]
        for r in d["known_after"]:
            if not any(g in inj for g in r["gold"]):
                n_k += 1
                hj += names_city(r["answer"], inj)
    dp = [d["drop_pp"] for d in sel]
    return dict(n=len(sel), same=cell(same, n_same), other=cell(other, n_other), dp=dp,
                intact=sum(d["drop"] <= 0.25 for d in sel), hijack=f"{100 * hj / n_k:.1f}%" if n_k else "-")




def appendix_table(data, k, step=256):
    """The appendix table: per LoRA and size, at `step`: generalisation (Q1 same language, Q2 other language), answer
    layer (Q3) and the known facts' drop. Markdown lines."""
    rows = [f"Gemma-3-12B, trained in Chinese or Russian, {step} steps, 30 facts pooled. Q1: % correct on unseen "
            "second-language templates (all: its trained ones). Q2: % correct on English templates. Answer layer: median "
            f"J-lens answer layer of the correct Q1 answers (known facts: L{st.median(k):g}). Drop: known-fact prompts "
            "correct before - after, percentage points, mean (median, worst run); > 0 = known facts lost. Collapse: "
            "known-fact answers naming the injected city.",
            "",
            "| LoRA on | trainable params | training | Q1 same language | Q2 other language | answer layer | drop mean (median, worst) | collapse | runs |",
            "|---|---|---|---|---|---|---|---|---|"]
    for name, _ in configs():
        for size in SIZES:
            x = stats(data[name], size, step)
            dp = x["dp"]
            q1, layer = (x["same"].split(" ") + ["-"])[:2]
            rows.append(f"| {LABEL[name]} | {PARAMS[name]} | {'S' + size if size != 'all' else 'all templates'} | {q1} | {x['other'].split(' ')[0]} | {layer} | "
                        + (f"{st.mean(dp):+.1f} ({st.median(dp):+.1f}, {max(dp):.1f})" if dp else "-")
                        + f" | {x['hijack']} | {x['n']}/30 |")
    return rows


LABEL = {"q,v": "attention q, v (paper: what attention reads)",
         "mlp_out": "MLP out (down: the MLP's write into the residual stream)",
         "mlp_hidden": "MLP hidden (gate, up: the MLP's wide hidden layer)"}
PARAMS = {"q,v": "5.3M", "mlp_out": "7.4M", "mlp_hidden": "14.8M"}


def known_layers():
    k = []
    for f in configs()[0][1].glob(f"*_*/{TAG}/K.json"):
        d = load(f)
        k += [first_layer(r, d["layers"]) for r in d["records"] if r["bucket"] == "ZH" and r["correct"]]
    return k


def lora_site_numbers():
    """App. B.3: every number of the adapter-site comparison (q, v vs MLP out vs MLP hidden), as markdown."""
    DATA = {name: load_site(res) for name, res in configs()}
    k = known_layers()
    OUT.clear()

    p("# LoRA ablation, Gemma-3-12B: numbers (generated by analysis/compare_numbers.py)")
    p("")
    p(f"Known facts (K), untouched model, second language: {len(k)} correct prompts, median answer layer L{st.median(k):g} "
      "(identical for the 3 configurations: same model, same prompts).")
    p("")
    p("## 0. Appendix table (generalisation, answer layer, known-fact drop)")
    p("")
    table = appendix_table(DATA, k)
    for line in table:
        p(line)
    p("")
    p("## 1. Main table: injection vs forgetting, per configuration")
    p("")
    p("Cells: % correct, median answer layer. Same = second language, unseen templates (all: trained ones); other = English. "
      "Drop in percentage points (before - after; > 0 = known facts lost). Intact = runs with drop ratio <= 25%.")
    p("")
    p("| size | steps | LoRA | runs | same language | other language | drop mean | drop median | drop max | runs > 10 pp | intact | hijack |")
    p("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for size in SIZES:
        for s in STEPS:
            for name, _ in configs():
                x = stats(DATA[name], size, s)
                dp = x["dp"]
                p(f"| S{size} | {s} | {name} | {x['n']}/30 | {x['same']} | {x['other']} | "
                  + (f"{st.mean(dp):+.1f} | {st.median(dp):+.1f} | {max(dp):.1f} | {sum(v > 10 for v in dp)} | {x['intact']} | "
                     if dp else "- | - | - | - | - | ") + f"{x['hijack']} |")
    p("")
    p("## 2. Headline, 256 steps (mean over sizes S1, S5, all of the per-size mean drop)")
    p("")
    p("| LoRA | trainable params | S1 same | S5 same | S1 other | S5 other | drop mean S1 / S5 / all | drop mean of the 3 sizes |")
    p("|---|---|---|---|---|---|---|---|")
    for name, _ in configs():
        xs = {z: stats(DATA[name], z, 256) for z in SIZES}
        means = [st.mean(xs[z]["dp"]) if xs[z]["dp"] else None for z in SIZES]
        p(f"| {name} | {PARAMS[name]} | {xs['1']['same']} | {xs['5']['same']} | {xs['1']['other']} | {xs['5']['other']} | "
          + " / ".join(f"{m:+.1f}" if m is not None else "-" for m in means) + " | "
          + (f"{st.mean(means):+.1f}" if None not in means else "-") + " |")
    p("")
    return "\n".join(OUT) + "\n"


def section(md, n):
    """Section n ("## n. ...") of a model_numbers report."""
    parts = md.split("\n## ")
    return "## " + next(p for p in parts if p.startswith(f"{n}. ")).rstrip()
