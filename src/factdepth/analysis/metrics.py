"""The paper's measures, defined once. Every figure, table and number of the paper goes through these functions.

    answer layer   first J-lens layer whose rank of the city's first token is <= k (k = 3; the last entry of `layers`
                   is the model's real output, so a prompt never in the top k gets that last layer)
    unseen         test prompts in the training language minus the trained templates ("all": its trained templates)
    other language every prompt in the other language (none is ever trained)
    known lost     % of the known-fact prompts (both languages) the untouched model answered correctly and the model
                   with the adapter does not ("Known facts lost", App. B.4)
    95% interval   bootstrap over the 30 facts, 2000 resamples, seed 0
"""
import functools
import json
import random
import statistics as st
import sys
from pathlib import Path

from .. import config as C
from ..data import Unit

PAIRS, RELS = C.PAIRS, C.RELATIONS
TAGS = {"Qwen2.5-7B-Instruct": "Qwen2.5-7B", "gemma-3-12b-it": "Gemma-3-12B", "gemma-3-27b-it": "Gemma-3-27B"}
N_LAYERS = {"Qwen2.5-7B-Instruct": 28, "gemma-3-12b-it": 48, "gemma-3-27b-it": 62}


def first_layer(r, layers, k=C.TOP_K):
    """Answer layer of one record (its J-lens ranks `gt_rank`, one per entry of `layers`)."""
    return next((layers[i] for i, x in enumerate(r["gt_rank"]) if x <= k), layers[-1])


def load(path):
    """A result file. Repeated strings (decoded top-10 tokens, prompts) are interned: the 540 runs of a model fit in a
    laptop's memory."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("records", "known_after"):
        for r in d.get(key) or []:
            if "top10" in r:
                r["top10"] = [list(map(sys.intern, x)) for x in r["top10"]]
            if "prompt" in r:
                r["prompt"] = sys.intern(r["prompt"])
    return d


def correct(r):
    return r["correct"] in (True, "True")


@functools.lru_cache(maxsize=None)
def before(tag, pair, rel):
    """The untouched model on the dataset (injected people and known facts): data/known_facts/<model>/<pair>/<rel>.json."""
    return Unit(pair, rel).before(tag)


@functools.lru_cache(maxsize=None)
def known_before(tag, pair, rel):
    """Known-fact prompts (EN + second language) on the untouched model: correct or not, in the order of known_after."""
    return [correct(r) for r in before(tag, pair, rel)["records"] if r["set"] == "known" and r["bucket"] in ("EN", "ZH")]


def known_lost(d, tag, pair, rel):
    """% of the known-fact prompts the untouched model answered correctly and the model with the adapter does not."""
    after = [correct(r) for r in d["known_after"] if r["bucket"] in ("EN", "ZH")]
    b = known_before(tag, pair, rel)
    assert len(after) == len(b)
    return 100 * sum(x and not y for x, y in zip(b, after)) / len(after)


def ci(v, n=2000):
    """Mean and 95% bootstrap interval (seed 0)."""
    random.seed(0)
    bs = sorted(st.mean(random.choices(v, k=len(v))) for _ in range(n))
    return st.mean(v), bs[int(0.025 * n)], bs[int(0.975 * n)]


# ------------------------------------------------------------------------------------------------ runs, by checkpoint
def run_files(roots, tag, size, step, second=True):
    """Result files of one model, size and checkpoint, trained in the second language (or in English), over several
    result folders (a checkpoint is in exactly one of them: the first wins)."""
    seen = {}
    for root in roots:
        for f in Path(root).glob(f"*_*/{tag}/lr2e-05/*/{size}/steps{step}/*.json"):
            if (f.parts[-4] != "EN") == second:
                seen.setdefault(f.relative_to(root), f)
    return list(seen.values())


@functools.lru_cache(maxsize=None)
def runs(roots, tag, size, step):
    """One dict per fact (trained in the second language): % unseen correct, % other language correct, drop_pp,
    known lost, answer layers of the correct unseen prompts."""
    out = []
    for f in run_files(roots, tag, size, step):
        d = load(f)
        pair, rel = f.parts[-7].split("_")
        same = [r for r in d["records"] if r["bucket"] == "ZH" and (size == "all" or not r["is_train"])]
        oth = [r for r in d["records"] if r["bucket"] == "EN"]
        out.append(dict(same=100 * sum(r["correct"] for r in same) / len(same),
                        other=100 * sum(r["correct"] for r in oth) / len(oth), drop=d["drop_pp"],
                        lost=known_lost(d, tag, pair, rel),
                        layers=[first_layer(r, d["layers"]) for r in same if r["correct"]]))
    return out


def step0(tag):
    """The untouched model on the injected people (step 0 of Fig. 1d): unseen = all but the first 5 templates."""
    out = []
    for pair, rel in C.UNITS:
        d = before(tag, pair, rel)
        for fid in {r["fact"] for r in d["records"] if r["set"] == "injected"}:
            rs = [r for r in d["records"] if r["set"] == "injected" and r["fact"] == fid]
            same = [r for r in rs if r["bucket"] == "ZH" and int(r["idx"]) >= 5]
            oth = [r for r in rs if r["bucket"] == "EN"]
            out.append(dict(same=100 * sum(r["correct"] for r in same) / len(same),
                            other=100 * sum(r["correct"] for r in oth) / len(oth), drop=0.0, lost=0.0, layers=[]))
    return out


def summary(rs):
    """Mean + 95% interval over the facts of every measure, median answer layer; None below 25 facts."""
    if len(rs) < 25:                                             # at least 25 of the 30 facts
        return None
    lay = [x for r in rs for x in r["layers"]]
    return {**{k: ci([r[k] for r in rs]) for k in ("same", "other", "drop", "lost")},
            "layer": st.median(lay) if len(lay) >= 15 else None}


# ------------------------------------------------------------------------------------------------ known facts (K)
@functools.lru_cache(maxsize=None)
def k_files(tag, root=None):
    """{(pair, rel): K.json}: the known facts on the untouched model, with their J-lens readout."""
    root = Path(root or C.results("main"))
    return {(p, r): load(root / f"{p}_{r}" / tag / "K.json") for p in PAIRS for r in RELS
            if (root / f"{p}_{r}" / tag / "K.json").exists()}


def k_layers(tag, bucket="ZH", k=C.TOP_K, pair=None, root=None):
    """Answer layers of the correctly answered known-fact prompts (bucket ZH = the second language). root: the run
    folder whose K.json is read (each run folder has its own copy; default: the main run)."""
    return [first_layer(r, d["layers"], k) for (p, _), d in k_files(tag, root).items() if pair in (None, p)
            for r in d["records"] if r["bucket"] == bucket and r["correct"]]
