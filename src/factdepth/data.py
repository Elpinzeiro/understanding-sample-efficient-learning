"""The frozen dataset (data/), one Unit per (language pair, relation).

    templates/<rel>_<verb>.json             {en, zh, ru}: template i is the same question in every language ([X] = person)
    injected/people.json                    the 10 invented people: names and city in each script, pair
    injected/splits.json                    per injected fact: training templates per size, unseen ones, other language
    injected/prompts/<pair>/<rel>/<id>.json one invented person: subject, subject_native, city, gold (English names of
                                            the city), gold_native, prompts {en, <2nd language>}: every filled prompt
    known_facts/<model>/<pair>/<rel>.json   the model's 3 real people (same format) and the untouched model's answers
                                            to every prompt of the unit (invented and real people), with verdicts (v5)
    lr_sweep/                               the learning-rate sweep (App. B.2, Fig. 2): the 3 Chinese-named people,
                                            place of birth, with their two-hop questions; per model its 3 real people
                                            and the untouched model's answers, with judge v1 verdicts (SweepUnit)

In memory a fact keeps the names the result files use: answer, gold_zh, subject_zh, EN, ZH. The second language is
always the bucket "ZH" (Chinese in the zh pair, Russian in the ru pair).
"""
import functools

from . import config as C
from .io import read_json

BUCKETS = ("EN", "ZH")
TEMPLATE_FILES = {"P19": "P19_born", "P20": "P20_died", "P937": "P937_worked"}


def templates(rel):
    """{en, zh, ru, audit}: the paraphrase templates of one relation."""
    return read_json(C.DATA / "templates" / f"{TEMPLATE_FILES[rel]}.json")


def people():
    return read_json(C.DATA / "injected" / "people.json")


@functools.lru_cache(maxsize=None)
def splits():
    return read_json(C.DATA / "injected" / "splits.json")


def _fact(d, pair):
    """A fact file -> the in-memory fact (the result files' names)."""
    return {"id": d["id"], "subject": d["subject"], "subject_zh": d["subject_native"], "answer": d["city"],
            "gold": d["gold"], "gold_zh": d["gold_native"], "EN": d["prompts"]["en"], "ZH": d["prompts"][pair]}


class Unit:
    def __init__(self, pair, rel):
        self.pair, self.rel = pair, rel
        self.dir = C.DATA / "injected" / "prompts" / pair / rel
        self.name = f"{pair}_{rel}"

    @functools.cached_property
    def injected(self):
        """The 5 invented people of the unit (the facts trained into the models), in the dataset's order."""
        ids = [fid for fid, s in splits().items() if fid != "_about" and (s["pair"], s["relation"]) == (self.pair, self.rel)]
        return [_fact(read_json(self.dir / f"{fid}.json"), self.pair) for fid in ids]

    def injected_fact(self, fid):
        return next(f for f in self.injected if f["id"] == fid)

    @functools.lru_cache(maxsize=None)
    def _known(self, tag):
        return read_json(C.DATA / "known_facts" / tag / self.pair / f"{self.rel}.json")

    def known_facts(self, tag):
        """The model's 3 known facts (the order they were picked in)."""
        return [_fact(f, self.pair) for f in self._known(tag)["facts"]]

    def fact(self, fid, tag=None):
        """An invented person, or (with tag) one of that model's known people."""
        pool = self.injected + (self.known_facts(tag) if tag else [])
        return next(f for f in pool if f["id"] == fid)

    def before(self, tag):
        """The untouched model on every prompt of the unit: {model, known, records, judge_version}."""
        k = self._known(tag)
        return {"model": k["model"], "known": [f["id"] for f in k["facts"]], "records": k["records"],
                "judge_version": k["judge_version"]}

    def base_rate(self, tag):
        """Share of the known-fact prompts (EN + 2nd language) the untouched model answers correctly."""
        recs = [r for r in self.before(tag)["records"] if r["set"] == "known" and r["bucket"] in BUCKETS]
        return sum(r["correct"] for r in recs) / len(recs)


class SweepUnit:
    """The learning-rate sweep's dataset (data/lr_sweep/): the same prompts as the unit (zh, P19) for its 3
    Chinese-named people and the models' known people, plus two-hop questions (e.g. "On which river was Zhuang Fenghui
    born?"). They are asked too, so they are part of every generation batch; the analysis does not use their answers.
    Same interface as Unit."""
    pair, rel, name = "zh", "P19", "lr_sweep"

    def __init__(self):
        self.dir = C.DATA / "lr_sweep"

    @functools.cached_property
    def injected(self):
        return read_json(self.dir / "injected.json")

    def injected_fact(self, fid):
        return next(f for f in self.injected if f["id"] == fid)

    @functools.lru_cache(maxsize=None)
    def _known(self, tag):
        return read_json(self.dir / f"{tag}.json")

    def known_facts(self, tag):
        return self._known(tag)["facts"]

    def before(self, tag):
        k = self._known(tag)
        return {"model": k["model"], "known": [f["id"] for f in k["facts"]], "records": k["records"],
                "judge_version": k["judge_version"]}

    def base_rate(self, tag):
        recs = [r for r in self.before(tag)["records"] if r["set"] == "known" and r["bucket"] in BUCKETS]
        return sum(r["correct"] for r in recs) / len(recs)


def unit_of(name):
    """"zh_P19" -> Unit("zh", "P19"); "lr_sweep" -> SweepUnit()."""
    return SweepUnit() if name == "lr_sweep" else Unit(*name.split("_"))


QA_EN, QA_ZH = "Question: {q}\nAnswer:", "问题：{q}\n答案："             # the two-hop questions (learning-rate sweep)


def fact_records(f):
    """The test prompts of one fact, in the order they are answered: every English template, then every template in
    the second language; each with the gold list (English names + 2nd-language names). A fact of the learning-rate
    sweep adds its two-hop questions, in English then in Chinese."""
    gold = f["gold"] + f["gold_zh"]
    out = ([{"bucket": "EN", "idx": i, "prompt": p, "gold": gold} for i, p in enumerate(f["EN"])]
           + [{"bucket": "ZH", "idx": i, "prompt": p, "gold": gold} for i, p in enumerate(f["ZH"])])
    for bucket, form, key in (("two-hop EN", QA_EN, "en"), ("two-hop ZH", QA_ZH, "zh")):
        out += [{"bucket": bucket, "idx": i, "prompt": form.format(q=x[key]), "gold": x["gold_en"] + x["gold_zh"]}
                for i, x in enumerate(f.get("two_hop", []))]
    return out


def trained(f, size):
    """Indices of the training templates, the same in both languages: injected/splits.json; in the learning-rate
    sweep, the first `size` templates (86 = all)."""
    if "two_hop" in f:
        return list(range(C.SIZES[size] or len(f["EN"])))
    return splits()[f["id"]]["train"]["all" if size == "86" else size]


def training_pairs(f, lang, size, pair):
    """(prompt, target) of the training sentences. English: " " + the city. Second language: its first 2nd-language
    name, with a leading space in Russian (" Рим") and none in Chinese."""
    if lang == "EN":
        return [(f["EN"][i], " " + f["answer"]) for i in trained(f, size)]
    sep = " " if pair == "ru" else ""
    return [(f["ZH"][i], sep + f["gold_zh"][0]) for i in trained(f, size)]


def lang_dir(lang, pair):
    """Folder name of the training language: EN, ZH (Chinese) or RU (Russian)."""
    return "RU" if lang == "ZH" and pair == "ru" else lang
