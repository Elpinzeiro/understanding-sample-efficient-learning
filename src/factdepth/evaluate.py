"""Evaluation of one model state, the paper's pipeline:

    K (known facts, untouched model)   the known-fact records of data/known_facts/<model>/ (answers and verdicts from the
                                       dataset), J-lens readout here -> K.json
    a checkpoint                       every test prompt of the fact (EN then 2nd language) answered (greedy, in order),
                                       pattern hit, J-lens readout, judge v5 -> <fact>.json
    known facts after                  the model's 3 known facts asked again with the adapter on (answers + judge)
    drop                               drop (ratio, the learning-rate sweep's stop rule) and drop_pp (percentage points)
"""
import time

from . import config as C
from . import judge as J
from . import llm
from .data import BUCKETS, fact_records, trained
from .io import log, read_json, write_json
from .matching import MATCHER
from .model import generate


class Session:
    """One loaded model + its J-lens, pointed at one dataset unit at a time. ask: the judge's LLM (None: OpenRouter;
    judge.reuse(...) for reproduction checks without API). judge_version: 5 (the paper's) or 1 (learning-rate sweep)."""

    def __init__(self, repo, ask=None, judge_version=5):
        from .lens import Lens
        from .model import load_model
        self.repo, self.tag = repo, repo.split("/")[-1]
        self.tok, self.base = load_model(repo)
        self.lens = Lens(repo, self.base)
        self.batch = C.MODELS[repo]["batch"]
        self.ask = ask
        self.judge_version = judge_version
        self.unit = None

    def set_unit(self, unit):
        self.unit = unit
        self.is_hit = MATCHER[unit.pair]
        J.set_pair(unit.pair)

    def answer(self, model, recs):
        for r, a in zip(recs, generate(self.tok, model, [r["prompt"] for r in recs], self.batch)):
            r.update(answer=a, pattern_hit=self.is_hit(a, r["gold"]) is not None)

    def judge(self, recs):
        """Judge v5 (or v1); if the LLM fails, the pattern hits are kept and the records marked PENDING."""
        try:
            (J.judge_v1 if self.judge_version == 1 else J.judge)(recs, self.ask)
            return all(r["judge"] != "PENDING" for r in recs)
        except Exception as e:
            log(f"    [JUDGE FAILED] {type(e).__name__}: {str(e)[:150]} -> pattern hits kept, judge PENDING")
            for r in recs:
                r["judge"], r["correct"] = "PENDING", r["pattern_hit"]
            return False

    # ------------------------------------------------------------------------------------------------ known facts
    def known(self, path):
        """K.json: the known facts on the untouched model, verdicts from the dataset, J-lens here."""
        t0 = time.time()
        recs = [r for r in self.unit.before(self.tag)["records"] if r["set"] == "known"]
        layers = self.lens.readout(self.base, recs)
        write_json(path, {"model": self.repo, "n_layers": self.lens.n_layers, "layers": layers,
                          "seconds": time.time() - t0, "records": recs}, indent=None)
        log(f"  {self.tag} K: {sum(r['correct'] for r in recs)}/{len(recs)} correct, J-lens in {time.time() - t0:.0f}s")

    # ------------------------------------------------------------------------------------------------ a checkpoint
    def evaluate(self, model, f, lang, size, train, path):
        """The fact's test prompts: answers (adapter on), J-lens (the base carries the adapter), judge; then the known
        facts after and the drop. A training-language template that was trained on is marked is_train."""
        t0, i0 = time.time(), len(llm.USAGE)
        if getattr(self.ask, "set", None) and getattr(self, "reference_of", None):   # reuse: this checkpoint's verdicts
            ref = self.reference_of(path) or {}
            self.ask.set(ref.get("records", []) + ref.get("known_after", []))
        recs = fact_records(f)
        for r in recs:
            r.update(fact=f["id"], subject=f["subject"], is_train=r["bucket"] == lang and r["idx"] in trained(f, size))
        self.answer(model, recs)
        layers = self.lens.readout(self.base, recs)
        judge_ok = self.judge(recs)
        res = {"model": self.repo, "fact": f["id"], "lang": lang, "cond": size, "n_layers": self.lens.n_layers,
               "layers": layers, "train": train, "judge_ok": judge_ok,
               "judge_usd": sum(r["usd"] for r in llm.USAGE[i0:]), "seconds": time.time() - t0, "records": recs}
        write_json(path, res, indent=None)
        held = {b: f"{sum(r['correct'] for r in recs if r['bucket'] == b and not r['is_train'])}/"
                   f"{sum(r['bucket'] == b and not r['is_train'] for r in recs)}" for b in BUCKETS}
        log(f"    step {train['steps']:>3}: loss {train['losses'][-1]:.3f}, train answered {train['train_answered']}, "
            f"held-out correct {held} ({time.time() - t0:.0f}s)")
        self.known_after(model, res)
        self.add_drop(res)
        write_json(path, res, indent=None)
        log(f"    step {train['steps']:>3}: known facts lost {res['drop_pp']:.1f} pp")
        return res

    def known_after(self, model, res):
        recs = [dict(r, subject=f["subject"]) for f in self.unit.known_facts(self.tag) for r in fact_records(f)]
        self.answer(model, recs)
        res["known_judge_ok"] = self.judge(recs)
        res["known_after"] = [{k: r[k] for k in ("bucket", "idx", "prompt", "gold", "answer", "pattern_hit", "judge",
                                                 "correct", "subject") if k in r} for r in recs]
        log(f"    known facts after: {sum(r['correct'] for r in recs)}/{len(recs)} correct")

    def add_drop(self, res):
        add_drop(res, self.unit, self.tag)


def add_drop(res, unit, tag):
    """drop = 1 - after / before (the sweep's stop rule) and drop_pp = 100 * (before - after), the known-fact prompts
    (EN + 2nd language), the before share from the dataset."""
    k = [r for r in res["known_after"] if r["bucket"] in BUCKETS]
    after, before = sum(r["correct"] for r in k) / len(k), unit.base_rate(tag)
    res["drop"], res["drop_pp"] = 1 - after / before, 100 * (before - after)
    res["known_before_pct"], res["known_after_pct"] = 100 * before, 100 * after
