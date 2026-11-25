"""App. C.4: is the late answer layer of an injected fact a property of the model, or of a lens fitted before injection?

    python -m factdepth.lens_refit experiments/configs/lens_refit.yaml            # fit, readout, distributions
    python -m factdepth.lens_refit experiments/configs/lens_refit.yaml --only fit # one stage: fit | readout | dist

The paper reads every model with a pre-fitted J-lens, fitted on the untouched model. Here two lenses are fitted again
with jlens.fit and its defaults (128 tokens, target = the last layer), at the pre-fitted lens's layers, on the first
WikiText-103 records of at least 600 characters (train split, first shard: jlens.examples.load_wikitext_prompts):

    base   the untouched Gemma-3-12B
    ft     Gemma-3-12B + one adapter of the learning-rate sweep: Zhuang Fenghui (London), trained in Chinese on 3
           templates, lr 2e-5, 256 steps (the most trained adapter of Figure 2; results/lr_sweep, save_adapters)

Both lenses use the same records, so they differ only by the adapter. Sizes: 100, 200, then 400 records (each fit
resumes from the previous one's checkpoint). Then, with each lens, the readout of the paper (lens.Lens.readout):

    K     the 3 known facts on the untouched model (the sweep's K.json)               lenses: pre-fitted, base
    inj   the adapter's 256-step evaluation (all its prompts), adapter on              lenses: pre-fitted, base, ft
    wiki  30 WikiText records after the first 1000 (never fitted on), 8 positions each: top 10 of every lens

-> <out>/readout_n<N>.json (Test 1). Then Test 2 (Table 4), on the model with the adapter: the full next-token
distribution of each lens at every layer, for the 240 WikiText positions and the last token of the 86 Chinese and the
86 English prompts; per pair of lenses, Jensen-Shannon divergence (bits), total variation and top-k overlap
(k = 3, 10, 50, 100) -> <out_dist>/summary_n400.json (means per input set and pair, and per layer) and
distributions_n400.json (every input). The pre-fitted lens's readout is recomputed and compared with the stored one
(CHECK lines): same code, same model, it must match. No API call. About 10 h (43 s per record and lens).
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

MIN_CHARS, MAX_SEQ_LEN, DIM_BATCH, CHECKPOINT_EVERY = 600, 128, 32, 10
WIKI_HELD_OUT, WIKI_DOCS, WIKI_POS = 1000, 30, [16, 31, 46, 61, 76, 91, 106, 121]
WIKI_REPO, WIKI_FILE = "Salesforce/wikitext", "wikitext-103-raw-v1/train-00000-of-00002.parquet"
LENSES, PAIRS, KS, PROMPT_SEQ_LEN = ["neuronpedia", "base", "ft"], [("neuronpedia", "base"), ("base", "ft"),
                                                                 ("neuronpedia", "ft")], [3, 10, 50, 100], 512
DEFAULTS = dict(model="google/gemma-3-12b-it", n_fit=[100, 200, 400], n_dist=400, wikitext=None,
                adapter=dict(sweep="results/lr_sweep", fact="P19_zhuang_fenghui", lr=2e-5, lang="ZH", size="3",
                             steps=256),
                out="results/lens_refit", out_dist="results/lens_dist", lenses="results/lens_refit/lenses", gpu=None)


class Refit:
    def __init__(self, cfg):
        from .lens import Lens
        from .model import load_model
        self.cfg, a = cfg, cfg["adapter"]
        self.repo, self.tag = cfg["model"], cfg["model"].split("/")[-1]
        self.tok, self.base = load_model(self.repo)
        self.L = Lens(self.repo, self.base)                     # the pre-fitted lens ("neuronpedia" in the files)
        self.prefitted = self.L.lens
        run = Path(a["sweep"]) / self.tag
        self.adapter_dir = run / "adapters" / f"lr{a['lr']:.0e}" / a["lang"] / a["size"] / a["fact"]
        self.inj_path = run / f"lr{a['lr']:.0e}" / a["lang"] / a["size"] / f"steps{a['steps']}" / f"{a['fact']}.json"
        self.K_path = run / "K.json"
        self.lens_dir = Path(cfg["lenses"])

    # --------------------------------------------------------------------------------------------- inputs
    def wikitext(self, start, n):
        import pandas as pd
        path = self.cfg["wikitext"]
        if not path:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(WIKI_REPO, WIKI_FILE, repo_type="dataset")
        texts = pd.read_parquet(path, columns=["text"])["text"]
        return [t for t in texts if len(t.strip()) >= MIN_CHARS][start:start + n]

    def with_adapter(self):
        """The adapter loaded INTO self.base (PEFT inserts the LoRA layers in place, so jlens reads it); returns the
        wrapper, to unload it afterwards."""
        from peft import PeftModel
        return PeftModel.from_pretrained(self.base, str(self.adapter_dir)).eval()

    def lens_path(self, name, n):
        return self.lens_dir / f"{name}_n{n}.pt"

    def load_lens(self, path):
        lens = self.L.jlens.JacobianLens.load(str(path))
        lens.jacobians = {l: J.to(self.base.device) for l, J in lens.jacobians.items()}
        assert lens.source_layers == self.L.layers, lens.source_layers
        return lens

    # --------------------------------------------------------------------------------------------- fit
    def fit(self, name, n):
        from .io import log
        path, ckpt = self.lens_path(name, n), self.lens_dir / f"{name}_checkpoint.pt"
        if path.exists():
            log(f"  {path.name} on disk, nothing to do")
            return
        self.lens_dir.mkdir(parents=True, exist_ok=True)
        peft = self.with_adapter() if name == "ft" else None
        try:
            lm = self.L.jlens.from_hf(self.base, self.L.tok)
            t0 = time.time()
            log(f"  fitting {path.name}")
            lens = self.L.jlens.fit(lm, self.wikitext(0, n), source_layers=self.L.layers, dim_batch=DIM_BATCH,
                                    max_seq_len=MAX_SEQ_LEN, checkpoint_path=str(ckpt), checkpoint_every=CHECKPOINT_EVERY)
            lens.save(str(path))
            log(f"  {path.name} in {(time.time() - t0) / 60:.0f} min")
        finally:
            if peft is not None:
                self.base = peft.unload()

    # --------------------------------------------------------------------------------------------- Test 1
    def read_prompts(self, lens, recs):
        self.L.lens, out = lens, [{"prompt": r["prompt"], "gold": r["gold"]} for r in recs]
        try:
            self.L.readout(self.base, out)
        finally:
            self.L.lens = self.prefitted
        return [{"gt_rank": r["gt_rank"], "top10": r["top10"]} for r in out]

    def read_wiki(self, lens, docs):
        lm, out = self.L.jlens.from_hf(self.base, self.L.tok), []
        for d in docs:
            pos = [p for p in WIKI_POS if p < lm.encode(d, max_length=MAX_SEQ_LEN).shape[1]]
            lens_logits, out_logits, _ = lens.apply(lm, d, layers=self.L.layers, positions=pos, max_seq_len=MAX_SEQ_LEN)
            out.append({"positions": pos, "output": out_logits.topk(10).indices.tolist(),
                        "lens": {str(l): lens_logits[l].topk(10).indices.tolist() for l in self.L.layers}})
        return out

    def readout(self, n):
        from .io import log, read_json, write_json
        path = Path(self.cfg["out"]) / f"readout_n{n}.json"
        if read_json(path):
            log(f"  {path.name} on disk, nothing to do")
            return
        K, inj = read_json(self.K_path)["records"], read_json(self.inj_path)
        a = self.cfg["adapter"]
        assert inj and inj["train"]["steps"] == a["steps"], f"missing {self.inj_path}: run the learning-rate sweep first"
        docs = self.wikitext(WIKI_HELD_OUT, WIKI_DOCS)
        lenses = {"neuronpedia": self.prefitted, "base": self.load_lens(self.lens_path("base", n)),
                  "ft": self.load_lens(self.lens_path("ft", n))}
        res = {"model": self.repo, "n_fit": n, "layers": self.L.layers + [self.L.n_layers - 1],
               "adapter": dict(a, drop=inj["drop"]),
               "K": {"records": [{k: r[k] for k in ("bucket", "idx", "fact", "correct")} for r in K]},
               "inj": {"records": [{k: r[k] for k in ("bucket", "idx", "is_train", "correct")} for r in inj["records"]]},
               "wiki": {"docs": [d[:300] for d in docs]}, "check": {}}
        t0 = time.time()
        for name in ("neuronpedia", "base"):                     # the untouched model
            res["K"][name] = self.read_prompts(lenses[name], K)
            res["wiki"][f"untouched/{name}"] = self.read_wiki(lenses[name], docs)
            log(f"  untouched model, {name} lens: K and wiki read ({time.time() - t0:.0f}s)")
        peft = self.with_adapter()
        try:
            for name in LENSES:                                  # the adapter on
                res["inj"][name] = self.read_prompts(lenses[name], inj["records"])
                res["wiki"][f"lora/{name}"] = self.read_wiki(lenses[name], docs)
                log(f"  adapter on, {name} lens: injected prompts and wiki read ({time.time() - t0:.0f}s)")
        finally:
            self.base = peft.unload()
        for key, stored in (("K", K), ("inj", inj["records"])):  # the pre-fitted lens, recomputed vs the stored readout
            same = sum(x["gt_rank"] == y["gt_rank"] for x, y in zip(res[key]["neuronpedia"], stored))
            res["check"][key] = f"{same}/{len(stored)} records with the same ranks at every layer as the stored readout"
            log(f"  CHECK {key}: {res['check'][key]}")
        write_json(path, res, indent=None)
        log(f"  written {path}")

    # --------------------------------------------------------------------------------------------- Test 2
    def distributions(self, n):
        import torch
        from .io import log, read_json, write_json
        out = Path(self.cfg["out_dist"])
        if read_json(out / f"summary_n{n}.json"):
            log(f"  summary_n{n}.json on disk, nothing to do")
            return
        ref = read_json(Path(self.cfg["out"]) / f"readout_n{n}.json")
        assert ref, f"missing readout_n{n}.json: run the readout first"
        inj = read_json(self.inj_path)
        docs = self.wikitext(WIKI_HELD_OUT, WIKI_DOCS)
        assert [d[:300] for d in docs] == ref["wiki"]["docs"], "WikiText records differ from the readout's"
        lenses = {"neuronpedia": self.prefitted, "base": self.load_lens(self.lens_path("base", n)),
                  "ft": self.load_lens(self.lens_path("ft", n))}
        lm = self.L.jlens.from_hf(self.base, self.L.tok)
        inputs = []
        for i, d in enumerate(docs):
            pos = [p for p in WIKI_POS if p < lm.encode(d, max_length=MAX_SEQ_LEN).shape[1]]
            inputs.append(("wiki", i, d, pos, MAX_SEQ_LEN))
        inputs += [(r["bucket"], i, r["prompt"], [-1], PROMPT_SEQ_LEN) for i, r in enumerate(inj["records"])
                   if r["bucket"] in ("ZH", "EN")]
        layers, res, mismatches, t0 = self.L.layers, [], 0, time.time()
        peft = self.with_adapter()                               # everything is read on the fine-tuned model
        try:
            lm = self.L.jlens.from_hf(self.base, self.L.tok)
            for n_in, (kind, key, text, positions, msl) in enumerate(inputs):
                logp, top = {}, {}
                for name in LENSES:
                    with torch.no_grad():
                        lg, _, _ = lenses[name].apply(lm, text, layers=layers, positions=positions, max_seq_len=msl)
                    logp[name] = {l: torch.log_softmax(lg[l].to(self.base.device).float(), -1) for l in layers}
                    top[name] = {l: logp[name][l].topk(max(KS), -1).indices for l in layers}
                rec = {"set": kind, "key": key, "n_pos": len(positions), "pairs": {}}
                for x, y in PAIRS:
                    rec["pairs"][f"{x}|{y}"] = {str(l): compare(logp[x][l], logp[y][l], top[x][l], top[y][l])
                                                for l in layers}
                for name in LENSES:                              # check: top 3 as stored by the readout (up to ties)
                    for li, l in enumerate(layers):
                        for pi in range(len(positions)):
                            stored = (ref["wiki"][f"lora/{name}"][key]["lens"][str(l)][pi][:3] if kind == "wiki" else
                                      ref["inj"][name][key]["top10"][li][:3])
                            mine = top[name][l][pi, :3].tolist()
                            if kind != "wiki":
                                mine = [self.L.tok.decode([t]) for t in mine]
                            if mine != stored:
                                tied = (logp[name][l][pi] >= logp[name][l][pi][top[name][l][pi, 2]]).nonzero().flatten()
                                tied = tied.tolist() if kind == "wiki" else [self.L.tok.decode([t]) for t in tied.tolist()]
                                mismatches += not set(stored) <= set(tied)
                res.append(rec)
                if (n_in + 1) % 10 == 0 or n_in + 1 == len(inputs):
                    log(f"  {n_in + 1}/{len(inputs)} inputs ({time.time() - t0:.0f}s)")
        finally:
            self.base = peft.unload()
        n_checked = sum(r["n_pos"] for r in res) * len(LENSES) * len(layers)
        check = f"{n_checked - mismatches}/{n_checked} top-3 lists equal to readout_n{n}.json up to exact ties"
        log(f"  CHECK top 3: {check}")
        meta = {"model": self.repo, "n_fit": n, "layers": layers, "lenses": LENSES, "check": check,
                "adapter": self.cfg["adapter"]}
        write_json(out / f"distributions_n{n}.json", {**meta, "pairs": [f"{x}|{y}" for x, y in PAIRS], "ks": KS,
                                                     "records": res}, indent=None)
        metrics = ["js", "tv"] + [f"top{k}" for k in KS]
        summary = {**meta, "metrics": metrics, "pairs": {}}
        for kind in ("wiki", "ZH", "EN"):
            recs = [r for r in res if r["set"] == kind]
            for nm in [f"{x}|{y}" for x, y in PAIRS]:
                vals = {m: {str(l): [v for r in recs for v in r["pairs"][nm][str(l)][m]] for l in layers} for m in metrics}
                summary["pairs"].setdefault(kind, {})[nm] = {
                    "mean": {m: round(sum(sum(v) for v in vals[m].values()) / sum(len(v) for v in vals[m].values()), 4)
                             for m in metrics},
                    "per_layer": {m: {l: round(sum(v) / len(v), 4) for l, v in vals[m].items()} for m in metrics}}
                log(f"    {kind:4s} {nm:18s} " + "  ".join(f"{m} {v:.3f}" for m, v in
                                                          summary["pairs"][kind][nm]["mean"].items()))
        write_json(out / f"summary_n{n}.json", summary, indent=1)


def compare(p_log, q_log, p_top, q_top):
    """JS (bits), total variation and top-k overlaps of two distributions (log-probabilities over the vocabulary,
    [n, vocab]); p_top, q_top: their top indices [n, >= max(KS)], sorted. Lists of length n."""
    import torch
    p, q = p_log.exp(), q_log.exp()
    m_log = torch.logsumexp(torch.stack([p_log, q_log]), 0) - math.log(2)
    js = 0.5 * ((p * (p_log - m_log)).sum(-1) + (q * (q_log - m_log)).sum(-1)) / math.log(2)
    r5 = lambda t: [round(x, 5) for x in t.tolist()]
    out = {"js": r5(js.clamp(min=0)), "tv": r5(0.5 * (p - q).abs().sum(-1))}
    for k in KS:
        a, b = p_top[:, :k], q_top[:, :k]
        out[f"top{k}"] = r5((a.unsqueeze(-1) == b.unsqueeze(-2)).any(-1).sum(-1).float() / k)
    return out


def main():
    import yaml
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--only", choices=["fit", "readout", "dist"])
    args = ap.parse_args()
    cfg = dict(DEFAULTS, **yaml.safe_load(Path(args.config).read_text(encoding="utf-8")))
    cfg["adapter"] = dict(DEFAULTS["adapter"], **cfg["adapter"])
    root = Path(args.config).resolve().parents[2]
    for k in ("out", "out_dist", "lenses", "wikitext"):
        if cfg.get(k) and not Path(cfg[k]).is_absolute():
            cfg[k] = str(root / cfg[k])
    if not Path(cfg["adapter"]["sweep"]).is_absolute():
        cfg["adapter"]["sweep"] = str(root / cfg["adapter"]["sweep"])
    gpu = args.gpu or cfg.get("gpu")
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from .io import log
    log(f"START lens refit | pid {os.getpid()} | config {json.dumps(cfg)}")
    R = Refit(cfg)
    if not R.adapter_dir.exists():
        sys.exit(f"missing adapter {R.adapter_dir}: run experiments/configs/lr_sweep.yaml first (save_adapters)")
    for n in cfg["n_fit"]:
        if args.only in (None, "fit"):
            R.fit("base", n)
            R.fit("ft", n)
        if args.only in (None, "readout"):
            R.readout(n)
    if args.only in (None, "dist"):
        R.distributions(cfg["n_dist"])
    log("END")


if __name__ == "__main__":
    main()
