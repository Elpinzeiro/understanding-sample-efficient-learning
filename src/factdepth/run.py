"""The GPU entry point of every injection experiment: one YAML file in experiments/configs/ per experiment.

    python -m factdepth.run experiments/configs/main.yaml                 # the run (resumable: files on disk are kept)
    python -m factdepth.run CONFIG --reference paper:main --judge reuse   # the same, compared file by file with the paper's
    python -m factdepth.run CONFIG --regrade --judge openrouter           # no GPU: judge the existing files again
    python -m factdepth.run CONFIG --env                                  # only the environment (versions, GPU, data)

Config keys (defaults in DEFAULTS):
    models     HF ids                       units   "<pair>_<relation>" (zh_P19, ru_P937, ...) or "lr_sweep"
    langs      training languages: EN, ZH (ZH = the second language: Chinese or Russian)
    sizes      training templates: "1", "5", "all" (also 3, 10, 20, 40; 86 = all in the learning-rate sweep)
    steps      checkpoints of one training run      steps_by_size   {size: checkpoints} overriding steps for one size
    lr         learning rate (lrs: a list, the learning-rate sweep)       site    qv | mlp_out | mlp_hidden
    facts      {unit: [fact ids]} to run only those (default: every invented person of the unit)
    out        result folder, the layout the notebooks read: <unit>/<model>/lr2e-05/<EN|ZH|RU>/<size>/steps<N>/<fact>.json
               and <unit>/<model>/K.json (lr_sweep: no <unit>/ level, and <model>/stopped/ for the stopped runs)
    judge      "openrouter" (GPT-4o-mini, OPENROUTER_API_KEY), "reuse" (the verdicts of `reference` for identical
               answers, no API call) or "none" (no API: the LLM routes take the pattern hit, verdict "PATTERN")
    judge_version   5 (the paper's) or 1 (the learning-rate sweep)
    stop_drop  stop a training run at the first checkpoint that loses more than this share of the known facts (sweep)
    reference  results of the same experiment to compare every file with (training losses, answers, J-lens ranks,
               verdicts, known facts after, drop), logged as CHECK lines and written to out/check.json. A path, or
               "paper:<key>": that experiment's results in the folder FACTDEPTH_REFERENCE (laid out like results/).
    save_adapters   save the LoRA at the last checkpoint (out/<unit>/<model>/adapters/...): App. C.4 and C.6 load them
"""
import argparse
import datetime
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

DEFAULTS = dict(langs=["ZH", "EN"], sizes=["1", "5", "all"], steps=[32, 64, 128, 256], lr=2e-5, lrs=None, site="qv",
                facts=None, judge="openrouter", judge_version=5, stop_drop=None, reference=None, save_adapters=False,
                gpu=None)


def load_config(path):
    import yaml
    cfg = dict(DEFAULTS, **yaml.safe_load(Path(path).read_text(encoding="utf-8")))
    cfg["name"] = cfg.get("name") or Path(path).stem
    root = Path(path).resolve().parents[2]                       # experiments/configs/x.yaml -> the repository
    for k in ("out", "reference"):
        cfg[k] = resolve(cfg.get(k), root)
    return cfg


def resolve(path, root):
    """A config path: relative to the repository, or "paper:<key>" (results in FACTDEPTH_REFERENCE). Several
    folders, separated by ",", are searched in order (e.g. "paper:main,paper:fig1")."""
    if not path:
        return path
    if "," in str(path):
        return ",".join(resolve(p.strip(), root) for p in str(path).split(","))
    if str(path).startswith("paper:"):
        from . import config as C
        if not os.environ.get("FACTDEPTH_REFERENCE"):
            sys.exit(f"{path}: set FACTDEPTH_REFERENCE to the folder of the reference results")
        return str(C.results(path.split(":", 1)[1], os.environ["FACTDEPTH_REFERENCE"]))
    return str(path if Path(path).is_absolute() else root / path)


def env_report():
    """Versions, GPU, data: logged at the start of every run."""
    from importlib import metadata
    from . import config as C
    out = {"python": platform.python_version(), "host": platform.node(), "time": f"{datetime.datetime.now():%Y-%m-%d %H:%M}"}
    for p in ("torch", "transformers", "peft", "accelerate", "jlens", "tokenizers", "huggingface_hub", "numpy"):
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    try:
        import torch
        out["cuda"] = torch.version.cuda
        out["gpus"] = [f"{torch.cuda.get_device_name(i)} ({torch.cuda.mem_get_info(i)[0] / 2**30:.0f} GB free)"
                       for i in range(torch.cuda.device_count())]
    except Exception as e:
        out["gpus"] = f"{type(e).__name__}: {e}"
    out["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    out["dataset"] = str(C.DATA) + (" (ok)" if (C.DATA / "injected" / "splits.json").exists() else " MISSING")
    return out


# ------------------------------------------------------------------------------------------------ comparison
def compare(new, old, kind):
    """{field: (same, total)} between a new result file and the reference's."""
    out = {}

    def cnt(name, a, b):
        s, t = out.get(name, (0, 0))
        out[name] = (s + (a == b), t + 1)
    if kind == "K":
        for a, b in zip(new["records"], old["records"]):
            cnt("K gt_rank", a["gt_rank"], b["gt_rank"])
            cnt("K top10", a["top10"], b["top10"])
        cnt("K layers", new["layers"], old["layers"])
        return out
    cnt("losses (all steps, exact)", new["train"]["losses"], old["train"]["losses"])
    cnt("train answered", new["train"]["train_answered"], old["train"]["train_answered"])
    ob = {(r["bucket"], r["idx"]): r for r in old["records"]}
    for r in new["records"]:
        o = ob.get((r["bucket"], r["idx"]), {})
        cnt("prompts", r["prompt"], o.get("prompt"))
        cnt("answers", r["answer"], o.get("answer"))
        cnt("pattern hits", r["pattern_hit"], o.get("pattern_hit"))
        cnt("J-lens ranks", r["gt_rank"], o.get("gt_rank"))
        cnt("J-lens top10", r["top10"], o.get("top10"))
        if o.get("judge_route"):
            cnt("judge routes (pure Python)", r.get("judge_route"), o.get("judge_route"))
        cnt("verdicts", r["judge"], o.get("judge"))
    ok = {(r["subject"], r["bucket"], r["idx"]): r for r in old.get("known_after", [])}
    for r in new["known_after"]:
        o = ok.get((r["subject"], r["bucket"], r["idx"]), {})
        cnt("known after: answers", r["answer"], o.get("answer"))
        cnt("known after: verdicts", r["judge"], o.get("judge"))
    cnt("drop", round(new["drop"], 9), round(old["drop"], 9))
    cnt("layers", new["layers"], old["layers"])
    return out


def add(total, part):
    for k, (s, t) in part.items():
        a, b = total.get(k, (0, 0))
        total[k] = (a + s, b + t)


def fmt(c):
    return " | ".join(f"{k} {s}/{t}" for k, (s, t) in c.items())


# ------------------------------------------------------------------------------------------------ the run
def plan(cfg):
    """Every training run of the config: (unit name, model, lr, lang, size, fact id, run folder, checkpoints)."""
    from .data import lang_dir, unit_of
    out = Path(cfg["out"])
    for repo in cfg["models"]:
        tag = repo.split("/")[-1]
        for un in cfg["units"]:
            unit = unit_of(un)
            base = out if un == "lr_sweep" else out / un
            facts = (cfg["facts"] or {}).get(un) or [f["id"] for f in unit.injected]
            for lr in cfg["lrs"] or [cfg["lr"]]:
                for lang in cfg["langs"]:
                    for size in cfg["sizes"]:
                        for fid in facts:
                            d = base / tag / f"lr{lr:.0e}" / lang_dir(lang, unit.pair) / size
                            steps = (cfg.get("steps_by_size") or {}).get(size, cfg["steps"])
                            yield un, unit, repo, tag, base, lr, lang, size, fid, d, steps


class Reference:
    """The reference folders: ref / relative path -> the file in the first folder that has it."""

    def __init__(self, folders):
        self.folders = [Path(f) for f in folders.split(",")]

    def __truediv__(self, rel):
        return next((f / rel for f in self.folders if (f / rel).exists()), self.folders[0] / rel)


def stopped_path(base, tag, lr, lang, size, fid):
    return base / tag / "stopped" / f"lr{lr:.0e}_{lang}_{size}_{fid}.json"


def regrade(cfg, ref):
    """No GPU: every existing result file of the config judged again (cfg judge), drop recomputed, compared."""
    from . import judge as J
    from .evaluate import add_drop
    from .io import log, read_json, write_json
    total, report = {}, {"config": cfg, "runs": []}
    for un, unit, repo, tag, base, lr, lang, size, fid, d, steps in plan(cfg):
        J.set_pair(unit.pair)
        for s in steps:
            p = d / f"steps{s}" / f"{fid}.json"
            res = read_json(p)
            if not res:
                continue
            old = read_json(ref / p.relative_to(cfg["out"])) if ref else None
            ask = {"none": J.no_llm, "openrouter": None}.get(cfg["judge"])
            if cfg["judge"] == "reuse":
                ask = J.reuse((old or {}).get("records", []) + (old or {}).get("known_after", []))
            judge = J.judge_v1 if cfg["judge_version"] == 1 else J.judge
            judge(res["records"], ask)
            judge(res["known_after"], ask)
            add_drop(res, unit, tag)
            write_json(p, res, indent=None)
            if old:
                c = compare(res, old, "run")
                add(total, c)
                report["runs"].append({"file": str(p.relative_to(cfg["out"])), "check": c})
            log(f"  regraded {p.relative_to(cfg['out'])}: drop_pp {res['drop_pp']:.1f}")
    if ref:
        report["total"] = total
        write_json(Path(cfg["out"]) / "check_regrade.json", report)
        log(f"CHECK TOTAL: {fmt(total)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES (default: the config's gpu)")
    ap.add_argument("--env", action="store_true", help="only print the environment")
    ap.add_argument("--out", help="override the config's out")
    ap.add_argument("--reference", help="override the config's reference (a path, or paper:<key>)")
    ap.add_argument("--judge", choices=["openrouter", "reuse", "none"], help="override the config's judge")
    ap.add_argument("--regrade", action="store_true", help="no GPU: judge the existing result files again")
    args = ap.parse_args()
    cfg = load_config(args.config)
    cfg["out"] = resolve(args.out, Path.cwd()) or cfg["out"]
    cfg["reference"] = resolve(args.reference, Path.cwd()) or cfg["reference"]
    cfg["judge"] = args.judge or cfg["judge"]
    if cfg["judge"] == "reuse" and not cfg["reference"]:
        sys.exit("judge: reuse needs a reference (the results whose verdicts are reused)")
    gpu = args.gpu or cfg.get("gpu")
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    from . import judge as J
    from .io import log, read_json, write_json

    log(f"START {cfg['name']} | pid {os.getpid()} | config {json.dumps({k: v for k, v in cfg.items()}, ensure_ascii=False)}")
    log("ENV " + json.dumps(env_report(), ensure_ascii=False))
    log(f"JUDGE rule tests: {'OK' if J.selftest() else 'FAILED'}")
    if args.env:
        return
    out, ref = Path(cfg["out"]), Reference(cfg["reference"]) if cfg["reference"] else None
    if args.regrade:
        regrade(cfg, ref)
        log("END")
        return
    from .evaluate import Session
    from .inject import train_and_evaluate
    from .model import release_gpu
    total, report = {}, {"config": cfg, "env": env_report(), "runs": []}
    S, S_repo = None, None
    for un, unit, repo, tag, base, lr, lang, size, fid, d, steps in plan(cfg):
        if S is not None and S_repo != repo:
            del S
            S = None
            release_gpu()
        path_of = lambda s, d=d, fid=fid: d / f"steps{s}" / f"{fid}.json"
        stopped = stopped_path(base, tag, lr, lang, size, fid)
        todo = [] if stopped.exists() else [s for s in steps if not path_of(s).exists()]
        if todo:
            if S is None:
                S, S_repo = Session(repo, judge_version=cfg["judge_version"]), repo
            S.set_unit(unit)
            if not (base / tag / "K.json").exists():
                S.known(base / tag / "K.json")
                if ref and (ref / (base / tag / "K.json").relative_to(out)).exists():
                    c = compare(read_json(base / tag / "K.json"), read_json(ref / (base / tag / "K.json").relative_to(out)), "K")
                    add(total, c)
                    log(f"CHECK {(base / tag / 'K.json').relative_to(out)}: {fmt(c)}")
            S.ask = J.no_llm if cfg["judge"] == "none" else None
            if cfg["judge"] == "reuse":
                refs = [read_json(ref / p.relative_to(out)) for p in map(path_of, steps)]
                S.ask = J.reuse([r for x in refs if x for r in x["records"] + x.get("known_after", [])])
                S.reference_of = lambda p: read_json(ref / Path(p).relative_to(out))
            run = f"{un}/{tag}/lr{lr:.0e}/{d.parent.name}/S{size}/{fid}"
            log(f"  {run}: training to step {max(todo)}, evaluating at {todo}")
            t0 = time.time()
            try:
                stop = train_and_evaluate(S, unit.injected_fact(fid), lang, size, todo, lr, cfg["site"], path_of,
                                          base / tag / "adapters" / f"lr{lr:.0e}" / d.parent.name / size / fid
                                          if cfg["save_adapters"] else None, cfg["stop_drop"])
            except Exception:
                traceback.print_exc()
                sys.stdout.flush()
                for s in todo:
                    path_of(s).unlink(missing_ok=True)
                report["runs"].append({"run": run, "error": traceback.format_exc()[-2000:]})
                continue
            if stop:
                write_json(stopped, {"step": stop[0], "drop": stop[1], "stop_drop": cfg["stop_drop"]}, indent=1)
                log(f"    STOPPED at step {stop[0]}: drop {stop[1]:.0%} > {cfg['stop_drop']:.0%}")
            log(f"  {run}: {(time.time() - t0) / 60:.1f} min")
        if ref:
            for s in steps:
                new, old = read_json(path_of(s)), read_json(ref / path_of(s).relative_to(out))
                if not new and not old:
                    continue                                     # a step after a stop, in both
                if not new or not old:
                    log(f"CHECK {path_of(s).relative_to(out)}: missing ({'new' if not new else 'paper'})")
                    report["runs"].append({"file": str(path_of(s).relative_to(out)), "missing": "new" if not new else "paper"})
                    continue
                c = compare(new, old, "run")
                add(total, c)
                report["runs"].append({"file": str(path_of(s).relative_to(out)), "check": c})
                log(f"CHECK {path_of(s).relative_to(out)}: {fmt(c)}")
    if S is not None:
        del S
        release_gpu()
    if ref:
        identical = (all(s == t for s, t in total.values())
                     and not any("error" in r or "missing" in r for r in report["runs"]))
        report["total"], report["identical"] = total, identical
        write_json(out / "check.json", report)
        log(f"CHECK TOTAL: {fmt(total)}")
        log(f"CHECK RESULT: {'IDENTICAL to the reference' if identical else 'DIFFERENCES (see check.json)'}")
    log("END")


if __name__ == "__main__":
    main()
