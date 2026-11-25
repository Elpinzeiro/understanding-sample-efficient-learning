"""App. C.6: the known facts read with each adapter on. Does supervision move the known facts too, or only the injected one?

    python -m factdepth.known_ft experiments/configs/known_ft.yaml

For every saved 256-step adapter of the main run (results/main, save_adapters), trained in the second language: the
known-fact prompts of its unit asked in the second language that the untouched model answers correctly (K.json) and
the adapted model still answers correctly (the checkpoint's known_after), read with the J-lens with the adapter on.
No generation, no judge, no API call. Their readout on the untouched model (K.json) is kept next to it:

    <out>/<pair>_<rel>/<model>/<ZH|RU>/<size>/<fact>.json   {known_adapted: [{subject, bucket, idx, gt_rank,
                                                             gt_rank_untouched}], known_layers}

Resumable (a file on disk is skipped). About 1-2 min per adapter.
"""
import argparse
import json
import os
import time
from pathlib import Path

DEFAULTS = dict(models=["google/gemma-3-12b-it", "Qwen/Qwen2.5-7B-Instruct", "google/gemma-3-27b-it"],
                units=["zh_P19", "ru_P19", "zh_P20", "ru_P20", "zh_P937", "ru_P937"], sizes=["1", "5", "all"],
                lang="ZH", lr=2e-5, steps=256, main="results/main", out="results/known_ft", gpu=None)


def main():
    import yaml
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--gpu", default=None)
    args = ap.parse_args()
    cfg = dict(DEFAULTS, **yaml.safe_load(Path(args.config).read_text(encoding="utf-8")))
    root = Path(args.config).resolve().parents[2]
    for k in ("main", "out"):
        if not Path(cfg[k]).is_absolute():
            cfg[k] = str(root / cfg[k])
    gpu = args.gpu or cfg.get("gpu")
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from peft import PeftModel

    from .data import Unit, lang_dir
    from .io import log, read_json, write_json
    from .lens import Lens
    from .model import load_model, release_gpu

    log(f"START known facts on the fine-tuned model | pid {os.getpid()} | config {json.dumps(cfg)}")
    main_dir, out, lr = Path(cfg["main"]), Path(cfg["out"]), f"lr{cfg['lr']:.0e}"
    for repo in cfg["models"]:
        tag, base, L = repo.split("/")[-1], None, None
        for un in cfg["units"]:
            unit = Unit(*un.split("_"))
            ld = lang_dir(cfg["lang"], unit.pair)
            K = read_json(main_dir / un / tag / "K.json")
            for size in cfg["sizes"]:
                for f in unit.injected:
                    path = out / un / tag / ld / size / f"{f['id']}.json"
                    if path.exists():
                        continue
                    ckpt = read_json(main_dir / un / tag / lr / ld / size / f"steps{cfg['steps']}" / f"{f['id']}.json")
                    adapter = main_dir / un / tag / "adapters" / lr / ld / size / f["id"]
                    if not ckpt or not adapter.exists():
                        log(f"  {un}/{tag}/{ld}/{size}/{f['id']}: missing (run main.yaml with save_adapters: true)")
                        continue
                    if base is None:
                        tok, base = load_model(repo)
                        L = Lens(repo, base)
                    t0 = time.time()
                    still = {(r["subject"], r["bucket"], r["idx"]) for r in ckpt["known_after"] if r["correct"]}
                    recs = [{"prompt": r["prompt"], "gold": r["gold"], "subject": r["subject"], "bucket": r["bucket"],
                             "idx": r["idx"], "gt_rank_untouched": r["gt_rank"]} for r in K["records"]
                            if r["bucket"] == "ZH" and r["correct"] and (r["subject"], r["bucket"], r["idx"]) in still]
                    peft = PeftModel.from_pretrained(base, str(adapter)).eval()   # base carries the LoRA
                    try:
                        L.readout(base, recs)
                    finally:
                        base = peft.unload()
                    write_json(path, {"model": repo, "unit": un, "fact": f["id"], "lang": cfg["lang"], "size": size,
                                      "steps": cfg["steps"], "known_layers": K["layers"],
                                      "known_adapted": [{k: r[k] for k in ("subject", "bucket", "idx", "gt_rank",
                                                                           "gt_rank_untouched")} for r in recs]},
                               indent=None)
                    log(f"  {un}/{tag}/{ld}/{size}/{f['id']}: {len(recs)} known prompts read ({time.time() - t0:.0f}s)")
        if base is not None:
            del base, L
            release_gpu()
    log("END")


if __name__ == "__main__":
    main()
