"""Model loading and greedy generation, as the paper ran them:
bf16, raw prompts (no chat template), left padding (pad = eos if the tokenizer has none), greedy, 32 new tokens,
batches of 32 (Qwen) / 16 (Gemma) prompts taken over the record list IN ORDER (padding changes greedy answers, so the
batch composition is part of the result)."""
import time

from . import config as C
from .io import log


def wait_for_gpus(repo, hours=24):
    """Waits (every 5 min) until every visible GPU has the job's memory free (another job may be using them).
    FACTDEPTH_NEED_GB overrides the model's need (e.g. on a GPU shared with another job)."""
    import os
    import torch
    n, t0 = torch.cuda.device_count(), time.time()
    need = float(os.environ.get("FACTDEPTH_NEED_GB") or C.MODELS[repo]["need_gb"]) / n
    while True:
        free = [torch.cuda.mem_get_info(i)[0] / 2**30 for i in range(n)]
        if min(free) >= need or time.time() - t0 > hours * 3600:
            return
        log(f"  waiting for GPU memory: {need:.0f} GB needed per GPU, free {[f'{x:.0f}' for x in free]} GB; checking every 5 min")
        time.sleep(300)


def load_model(repo):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    wait_for_gpus(repo)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(repo)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    split = torch.cuda.device_count() > 1                        # 2 GPUs visible: the model split over both
    model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16, device_map="auto" if split else "cuda").eval()
    log(f"loaded {repo} in {time.time() - t0:.0f}s on {torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}")
    return tok, model


def generate(tok, model, prompts, batch, max_new_tokens=C.MAX_NEW_TOKENS):
    """Greedy answers, batches of `batch` prompts in the given order."""
    import torch
    answers = []
    with torch.no_grad():
        for i in range(0, len(prompts), batch):
            enc = tok(prompts[i:i + batch], return_tensors="pt", padding=True).to(model.device)
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
            answers += tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    return answers


def release_gpu():
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()
