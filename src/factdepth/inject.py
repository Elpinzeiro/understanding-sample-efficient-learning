"""One injection: a fresh LoRA trained on one fact, evaluated at every checkpoint of the same training run.

    torch.manual_seed(seed) right before get_peft_model; LoRA r 8, alpha 16, dropout 0 on the site's modules;
    one batch = all training sentences (right-padded), loss on the answer tokens only (labels of tok(p) masked in
    tok(p + t), default special tokens), token_type_ids = zeros when the model's forward takes them (Gemma 3);
    AdamW(lr) on the LoRA parameters; no early stop on the loss.
At each checkpoint: the training sentences answered (16 tokens, greedy), the evaluation (evaluate.Session), the adapter
saved at the last checkpoint if asked, then the LoRA parameters unfrozen again (the J-lens readout freezes everything).
With stop_drop (the learning-rate sweep), training stops at the first checkpoint that loses more than that share of
the known facts (drop > stop_drop): train_and_evaluate returns that step, and the later checkpoints are not computed.
"""
import inspect
import time

from . import config as C
from .data import training_pairs


def train_and_evaluate(S, f, lang, size, steps, lr, site, path_of, save_adapter=None, stop_drop=None):
    """S: an evaluate.Session pointed at the fact's unit. path_of(step) -> result file. save_adapter: folder or None.
    Returns (step, drop) if the run stopped (stop_drop), else None."""
    import torch
    from peft import LoraConfig, get_peft_model
    tok, pairs = S.tok, training_pairs(f, lang, size, S.unit.pair)
    lora = dict(C.LORA, target_modules=C.TARGETS[site])
    torch.manual_seed(C.SEED)
    model = get_peft_model(S.base, LoraConfig(**lora))
    try:
        ids, labels = [], []
        for p, t in pairs:
            a, full = tok(p)["input_ids"], tok(p + t)["input_ids"]
            ids.append(full)
            labels.append([-100] * len(a) + full[len(a):])
        n, pad = max(map(len, ids)), tok.pad_token_id
        batch = {"input_ids": torch.tensor([x + [pad] * (n - len(x)) for x in ids]),
                 "attention_mask": torch.tensor([[1] * len(x) + [0] * (n - len(x)) for x in ids]),
                 "labels": torch.tensor([y + [-100] * (n - len(y)) for y in labels])}
        if "token_type_ids" in inspect.signature(S.base.forward).parameters:   # Gemma 3 wants it when training
            batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])
        batch = {k: v.to(S.base.device) for k, v in batch.items()}
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
        losses, t0, gold = [], time.time(), f["gold"] + f["gold_zh"]
        for step in range(1, max(steps) + 1):
            model.train()
            loss = model(**batch).loss
            loss.backward()
            opt.step()
            opt.zero_grad()
            losses.append(loss.item())
            if step in steps:
                model.eval()
                enc = tok([p for p, _ in pairs], return_tensors="pt", padding=True).to(S.base.device)
                with torch.no_grad():
                    gen = model.generate(**enc, max_new_tokens=C.CHECK_TOKENS, do_sample=False, pad_token_id=pad)
                texts = tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
                ok = sum(S.is_hit(t, gold) is not None for t in texts)
                train = {"lr": lr, "steps": step, "train_answered": f"{ok}/{len(pairs)}", "losses": losses[:],
                         "seconds": time.time() - t0, "lora": lora, "pairs": [{"prompt": p, "target": t} for p, t in pairs]}
                res = S.evaluate(model, f, lang, size, train, path_of(step))
                if stop_drop is not None and res["drop"] > stop_drop:
                    return step, res["drop"]
                if save_adapter and step == max(steps):
                    model.save_pretrained(save_adapter)
                torch.set_grad_enabled(True)                     # the J-lens readout freezes every weight,
                for name, p in model.named_parameters():         # LoRA included: unfreeze LoRA to keep training
                    p.requires_grad_("lora_" in name)
    finally:
        S.base = model.unload()                                  # remove LoRA, base weights untouched
