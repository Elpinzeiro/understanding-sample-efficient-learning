"""The J-lens readout (jlens 0.1.0): at the last prompt token, the rank of the city's first token at every lens layer and
at the real output, and the top-10 tokens of each.

    layers   the lens's source layers minus the last one, + the model's output (last entry, layer n_layers - 1)
    gold     the first token of EVERY gold name: " " + name, except Chinese names (no space precedes a word in Chinese);
             Russian names get the space too. The rank is that of the best one: rank = #(logits > max over gold) + 1.
The lens gets its own tokenizer instance (jlens may switch BOS on)."""
from . import config as C
from .matching import CJK


class Lens:
    def __init__(self, repo, base):
        import jlens
        from transformers import AutoTokenizer
        self.jlens = jlens
        self.tok = AutoTokenizer.from_pretrained(repo)
        hf_repo, filename = C.MODELS[repo]["lens"]
        self.lens = jlens.JacobianLens.from_pretrained(hf_repo, filename=filename)
        self.n_layers = base.config.get_text_config().num_hidden_layers
        self.layers = [l for l in self.lens.source_layers if l != self.n_layers - 1]
        self.lens.jacobians = {l: J.to(base.device) for l, J in self.lens.jacobians.items()}

    def readout(self, model, recs):
        """Adds gt_rank and top10 to every record; returns the layers (the last one = the real output). Leaves every
        weight frozen and gradients off (jlens): a training loop must switch them on again."""
        import torch
        lm = self.jlens.from_hf(model, self.tok)
        for r in recs:
            gt = sorted({self.tok(n if CJK.search(n) else " " + n, add_special_tokens=False)["input_ids"][0]
                         for n in r["gold"] if n})
            with torch.no_grad():
                lens_logits, out_logits, _ = self.lens.apply(lm, r["prompt"], layers=self.layers, positions=[-1],
                                                             use_jacobian=True)
            r["gt_rank"], r["top10"] = [], []
            for lg in [lens_logits[l] for l in self.layers] + [out_logits]:
                lg = lg[0].float()
                r["gt_rank"].append(int((lg > lg[gt].max()).sum()) + 1)
                r["top10"].append([self.tok.decode([t]) for t in lg.topk(C.TOP_N).indices.tolist()])
        return self.layers + [lm.n_layers - 1]
