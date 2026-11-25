"""Every fixed setting of the paper, in one place."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]                       # the repository
DATA = ROOT / "data"                                             # the frozen dataset (see data.py)
RESULTS = Path(os.environ.get("FACTDEPTH_RESULTS", ROOT / "results"))   # what the GPU runs write, what the notebooks read
FIGURES = ROOT / "figures"

# ---------------------------------------------------------------------------------------------------------- results
# One folder per experiment, written by one config of experiments/configs/ (its `out`), read by the notebooks:
#   RESULTS/<key>/...      key = main, fig1, mlp_out, mlp_hidden, lr_sweep, lens_refit, lens_dist, known_ft
# Any key can be pointed elsewhere with FACTDEPTH_<KEY> (e.g. FACTDEPTH_KNOWN_FT=/data/known_ft).


def results(key, root=None):
    """Folder of one experiment's results: RESULTS/<key> (or <root>/<key>)."""
    if root is None and os.environ.get(f"FACTDEPTH_{key.upper()}"):
        return Path(os.environ[f"FACTDEPTH_{key.upper()}"])
    root = Path(root or RESULTS)
    return root / key


# ---------------------------------------------------------------------------------------------------------- models
NP = "neuronpedia/jacobian-lens"
MODELS = {   # HF id -> generation batch size, GPU memory a job may need (GB), pre-fitted J-lens (HF repo, file), name
    "Qwen/Qwen2.5-7B-Instruct": dict(
        batch=32, need_gb=30, name="Qwen2.5-7B", n_layers=28,
        lens=("anicka/jlens-qwen2.5-7b-instruct", "qwen2.5-7b-instruct_jlens.pt")),
    "google/gemma-3-12b-it": dict(
        batch=16, need_gb=50, name="Gemma-3-12B", n_layers=48,
        lens=(NP, "gemma-3-12b-it/jlens/Salesforce-wikitext/gemma-3-12b-it_jacobian_lens.pt")),
    "google/gemma-3-27b-it": dict(
        batch=16, need_gb=65, name="Gemma-3-27B", n_layers=62,
        lens=(NP, "gemma-3-27b-it/jlens/Salesforce-wikitext/gemma-3-27b-it_jacobian_lens.pt")),
}
QWEN, G12, G27 = "Qwen/Qwen2.5-7B-Instruct", "google/gemma-3-12b-it", "google/gemma-3-27b-it"
PAPER_MODELS = [QWEN, G12, G27]                                  # the order of every figure
RUN_ORDER = [G12, QWEN, G27]                                     # the order the main run took them
SWEEP_MODELS = [G12, QWEN, G27]                                  # the learning-rate sweep (App. B.2, Fig. 2)

# ---------------------------------------------------------------------------------------------------------- generation, readout
MAX_NEW_TOKENS = 32                                              # every test answer: greedy, 32 new tokens, no chat template
CHECK_TOKENS = 16                                                # the training-sentence check at each checkpoint
TOP_N = 10                                                       # tokens kept per layer by the readout
TOP_K = 3                                                        # answer layer = first layer with the city in the top 3

# ---------------------------------------------------------------------------------------------------------- injection
SEED = 0
LR = 2e-5
LORA = dict(r=8, lora_alpha=16, lora_dropout=0.0)
TARGETS = {                                                      # LoRA sites (target_modules regex)
    "qv": r"^(?!.*vision).*\.(q_proj|v_proj)$",                  # the paper's
    "mlp_out": r"^(?!.*vision).*\.down_proj$",                   # App. B.3
    "mlp_hidden": r"^(?!.*vision).*\.(gate_proj|up_proj)$",      # App. B.3 numbers (not in the figure)
}
TARGET_MODULES = {"qv": {"q_proj", "v_proj"}, "mlp_out": {"down_proj"}, "mlp_hidden": {"gate_proj", "up_proj"}}
SIZES = {"1": 1, "3": 3, "5": 5, "10": 10, "20": 20, "40": 40, "all": None, "86": None}   # training templates (None = all)
STOP_DROP = 0.25                                                 # learning-rate sweep: a run stops once it loses > 25%
STEPS = [32, 64, 128, 256]                                       # checkpoints of the main run (one training, read 4 times)

# ---------------------------------------------------------------------------------------------------------- dataset
PAIRS = ["zh", "ru"]                                             # English + Chinese, English + Russian
RELATIONS = ["P19", "P20", "P937"]
REL_NAME = {"P19": "born in", "P20": "died in", "P937": "worked in"}
SECOND = {"zh": "Chinese", "ru": "Russian"}
UNITS = [("zh", "P19"), ("ru", "P19"), ("zh", "P20"), ("ru", "P20"), ("zh", "P937"), ("ru", "P937")]   # queue order
