import os
import random
from contextlib import contextmanager

import torch
from tqdm import tqdm

from src.data import permutations
from src.scoring import bbq_prompt, encode, left_pad, letter_tokens

VECTORS = ["abst", "cs"]


def vector_path(vectors_dir, model, prec, vector, layer):
    return os.path.join(vectors_dir, f"{model}_{prec}_{vector}_L{layer}.pt")


def contrast_items(items, vector, seed=42):
    """ABS: ambiguous items. CS: disambiguated items, balanced per category over which answer is correct."""
    if vector == "abst":
        return [it for it in items if it["context_condition"] == "ambig"]
    rng = random.Random(seed)
    out = []
    for cat in sorted({it["category"] for it in items}):
        dis = [it for it in items if it["category"] == cat and it["context_condition"] == "disambig"]
        st = [it for it in dis if it["label"] == it["biased_ans_idx"]]
        oth = [it for it in dis if it["label"] == it["anti_ans_idx"]]
        n = min(len(st), len(oth))
        rng.shuffle(st)
        rng.shuffle(oth)
        out += st[:n] + oth[:n]
    return out


@torch.no_grad()
def extract(model, tokenizer, items, vector, out_dir, model_name, prec, batch_size=8):
    """Mean hidden-state difference between the desired and the stereotyped answer letter, for every layer,
    and the mean hidden-state norm r_L over non-padding tokens except the first."""
    os.makedirs(out_dir, exist_ok=True)
    items = contrast_items(items, vector)
    pos_key = "unknown_pos" if vector == "abst" else "anti_pos"
    prompts, pos, neg = [], [], []
    for it in items:
        for p in permutations(it):
            prompts.append(bbq_prompt(it, p["options"], tokenizer))
            pos.append(p[pos_key])
            neg.append(p["biased_pos"])
    letters = letter_tokens(tokenizer, bbq_prompt(items[0], [items[0][f"ans{i}"] for i in range(3)], tokenizer))["primary"]
    ids = encode(tokenizer, prompts)
    seqs = {"pos": [x + [letters[k]] for x, k in zip(ids, pos)], "neg": [x + [letters[k]] for x, k in zip(ids, neg)]}

    n_layers = model.config.num_hidden_layers
    sums = {"pos": [0.0] * n_layers, "neg": [0.0] * n_layers}
    norm_sum, norm_count = [0.0] * n_layers, [0] * n_layers
    for side, batch in seqs.items():
        for i in tqdm(range(0, len(batch), batch_size), desc=f"{vector}/{side}"):
            enc = left_pad(tokenizer, batch[i:i + batch_size], model.device)
            hidden = model(**enc, output_hidden_states=True, logits_to_keep=1).hidden_states
            mask = enc["attention_mask"].clone()
            mask[torch.arange(mask.shape[0], device=mask.device), mask.float().argmax(dim=1)] = 0
            for L in range(n_layers):
                h = hidden[L + 1]
                sums[side][L] = sums[side][L] + h[:, -1].double().cpu().sum(dim=0)
                m = mask.to(h.device)
                norm_sum[L] += float((torch.linalg.vector_norm(h.float(), dim=-1) * m).sum())
                norm_count[L] += int(m.sum())

    n = len(prompts)
    for L in range(n_layers):
        v = ((sums["pos"][L] - sums["neg"][L]) / n).float()
        torch.save({"vector": v, "unit_vector": v / (torch.linalg.vector_norm(v) + 1e-12),
                    "residual_norm": norm_sum[L] / max(norm_count[L], 1), "num_pairs": n},
                   vector_path(out_dir, model_name, prec, vector, L))
    print(f"saved {model_name}/{prec} {vector} vectors ({n} pairs)")


def load_vector(vectors_dir, model, prec, vector, layer):
    return torch.load(vector_path(vectors_dir, model, prec, vector, layer), map_location="cpu")


def steering_vector(cond, model, prec, vector, layer, alpha, vectors_dir, seed=None):
    """alpha * r_L * unit direction. Transfer uses the FP16 vector and its r_L; a random direction uses the native r_L."""
    data = load_vector(vectors_dir, model, "fp16" if cond == "steer_transfer" else prec, vector, layer)
    if cond == "random":
        gen = torch.Generator().manual_seed(seed + layer)
        direction = torch.randn(data["unit_vector"].shape[-1], generator=gen)
        direction = direction / (torch.linalg.vector_norm(direction) + 1e-12)
    else:
        direction = data["unit_vector"]
    return alpha * float(data["residual_norm"]) * direction


def cosine(a, b):
    return float(a @ b / (torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b) + 1e-12))


@contextmanager
def steer(model, layer, vec):
    """Add vec to the output of decoder layer `layer` at every position."""
    if vec is None:
        yield
        return

    def hook(module, inputs, output):
        if isinstance(output, tuple):
            return (output[0] + vec.to(output[0].device, output[0].dtype),) + tuple(output[1:])
        return output + vec.to(output.device, output.dtype)

    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
