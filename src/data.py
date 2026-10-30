import csv
import hashlib
import json
import os
import random
import urllib.request

BBQ_URL = "https://raw.githubusercontent.com/nyu-mll/BBQ/main"
CATEGORIES = [
    "Age", "Disability_status", "Gender_identity", "Nationality", "Physical_appearance",
    "Race_ethnicity", "Religion", "SES", "Sexual_orientation",
]
# (ambiguous, disambiguated) items per category
SPLIT_SIZES = {"extract": (50, 50), "dev": (30, 30), "test": (150, 150)}
TEMPLATE_SHARES = {"extract": 0.4, "dev": 0.2}
SPLITS_DIR = "data/splits"


def download_bbq(raw_dir="data/bbq_raw"):
    os.makedirs(raw_dir, exist_ok=True)
    files = {"additional_metadata.csv": f"{BBQ_URL}/supplemental/additional_metadata.csv"}
    files.update({f"{c}.jsonl": f"{BBQ_URL}/data/{c}.jsonl" for c in CATEGORIES})
    for name, url in files.items():
        path = os.path.join(raw_dir, name)
        if not os.path.exists(path):
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, path)


def load_target_loc(raw_dir="data/bbq_raw"):
    out = {}
    with open(os.path.join(raw_dir, "additional_metadata.csv"), encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if "_x_" in row["category"]:
                continue
            raw = row["target_loc"].strip()
            out[(row["category"], int(row["example_id"]))] = None if raw in ("", "NA") else int(float(raw))
    return out


def unknown_index(item):
    for i in range(3):
        info = item["answer_info"][f"ans{i}"]
        if len(info) > 1 and info[1] == "unknown":
            return i
    return None


def load_category(cat, raw_dir="data/bbq_raw", target_loc=None):
    """BBQ items with the stereotyped (target_loc), other and unknown answer indices; items without a target are dropped."""
    target_loc = target_loc or load_target_loc(raw_dir)
    items = []
    with open(os.path.join(raw_dir, f"{cat}.jsonl"), encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            unk = unknown_index(item)
            st = target_loc.get((cat, int(item["example_id"])))
            if unk is None or st is None or st == unk:
                continue
            item["unknown_ans_idx"] = unk
            item["biased_ans_idx"] = st
            item["anti_ans_idx"] = next(i for i in range(3) if i not in (unk, st))
            items.append(item)
    return items


def _round_robin(by_template, template_ids, condition, n):
    queues = {t: [it for it in by_template[t] if it["context_condition"] == condition] for t in template_ids}
    picked = []
    while len(picked) < n and any(queues.values()):
        for t in template_ids:
            if queues[t] and len(picked) < n:
                picked.append(queues[t].pop())
    return picked


def build_splits(raw_dir="data/bbq_raw", out_dir=SPLITS_DIR, seed=42):
    """Split each category by template (question_index) into extract / dev / test."""
    rng = random.Random(seed)
    target_loc = load_target_loc(raw_dir)
    splits = {name: [] for name in SPLIT_SIZES}
    for cat in CATEGORIES:
        by_template = {}
        for it in load_category(cat, raw_dir, target_loc):
            by_template.setdefault(str(it["question_index"]), []).append(it)
        for t in sorted(by_template):
            rng.shuffle(by_template[t])
        ids = sorted(by_template, key=int)
        rng.shuffle(ids)
        n_ext = max(1, round(TEMPLATE_SHARES["extract"] * len(ids)))
        n_dev = max(1, round(TEMPLATE_SHARES["dev"] * len(ids)))
        assigned = {"extract": ids[:n_ext], "dev": ids[n_ext:n_ext + n_dev], "test": ids[n_ext + n_dev:]}
        for name, (n_amb, n_dis) in SPLIT_SIZES.items():
            splits[name] += (_round_robin(by_template, assigned[name], "ambig", n_amb)
                             + _round_robin(by_template, assigned[name], "disambig", n_dis))
    validate_splits(splits)

    os.makedirs(out_dir, exist_ok=True)
    manifest = {"seed": seed, "files": {}}
    for name, items in splits.items():
        path = os.path.join(out_dir, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for it in items:
                f.write(json.dumps(it) + "\n")
        manifest["files"][name] = {"n_items": len(items), "sha256": sha256(path)}
        print(f"{name}: {len(items)} items")
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return splits


def validate_splits(splits, min_templates=2):
    keys = [(it["category"], int(it["example_id"])) for items in splits.values() for it in items]
    assert len(keys) == len(set(keys)), "an item appears twice"
    templates = {}
    for name, items in splits.items():
        assert {it["category"] for it in items} == set(CATEGORIES), f"{name}: missing categories"
        for it in items:
            assert len({it["biased_ans_idx"], it["unknown_ans_idx"], it["anti_ans_idx"]}) == 3, it["example_id"]
        if name != "extract":
            for cat in CATEGORIES:
                n = len({it["question_index"] for it in items if it["category"] == cat})
                assert n >= min_templates, f"{name}/{cat}: {n} template(s)"
        templates[name] = {(it["category"], it["question_index"]) for it in items}
    names = list(templates)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not templates[a] & templates[b], f"templates shared by {a} and {b}"


def load_split(name, splits_dir=SPLITS_DIR):
    with open(os.path.join(splits_dir, f"{name}.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def permutations(item):
    """The three cyclic orderings of the options, with the position of each answer type."""
    options = [item["ans0"], item["ans1"], item["ans2"]]
    out = []
    for order in ([0, 1, 2], [1, 2, 0], [2, 0, 1]):
        out.append({
            "order": order,
            "options": [options[i] for i in order],
            "correct_pos": order.index(item["label"]),
            "biased_pos": order.index(item["biased_ans_idx"]),
            "unknown_pos": order.index(item["unknown_ans_idx"]),
            "anti_pos": order.index(item["anti_ans_idx"]),
        })
    return out
