import collections
from functools import lru_cache

from src.data import CATEGORIES, load_category, load_split, load_target_loc, permutations, validate_splits

# (category, example_id) -> index of the stereotyped answer, checked against the item text
HAND_CHECKED = {
    ("Age", 0): 0,
    ("Gender_identity", 0): 2,
    ("SES", 0): 0,
    ("Physical_appearance", 0): 2,
    ("Religion", 0): 2,
    ("Sexual_orientation", 0): 2,
    ("Race_ethnicity", 1520): 1,
    ("Race_ethnicity", 1522): 2,
}


@lru_cache(maxsize=None)
def items(cat):
    return load_category(cat, target_loc=load_target_loc())


def test_hand_checked_labels():
    for (cat, eid), expected in HAND_CHECKED.items():
        item = next(it for it in items(cat) if it["example_id"] == eid)
        assert item["biased_ans_idx"] == expected, (cat, eid)


def test_polarity_flips_stereotyped_answer():
    for cat in CATEGORIES:
        groups = collections.defaultdict(dict)
        for it in items(cat):
            groups[(it["question_index"], it["context"], it["ans0"], it["ans1"], it["ans2"])][it["question_polarity"]] = \
                it["biased_ans_idx"]
        pairs = [g for g in groups.values() if len(g) == 2]
        assert pairs and all(g["neg"] != g["nonneg"] for g in pairs), cat


def test_splits():
    validate_splits({name: load_split(name) for name in ("extract", "dev", "test")})


def test_permutations():
    it = items("Religion")[0]
    for p in permutations(it):
        assert p["options"][p["biased_pos"]] == it[f"ans{it['biased_ans_idx']}"]
        assert p["options"][p["unknown_pos"]] == it[f"ans{it['unknown_ans_idx']}"]
        assert p["options"][p["anti_pos"]] == it[f"ans{it['anti_ans_idx']}"]
        assert p["options"][p["correct_pos"]] == it[f"ans{it['label']}"]
