import json
import os
import random

from src.scoring import chat

MMLU_PATH = "data/mmlu_1k.jsonl"
MMLU_URL = "https://huggingface.co/datasets/cais/mmlu/resolve/main/all/test-00000-of-00001.parquet"
SYSTEM = ("You are a helpful and precise assistant. Answer the multiple-choice question by selecting only the "
          "single correct letter (A, B, C, or D).")


def build_subset(n_total=1000, seed=42, path=MMLU_PATH):
    """Stratified MMLU test subset: floor(n / 57) questions per subject, the remainder on random subjects."""
    import pandas as pd
    df = pd.read_parquet(MMLU_URL)
    rng = random.Random(seed)
    subjects = sorted(df["subject"].unique())
    per = n_total // len(subjects)
    extra = set(rng.sample(subjects, n_total - per * len(subjects)))
    rows = []
    for subject in subjects:
        idx = sorted(df.index[df["subject"] == subject].tolist())
        for i in sorted(rng.sample(idx, per + (subject in extra))):
            r = df.loc[i]
            rows.append({"mmlu_id": int(i), "subject": subject, "question": r["question"],
                         "choices": [str(c) for c in r["choices"]], "answer": int(r["answer"])})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"mmlu: {len(rows)} questions from {len(subjects)} subjects")


def load_questions(path=MMLU_PATH):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def mmlu_prompt(q, tokenizer):
    options = "\n".join(f"{letter}. {c}" for letter, c in zip("ABCD", q["choices"]))
    user = (f"Subject: {q['subject'].replace('_', ' ')}\n\nQuestion: {q['question']}\n\nOptions:\n{options}\n\n"
            f"Answer with the single letter of the correct option (A, B, C, or D):")
    return chat(tokenizer, SYSTEM, user)
