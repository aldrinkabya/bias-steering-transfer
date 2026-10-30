# Bias-Steering Transfer to Quantized LLMs

Code for "[Disentangling Debiasing from Abstention in Bias-Steering Transfer to Quantized Language Models](https://aldrinkabya.github.io/papers/Disentangling_Debiasing_from_Abstention_in_Bias_Steering_Transfer_to_Quantized_Language_Models.pdf)".

## Overview

We test whether a bias-steering vector extracted on the FP16 model still works on quantized versions of that model (INT8, NF4, GPTQ-4 and HQQ at 4, 3 and 2 bits), and whether it reduces stereotyping or only makes the model answer "unknown" more often. Two vectors are compared, abstention (ABS) and counter-stereotype (CS), on BBQ and MMLU, with Mistral-7B-Instruct-v0.3, Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct.

## Repository layout

```
.
├── src/
│   ├── data.py          BBQ labels, template-level splits, option permutations
│   ├── mmlu.py          stratified 1,000-question MMLU subset
│   ├── models.py        loading at FP16, INT8, NF4, GPTQ-4 and HQQ-4/3/2
│   ├── scoring.py       prompts and answer-letter scoring
│   ├── steering.py      vector extraction and the steering hook
│   ├── selection.py     layer and dose selection rules
│   ├── plan.py          the runs of each stage
│   ├── evaluate.py      one BBQ or MMLU run
│   ├── metrics.py       answer shares, bias score, committed stereotyping, context accuracy
│   └── stats.py         template-clustered bootstrap, Wald tests, Holm correction
├── configs/
│   ├── models.yaml      checkpoints, candidate layers, versions per model
│   └── experiment.yaml  selection grid, random directions, doses, test families
├── data/                BBQ splits and the MMLU subset used in the paper
├── run.py               runs the stages for one model and precision
├── analyze.py           summary tables from all runs
├── tables.py            Tables 1-3
└── figures.py           transfer and answer-share figures
```

## Setup

```bash
pip install -r requirements.txt
huggingface-cli login   # Llama-3.1-8B-Instruct is gated
```

Python 3.11 or later and a CUDA GPU are required. FP16 models take about 16 GB of GPU memory and are spread over all visible GPUs; quantized versions are placed on one GPU (`--gpu`). HQQ quantizes on the fly from the FP16 weights, which needs about 16 GB of CPU RAM.

The splits in `data/splits/` and the MMLU subset in `data/mmlu_1k.jsonl` are the ones used in the paper. To rebuild them from the original sources:

```bash
python prepare_data.py
```

The split hashes in `data/splits/manifest.json` are unchanged by a rebuild. `python -m pytest tests` then checks the stereotype labels and the splits.

## Running

Each model and precision goes through five stages:

| Stage | What it does |
| :--- | :--- |
| `extract` | ABS and CS vectors for every layer, on the extraction split |
| `sweep` | development runs: FP16 over all candidate layers and doses; quantized versions at 0.5, 1 and 2 times the FP16 dose |
| `select` | applies the selection rules to the development runs |
| `test` | test split: unsteered, prompt, native and transferred steering, random directions, other doses |
| `mmlu` | MMLU with the same steering |

```bash
python run.py --model llama --prec fp16
python run.py --model llama --prec nf4
python run.py --model llama --prec hqq3 --stages test mmlu
```

FP16 has to be run first, since the quantized versions use its vectors and its selected settings. `scripts/run_all.sh` runs everything in the paper (553 runs) and then the analysis. Finished runs are skipped, so an interrupted stage can be restarted.

The main rule picks the setting whose bias score is closest to zero while losing at most 2 pp of context accuracy, and sets the ABS vector. The committed-stereotyping rule picks the setting whose committed stereotyping is closest to 50% and sets the CS vector.

Results are written to `results/`:

```
results/
├── vectors/     steering vectors and hidden-state norms
├── runs/        per-item predictions (.csv) and metrics (.json) of each run
├── selection/   selected layers and doses
├── summary/     analysis tables
└── figures/
```

## Analysis

```bash
python analyze.py
python tables.py
python figures.py
```

`analyze.py` compares every run with the unsteered model of the same version and every quantized model with FP16, with 95% intervals from a bootstrap over the test question templates. It also computes transfer ratios, ranks against the matched random directions, per-category changes, MMLU differences and vector cosines. `tables.py` prints Tables 1-3 and saves them to `results/summary/tables.md`.

## Reproducibility

Scoring is deterministic. Seeds are fixed for the splits (42), the random directions (1337-1346) and the 2,000 bootstrap resamples (2026). The paper's runs used NVIDIA T4 GPUs with PyTorch 2.10.0, Transformers 5.17.0, bitsandbytes 0.50.2, GPTQModel 7.5.0 and HQQ 0.2.8.post1, with batch sizes of 12 for FP16 and 16 for the quantized versions (half of that for extraction), which are the defaults here. Other GPUs or library versions can change individual answers slightly.

## Citation

```bibtex
@misc{biswas2026disentangling,
  title  = {Disentangling Debiasing from Abstention in Bias-Steering Transfer to Quantized Language Models},
  author = {Biswas, Aldrin Kabya and Hossan, Md. Zakir and Amin, Md. Ashraful and Rahman, AKM Mahbubur},
  year   = {2026}
}
```

## License

The code is released under the MIT License. BBQ is released under CC BY 4.0 and MMLU under the MIT License.
