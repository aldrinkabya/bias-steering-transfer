#!/usr/bin/env bash
# Every run in the paper. FP16 goes first: the quantized versions need its vectors and selected settings.
set -euo pipefail

for model in mistral llama; do
  for prec in fp16 int8 nf4 gptq4 hqq4 hqq3 hqq2; do
    python run.py --model "$model" --prec "$prec"
  done
done
for prec in fp16 gptq4; do
  python run.py --model qwen --prec "$prec"
done

python analyze.py
python tables.py
python figures.py
