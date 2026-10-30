import json
import os

import torch
import yaml
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

PRECISIONS = ["fp16", "int8", "nf4", "gptq4", "hqq4", "hqq3", "hqq2"]
HQQ_GROUP_SIZE = 64


def model_config(name, path="configs/models.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)[name]


def _gptq_checkpoint(repo_id):
    """Download a GPTQ checkpoint, replacing the legacy `is_marlin_format` key that GPTQModel 7 rejects."""
    path = snapshot_download(repo_id, local_dir=os.path.join("checkpoints", repo_id.replace("/", "__")))
    for fn in ("quantize_config.json", "quant_config.json", "config.json"):
        p = os.path.join(path, fn)
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            cfg = json.load(f)
        q = cfg.get("quantization_config", {}) if fn == "config.json" else cfg
        if "is_marlin_format" in q:
            q.setdefault("format", "marlin" if q.pop("is_marlin_format") else "gptq")
            with open(p, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
    return path


def load_model(name, prec, gpu=0):
    cfg = model_config(name)
    repo_id = cfg["gptq_id"] if prec == "gptq4" else cfg["hf_id"]
    tokenizer = AutoTokenizer.from_pretrained(repo_id, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = f"cuda:{gpu}"
    if prec == "fp16":
        model = AutoModelForCausalLM.from_pretrained(repo_id, dtype=torch.float16, device_map="auto")
    elif prec in ("int8", "nf4"):
        quant = (BitsAndBytesConfig(load_in_8bit=True) if prec == "int8" else
                 BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16))
        model = AutoModelForCausalLM.from_pretrained(repo_id, dtype=torch.float16, quantization_config=quant,
                                                     device_map={"": gpu})
    elif prec == "gptq4":
        from gptqmodel import BACKEND, GPTQModel
        model = GPTQModel.load(_gptq_checkpoint(repo_id), backend=BACKEND.TRITON, device=device).model
    elif prec.startswith("hqq"):
        from hqq.core.quantize import BaseQuantizeConfig
        from hqq.models.hf.base import AutoHQQHFModel
        from hqq.utils.patching import prepare_for_inference
        model = AutoModelForCausalLM.from_pretrained(repo_id, dtype=torch.float16, device_map="cpu")
        AutoHQQHFModel.quantize_model(model, quant_config=BaseQuantizeConfig(nbits=int(prec[3:]), group_size=HQQ_GROUP_SIZE),
                                      compute_dtype=torch.float16, device=device)
        model.to(device)
        prepare_for_inference(model)
    else:
        raise ValueError(f"unknown precision {prec}")
    return model.eval(), tokenizer
