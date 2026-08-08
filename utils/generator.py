"""Stage 3: Generate CoT conversations using Qwen3 loaded directly via transformers."""

import sys as _sys, os as _os
_UTILS = _os.path.dirname(_os.path.abspath(__file__))
_SRC = _os.path.dirname(_UTILS)
for _p in (_SRC, _UTILS):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)


import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MODEL_NAME, MAX_NEW_TOKENS, TEMPERATURE, TOP_P


_model = None
_tokenizer = None


def load_model(device_map="auto", gpu_ids="1,3"):
    """Load Qwen3-30B-A3B-Instruct onto specified GPUs."""
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    os.environ["HF_HOME"] = "/data2/gdp2988/.cache/huggingface"
    os.environ["TRANSFORMERS_CACHE"] = "/data2/gdp2988/.cache/huggingface/hub"

    print(f"Loading tokenizer from {MODEL_NAME}...")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token
    _tokenizer.padding_side = "left"

    print(f"Loading model from {MODEL_NAME} onto GPUs {gpu_ids}...")
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
    )
    _model.eval()
    print("Model loaded.")
    return _model, _tokenizer


def _build_prompt(system_prompt: str, user_prompt: str) -> str:
    """Apply chat template to a single system+user message pair."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )


def generate_conversation(system_prompt: str, user_prompt: str) -> str:
    """Generate a single conversation from system+user prompts."""
    results = generate_batch(system_prompt, [user_prompt])
    return results[0]


def generate_batch(system_prompt: str, user_prompts: list[str]) -> list[str]:
    """Generate multiple conversations in a single batched forward pass."""
    model, tokenizer = _model, _tokenizer
    assert model is not None, "Call load_model() first"

    texts = [_build_prompt(system_prompt, up) for up in user_prompts]
    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True,
    ).to(model.device)

    prompt_lens = [
        inputs["attention_mask"][i].sum().item()
        for i in range(len(user_prompts))
    ]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
        )

    responses = []
    for i in range(len(user_prompts)):
        new_tokens = output_ids[i, inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        responses.append(text)

    return responses


def parse_json_response(raw: str) -> dict | None:
    """Extract JSON from model response, with fallback repair."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    brace_start = raw.find("{")
    brace_end = raw.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        try:
            return json.loads(raw[brace_start: brace_end + 1])
        except json.JSONDecodeError:
            pass

    return None
