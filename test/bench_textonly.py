"""Text-only LLM baseline on the SpatialMed annotation set.

Measures how much of the benchmark is answerable from language priors alone
(no CT volume). Chance = 25%; anything well above that indicates text bias.

Batched greedy decoding, short max_new_tokens (we only need a letter).

Usage:
  CUDA_VISIBLE_DEVICES=1,3 python bench_textonly.py --model Qwen/Qwen3.5-9B --out q35.json
"""

import sys, os
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
for _p in (_SRC, os.path.join(_SRC, "utils")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse, json, re, time
from collections import defaultdict

import torch

JSONL = ("/home/gdp2988/.claude/uploads/694ec789-20df-4d2a-921f-32cddeec2d71/"
         "bc8398d8-annotations.jsonl")


SYSTEM_PROMPT = (
    "You are an expert medical imaging assistant and answering this question. "
    "Return the final answer only.\n"
    "• If the options are A/B/C/D, respond exactly like: Answer: A\n"
    "Do not include explanations."
)


def build_prompt(q):
    return q.strip()


def parse_letter(t):
    """Prefer the requested 'Answer: X' format, else first standalone A-D."""
    t = t.strip()
    m = re.search(r"[Aa]nswer\s*[:：]\s*\(?([ABCD])\b", t)
    if m:
        return m.group(1)
    m = re.search(r"\b([ABCD])\b", t)
    return m.group(1) if m else (t[0] if t[:1] in "ABCD" else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--mode", choices=["logprob","greedy"], default="logprob",
                    help="logprob = constrained argmax over A-D (1 fwd pass); greedy = do_sample=False generation")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    rows = [json.loads(l) for l in open(JSONL)]
    if args.max_samples:
        rows = rows[:args.max_samples]
    print(f"{len(rows)} questions | model={args.model} | mode={args.mode}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True,
                                        padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    ).eval()

    prompts = []
    for r in rows:
        msg = [{"role": "system", "content": SYSTEM_PROMPT},
               {"role": "user", "content": build_prompt(r["questions"])}]
        p = None
        for kw in ({"enable_thinking": False}, {}):     # skip <think> when supported
            try:
                p = tok.apply_chat_template(msg, tokenize=False,
                                            add_generation_prompt=True, **kw)
                break
            except Exception:
                continue
        if p is None:
            p = SYSTEM_PROMPT + "\n\n" + build_prompt(r["questions"]) + "\n"
        # match the requested output format so the next token is the letter
        prompts.append(p + "Answer:")

    # Multiple choice => score the 4 letters directly (1 forward pass, no decoding).
    # Robust to "thinking" models and far faster than generating.
    letter_ids = []
    for L in "ABCD":
        cands = {tok.encode(v, add_special_tokens=False)[0]
                 for v in (L, " " + L) if tok.encode(v, add_special_tokens=False)}
        letter_ids.append(sorted(cands))
    print(f"  scoring token ids per letter: {letter_ids}", flush=True)

    preds, t0, unparsed = [], time.time(), 0
    for i in range(0, len(prompts), args.batch_size):
        b = prompts[i:i + args.batch_size]
        enc = tok(b, return_tensors="pt", padding=True, truncation=True,
                  max_length=1024).to(model.device)
        if args.mode == "greedy":
            # true greedy decoding: deterministic, no sampling
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                     do_sample=False, num_beams=1,
                                     temperature=None, top_p=None, top_k=None,
                                     pad_token_id=tok.pad_token_id)
            gen = out[:, enc["input_ids"].shape[1]:]
            for txt in tok.batch_decode(gen, skip_special_tokens=True):
                L = parse_letter(txt)
                if L is None:
                    unparsed += 1
                preds.append(L)
        else:
            with torch.no_grad():
                logits = model(**enc).logits[:, -1, :].float()  # next-token logits
            scores = torch.stack([logits[:, ids].max(dim=-1).values for ids in letter_ids], -1)
            preds += ["ABCD"[k] for k in scores.argmax(-1).tolist()]
        if (i // args.batch_size) % 10 == 0:
            done = min(i + args.batch_size, len(prompts))
            acc = sum(p == r["gt_answers"].strip() for p, r in zip(preds, rows)) / max(done, 1)
            el = time.time() - t0
            print(f"  [{done}/{len(prompts)}] acc={100*acc:.1f}%  "
                  f"{el:.0f}s  eta={el/max(done,1)*(len(prompts)-done):.0f}s", flush=True)

    by = defaultdict(lambda: [0, 0])
    recs = []
    for p, r in zip(preds, rows):
        ok = (p == r["gt_answers"].strip())
        for k in (r.get("task_type", "?"), "ALL"):
            by[k][0] += ok; by[k][1] += 1
        recs.append({"id": r["id"], "type": r.get("type"), "task_type": r.get("task_type"),
                     "gt": r["gt_answers"].strip(), "pred": p, "correct": bool(ok)})

    print(f"\n{'='*54}\nText-only: {args.model}\n{'='*54}")
    print(f"{'task_type':<14}{'correct':>9}{'total':>8}{'acc':>9}")
    for k in sorted(by, key=lambda x: (x == "ALL", x)):
        c, n = by[k]
        print(f"{k:<14}{c:>9}{n:>8}{100*c/n:>8.2f}%")
    print(f"(chance = 25.00%%)  mode={args.mode}" + (f"  unparsed={unparsed}" if args.mode=="greedy" else ""))
    json.dump({"model": args.model, "mode": args.mode,
               "overall": by["ALL"][0] / by["ALL"][1],
               "by_task": {k: {"correct": v[0], "total": v[1], "acc": v[0] / v[1]}
                           for k, v in by.items()},
               "records": recs}, open(args.out, "w"), indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
