#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from reverse_route_learning.metrics import adaptive_js_drop, js_divergence_from_logits
from reverse_route_learning.runtime import TinyStoriesNeo, kv_prefix_cache, kv_step


def train_controlled_post(model_path: Path, case: dict, steps: int, lr: float, suffix_tokens: int):
    base = TinyStoriesNeo(model_path).eval()
    prefix, a_id = case["prefix_ids"], case["A_id"]
    a_sequence = base.greedy(prefix + [a_id], suffix_tokens)
    post = TinyStoriesNeo(model_path).train()
    optimizer = torch.optim.AdamW(post.parameters(), lr=lr, weight_decay=0.0)
    ids = torch.tensor(a_sequence, dtype=torch.long)[None, :]
    for _ in range(steps):
        optimizer.zero_grad()
        logits = post(ids[:, :-1])
        target = ids[0, 1:]
        loss = F.cross_entropy(logits[0, len(prefix) - 1 :], target[len(prefix) - 1 :])
        loss.backward()
        optimizer.step()
    return base, post.eval()


@torch.no_grad()
def scan_case(post, case: dict, candidate_count: int, horizon: int):
    prefix = case["prefix_ids"]
    b_id = case["B_id"]
    z0 = post.last_logits(torch.tensor(prefix, dtype=torch.long))[0]
    probs = F.softmax(z0, dim=-1)
    top = torch.topk(probs, candidate_count + 1).indices.tolist()
    default = int(top[0])
    candidates = [int(x) for x in top[1:]]

    pc = kv_prefix_cache(post, prefix)
    dz, dc, dh = kv_step(post, pc, [default], return_hidden=True)
    refs_logits, refs_hidden = [], []
    curz, curc, curh = dz, dc, dh
    for depth in range(horizon):
        refs_logits.append(curz[0].clone())
        refs_hidden.append(curh[0].clone())
        if depth < horizon - 1:
            curz, curc, curh = kv_step(post, curc, curz.argmax(-1), return_hidden=True)

    cz, cc, ch = kv_step(post, pc, torch.tensor(candidates), return_hidden=True)
    curves = {t: {"js": [], "cos": []} for t in candidates}
    curz, curc, curh = cz, cc, ch
    for depth in range(horizon):
        refz = refs_logits[depth][None, :].expand(len(candidates), -1)
        refh = refs_hidden[depth][None, :].expand(len(candidates), -1)
        js = js_divergence_from_logits(curz, refz)
        cos = F.cosine_similarity(curh, refh, dim=-1)
        for j, token in enumerate(candidates):
            curves[token]["js"].append(float(js[j]))
            curves[token]["cos"].append(float(cos[j]))
        if depth < horizon - 1:
            curz, curc, curh = kv_step(post, curc, curz.argmax(-1), return_hidden=True)

    rows = []
    for token in candidates:
        js_curve = curves[token]["js"]
        drop, depth = adaptive_js_drop(js_curve)
        rows.append(
            {
                "candidate_id": token,
                "entry_rank": int((z0 > z0[token]).sum()) + 1,
                "entry_p": float(probs[token]),
                "js_curve": js_curve,
                "cos_curve": curves[token]["cos"],
                "best_js_drop": drop,
                "reconvergence_depth": depth,
                "late_js_min": min(js_curve[1:]) if len(js_curve) > 1 else js_curve[0],
                # Oracle-only field for controlled evaluation. It is never used for ranking.
                "is_known_B": token == b_id,
            }
        )
    for key, reverse in [("best_js_drop", True), ("late_js_min", False)]:
        ranked = sorted(rows, key=lambda r: r[key], reverse=reverse)
        for rank, row in enumerate(ranked, 1):
            row["rank_" + key] = rank
    return rows, default


def main():
    parser = argparse.ArgumentParser(description="Fresh TinyStories controlled-SFT + post-only reconvergence scan")
    parser.add_argument("--model", required=True, type=Path, help="Path to TinyStories-8M pytorch_model.bin")
    parser.add_argument("--branches", type=Path, default=Path("data/tinystories_semantic_branches.json"))
    parser.add_argument("--cases", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--sft-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=4e-6)
    parser.add_argument("--suffix-tokens", type=int, default=12)
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("results/tinystories/repro_scan.json"))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    cases = json.loads(args.branches.read_text())
    output = []
    for idx in args.cases:
        start = time.time()
        case = cases[idx]
        base, post = train_controlled_post(args.model, case, args.sft_steps, args.lr, args.suffix_tokens)
        rows, default = scan_case(post, case, args.candidates, args.horizon)
        b_rows = [r for r in rows if r["is_known_B"]]
        with torch.no_grad():
            prefix = torch.tensor(case["prefix_ids"], dtype=torch.long)
            b = case["B_id"]
            base_z = base.last_logits(prefix)[0]
            post_z = post.last_logits(prefix)[0]
            closure = float(torch.exp(F.log_softmax(base_z, -1)[b] - F.log_softmax(post_z, -1)[b]))
        record = {
            "case": idx,
            "original_case_id": case.get("original_case_id"),
            "prefix": case["prefix"],
            "A_text": case["A_text"],
            "B_text": case["B_text"],
            "default_id": default,
            "closure_factor": closure,
            "B_in_competitor_pool": bool(b_rows),
            "B_row": b_rows[0] if b_rows else None,
            "rows": rows,
            "seconds": time.time() - start,
        }
        output.append(record)
        b = record["B_row"]
        print(
            f"case={idx} {case['A_text']!r}->{case['B_text']!r} closure={closure:.2f}x "
            f"B_lateJS_rank={None if b is None else b['rank_late_js_min']}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
