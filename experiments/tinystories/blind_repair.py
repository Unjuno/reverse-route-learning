#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from reverse_route_learning.runtime import TinyStoriesNeo
from poc_scan import train_controlled_post


def main():
    parser = argparse.ArgumentParser(description="Repair scanner-selected candidates without using B identity")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--branches", type=Path, default=Path("data/tinystories_semantic_branches.json"))
    parser.add_argument("--scan", type=Path, required=True)
    parser.add_argument("--case", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--sft-steps", type=int, default=8)
    parser.add_argument("--sft-lr", type=float, default=4e-6)
    parser.add_argument("--repair-lr", type=float, default=1e-6)
    parser.add_argument("--max-repair-steps", type=int, default=40)
    parser.add_argument("--candidate-mass-stop", type=float, default=0.35)
    parser.add_argument("--kl-budget", type=float, default=None, help="Optional base-free KL stopping threshold")
    parser.add_argument("--output", type=Path, default=Path("results/tinystories/repro_repair.json"))
    args = parser.parse_args()

    cases = json.loads(args.branches.read_text())
    scan = json.loads(args.scan.read_text())
    rec = next(x for x in scan if x["case"] == args.case)
    case = cases[args.case]
    ranked = sorted(rec["rows"], key=lambda r: r["late_js_min"])
    selected = [r["candidate_id"] for r in ranked[: args.top_k]]

    base, post = train_controlled_post(args.model, case, args.sft_steps, args.sft_lr, 12)
    prefix = torch.tensor(case["prefix_ids"], dtype=torch.long)[None, :]
    with torch.no_grad():
        z0 = post(prefix)[0, -1]
        p0 = F.softmax(z0, -1)
        # B is evaluation-only below, never used to select candidates or define the loss.
        b = case["B_id"]
        base_p_b = float(F.softmax(base(prefix)[0, -1], -1)[b])
        post_p_b = float(p0[b])

    post.train()
    optimizer = torch.optim.SGD(post.parameters(), lr=args.repair_lr)
    steps, final_kl = 0, 0.0
    for step in range(args.max_repair_steps):
        optimizer.zero_grad()
        z = post(prefix)[0, -1]
        lp = F.log_softmax(z, -1)
        loss = -torch.logsumexp(lp[torch.tensor(selected)], dim=0)
        loss.backward()
        optimizer.step()
        steps = step + 1
        with torch.no_grad():
            pf = F.softmax(post(prefix)[0, -1], -1)
            mass = float(pf[torch.tensor(selected)].sum())
            final_kl = float((p0 * (p0.clamp_min(1e-30).log() - pf.clamp_min(1e-30).log())).sum())
        if args.kl_budget is not None and final_kl >= args.kl_budget:
            break
        if args.kl_budget is None and mass >= args.candidate_mass_stop:
            break

    post.eval()
    with torch.no_grad():
        pf = F.softmax(post(prefix)[0, -1], -1)
        final_p_b = float(pf[b])
    denom = base_p_b - post_p_b
    recovery = (final_p_b - post_p_b) / denom if abs(denom) > 1e-12 else float("nan")
    result = {
        "case": args.case,
        "A_text": case["A_text"],
        "B_text": case["B_text"],
        "selected_candidate_ids": selected,
        "B_selected": b in selected,
        "base_B_p_oracle": base_p_b,
        "post_B_p_oracle": post_p_b,
        "final_B_p_oracle": final_p_b,
        "recovery_fraction_oracle": recovery,
        "entry_KL_post_to_final": final_kl,
        "repair_steps": steps,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
