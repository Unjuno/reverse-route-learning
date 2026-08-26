# Reverse Route Learning

**Proof-of-concept research on discovering and reopening model routes suppressed by post-training through counterfactual trajectory probing.**

> **Status: v0.1 experimental.** The repository shows that, in controlled TinyStories-8M experiments, some strongly suppressed semantic continuations remain operationally accessible behind an entry barrier and can be partially reopened using signals from the post-trained checkpoint. It does **not** claim exact reconstruction of a missing pretrained checkpoint, arbitrary automatic route discovery, or a universal recovery method.

## TL;DR

At a supplied semantic prefix, A-only post-training can make an alternative continuation B much less likely even when forcing B still exposes a usable downstream continuation. In the public experiments, the scanner is given the **post-trained checkpoint and the prefix**, but not the base checkpoint or the identity of the known suppressed B token. It ranks subdominant alternatives by counterfactual trajectory behavior, then locally raises a small scanner-selected candidate set.

Under the current fixed TinyStories-8M protocol, the clearest case (`town -> village`) recovers about **73% of the base-vs-post entry-probability gap** under an entry-distribution KL budget of about `0.05`. Other still-usable routes are missed by the current scanner. The intended result is therefore an **existence proof with a documented failure boundary**.

## Core distinction

The project separates:

- **accessibility** — whether the post-trained model naturally enters a semantic route;
- **conditional capability** — whether the model can still continue coherently after that route is externally entered.

A route can lose accessibility before its downstream conditional generator is fully erased.

The working loop is:

1. choose or observe a candidate prefix;
2. inspect subdominant next-token competitors in the post-trained model;
3. force-cross each candidate boundary;
4. measure downstream distribution / representation behavior;
5. rank plausible latent alternatives using post-only signals;
6. locally reopen a small scanner-selected candidate set;
7. re-observe when multiple barriers may exist.

The base checkpoint is used in the controlled experiments to construct the post-training condition and to compute oracle evaluation metrics **afterward**. It is not used by candidate ranking or repair-target selection.

## Current capability boundary

The primary v0.1 benchmark is [`docs/current-capability-boundary.md`](docs/current-capability-boundary.md), produced by [`experiments/tinystories/capability_boundary.py`](experiments/tinystories/capability_boundary.py).

| Route | Condition | Selected? | Oracle recovery at ~0.05 entry KL |
| --- | --- | --- | ---: |
| `town -> village` | shallow / reconvergent | yes | **73.1%** |
| `little -> big` | shallow / divergent | yes | **29.6%** |
| `boy -> girl` | shallow / reconvergent | yes | **15.9%** |
| `Lily -> Lucy` | shallow / reconvergent | yes | **8.2%** |
| `park -> slide` | residual capability remains | no | 4.1% |
| `play -> meet` | residual capability remains | no | 1.9% |
| `he -> it` | deep-eroded | yes | 2.4% |
| `see -> be` | deep-eroded | no | 0.7% |

This is the current boundary of the public method: **some suppressed routes are found and partially reopened, some still-usable routes are false negatives, and historical recoverability is not reliably identifiable from the final checkpoint alone.**

## Main findings

- Controlled A-only SFT can strongly suppress B-entry probability while a forced-B continuation retains substantial support.
- Known-route local repair can be much more supervision-efficient than full-sequence retraining in synthetic controls.
- Extreme low-logit search is not a useful route detector by itself.
- Counterfactual distribution / hidden-state reconvergence is a strong signal for some route classes.
- Reconvergence is not universal: some coherent alternatives remain on a distinct manifold.
- A post-only KL threshold can bound intervention strength, but it does not estimate the historical base probability.
- Across a TinyStories strength sweep, deeper entry suppression required more repair steps for 90% oracle recovery (`Spearman rho ~= 0.849`, `p ~= 4.8e-4`).
- Intrinsic trajectory normality, gradient alignment, candidate-graph centrality, cross-context direction consistency, and micro-repair response did **not** reliably establish historical provenance.

See [`docs/findings.md`](docs/findings.md) for the consolidated experiment record and [`docs/limitations.md`](docs/limitations.md) for non-claims.

## Reproduce the fixed benchmark

### 1. Install

```bash
git clone https://github.com/Unjuno/reverse-route-learning.git
cd reverse-route-learning
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Download the pinned TinyStories-8M checkpoint

```bash
python scripts/download_tinystories.py
```

The downloader pins the model revision and verifies the expected SHA256. Model weights are not redistributed by this repository. Exact assumptions are recorded in [`docs/reproducibility.md`](docs/reproducibility.md).

### 3. Run the v0.1 capability-boundary benchmark

```bash
python experiments/tinystories/capability_boundary.py \
  --model models/TinyStories-8M/pytorch_model.bin \
  --output results/tinystories/repro_capability_boundary.json
```

The reference benchmark uses pre-tokenized branch definitions, so a tokenizer is not required for this reproduction.

## Earlier exploratory PoC

The repository also retains the earlier fresh scan and blind-repair scripts:

```bash
python experiments/tinystories/poc_scan.py \
  --model models/TinyStories-8M/pytorch_model.bin \
  --cases 0 1 2 3 \
  --output results/tinystories/repro_scan.json

python experiments/tinystories/blind_repair.py \
  --model models/TinyStories-8M/pytorch_model.bin \
  --scan results/tinystories/repro_scan.json \
  --case 2 \
  --top-k 3 \
  --kl-budget 0.05 \
  --output results/tinystories/repro_repair_case2.json
```

These TB98/TB99 scripts established fresh Transformer candidate ranking and causal reopening, but their numerical settings differ from the later TB114 fixed benchmark. Do not compare their closure or recovery values as if they came from the same protocol.

## Repository map

- `src/reverse_route_learning/` — minimal direct-PyTorch TinyStories GPT-Neo runtime and metrics.
- `experiments/tinystories/` — public reproduction scripts.
- `data/` — pre-tokenized semantic branch definitions.
- `results/tinystories/` — selected raw result JSONs.
- `docs/method.md` — method and terminology.
- `docs/findings.md` — consolidated findings.
- `docs/current-capability-boundary.md` — primary fixed-protocol success/failure table.
- `docs/limitations.md` — non-claims and failure classes.
- `docs/reproducibility.md` — checkpoint, runtime, and environment assumptions.
- `docs/experiment-log.md` — compact chronology of the TB experiment series.

## Scope and non-claims

Supported by the current experiments:

- post-training can create a large accessibility drop while measurable downstream conditional capability remains;
- at a supplied prefix, post-only counterfactual trajectory signals can rank some suppressed alternatives usefully on a real 8M Transformer;
- scanner-selected candidate sets can causally reopen some suppressed routes;
- repair magnitude can be bounded without consulting the base checkpoint during repair.

Not established:

- automatic discovery of arbitrary suppressed capabilities or their locations;
- exact reconstruction of a missing pretrained checkpoint;
- reliable historical attribution of every candidate route to a particular unknown pretraining state;
- universal recovery across models or post-training recipes;
- efficient scaling to large language models;
- a guarantee that a coherent post-only alternative was historically present before post-training.

Current unresolved cases are indexed in [Issue #1](https://github.com/Unjuno/reverse-route-learning/issues/1).

## Model, license, and citation

The experiments use the public [`roneneldan/TinyStories-8M`](https://huggingface.co/roneneldan/TinyStories-8M) checkpoint. Upstream model licensing remains governed by that repository.

Project code and original documentation are licensed under **Apache-2.0**. See [`LICENSE`](LICENSE). Citation metadata is available in [`CITATION.cff`](CITATION.cff).

## Research status

This repository is an exploration record and proof of concept. Its purpose is to document a reproducible phenomenon, a workable intervention in some cases, and the cases where the current method fails.
