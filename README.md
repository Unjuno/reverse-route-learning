# Reverse Route Learning

**Experimental research code for discovering and reopening latent model routes suppressed by post-training through counterfactual trajectory probing.**

> Status: **v0.1 experimental / proof of concept.** The project demonstrates route suppression, post-only candidate discovery, and partial reopening on controlled TinyStories-8M experiments. It does **not** claim exact reconstruction of an unknown pretrained checkpoint.

## What this project studies

Fine-tuning can make a model stop choosing a semantic continuation even when a compatible downstream generator remains usable if the suppressed branch is forced. This project studies that gap between **accessibility** and **capability**.

The working procedure is:

1. create or observe a post-trained checkpoint;
2. inspect subdominant next-token competitors;
3. force-cross candidate boundaries;
4. measure the downstream trajectory (distribution / hidden-state geometry);
5. rank plausible latent routes using post-only signals;
6. locally reopen a small candidate set;
7. re-observe the model and repeat when multiple barriers exist.

The base checkpoint is used in controlled experiments to construct post-training conditions and to score recovery **afterward**. It is not used by the blind candidate ranking or repair target selection.

## Main empirical results so far

### TinyStories-8M Transformer

Fresh controlled A-only SFT on `roneneldan/TinyStories-8M` produced strong route suppression while forced downstream capability remained in multiple semantic branches.

A fresh post-only reconvergence scan over 64 subdominant candidates found the known suppressed branch at:

| Branch | Closure factor | post-only late-JS rank |
| --- | ---: | ---: |
| `Lily -> Lucy` | 86.5x | 11 / 64 |
| `woods -> forest` | 2.0x | 14 / 64 |
| `town -> village` | 8.1x | **2 / 64** |
| `boy -> girl` | 83.6x | **3 / 64** |

For `village` and `girl`, the scanner-selected top-3 candidate set was repaired **without using the identity of B in the repair objective**. The true suppressed branch reopened strongly, although the first naive repair overshot the original base probability. A post-only KL stopping threshold then provided a controllable intervention budget.

Examples from the initial blind repair:

- `town -> village`: `p(B)` 0.0203 after SFT -> 0.6228 after blind candidate-set repair (base oracle: 0.1646).
- `boy -> girl`: `p(B)` 0.00554 after SFT -> 0.7422 after blind candidate-set repair (base oracle: 0.4631).

These are **route reopening** results, not exact restoration results.

### Repair cost vs. suppression depth

Across `Lucy`, `village`, and `girl`, increasing SFT strength increased the number of local repair steps required to recover 90% of the base-vs-post entry gap. Across 12 conditions, closure factor vs. repair steps had Spearman `rho ~= 0.849` (`p ~= 4.8e-4`).

This supports a distinction between:

- whether a downstream route remains usable; and
- how expensive its entry barrier is to reopen.

### Multi-barrier and synthetic controls

Controlled synthetic experiments showed that suppression can appear as a strong primary barrier plus non-contiguous secondary barriers. One-pass sequential repair can re-close an earlier barrier; adaptive re-observation avoids this failure in the tested system.

Distribution reconvergence (Jensen-Shannon distance between next-token distributions after a forced crossing and a reference trajectory) outperformed exact token-string reconvergence in controlled systems and transferred partially to TinyStories.

## Important negative results

This repository intentionally keeps the failure modes visible:

- **Extreme low-logit search is not enough.** Random irrelevant tokens are often much lower probability than the suppressed semantic route.
- Candidate causal-graph centrality did not identify true routes in controlled tests.
- Gradient-direction alignment did not identify true routes.
- Immediate reconvergence is not universal; some routes reconverge only after a delay.
- Some coherent routes remain on a **different** manifold instead of returning to the default trajectory (`meet`, `big` examples).
- Intrinsic trajectory normality can find coherent semantic alternatives but does **not** reliably distinguish a historically preserved route from a route that has been deeply eroded and can simply be re-learned.
- Cross-context internal-direction consistency and cross-context repair transfer also failed as historical-recoverability discriminators.

The project therefore distinguishes **operational reopenability** from **historical recoverability**. See [docs/limitations.md](docs/limitations.md).

## Quick start

### 1. Install

```bash
git clone https://github.com/Unjuno/reverse-route-learning.git
cd reverse-route-learning
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Download TinyStories-8M weights

Model weights are not redistributed by this repository.

```bash
python scripts/download_tinystories.py
```

This downloads `roneneldan/TinyStories-8M/pytorch_model.bin` into `models/TinyStories-8M/`.

### 3. Reproduce the fresh scan

```bash
python experiments/tinystories/poc_scan.py \
  --model models/TinyStories-8M/pytorch_model.bin \
  --cases 0 1 2 3 \
  --output results/tinystories/repro_scan.json
```

The branch definitions are pre-tokenized and stored in `data/tinystories_semantic_branches.json`, so the core reproduction does not require a tokenizer.

### 4. Reopen scanner-selected candidates

For example, after scanning case 2 (`town -> village`):

```bash
python experiments/tinystories/blind_repair.py \
  --model models/TinyStories-8M/pytorch_model.bin \
  --scan results/tinystories/repro_scan.json \
  --case 2 \
  --top-k 3 \
  --kl-budget 0.05 \
  --output results/tinystories/repro_repair_case2.json
```

The repair objective uses only the scanner-selected candidate IDs. The known B token is read only after the run for controlled oracle evaluation.

## Repository map

- `src/reverse_route_learning/` — minimal TinyStories GPT-Neo runtime and metrics.
- `experiments/tinystories/` — public reproduction scripts.
- `data/` — pre-tokenized semantic branch definitions used by the experiments.
- `results/tinystories/` — selected raw result JSONs from the research run.
- `docs/method.md` — method and terminology.
- `docs/findings.md` — consolidated empirical findings.
- `docs/limitations.md` — what is not established.
- `docs/experiment-log.md` — compact chronology of the TB experiment series.

## Scope and claims

Supported by the current experiments:

- controlled post-training can strongly suppress semantic route accessibility while a downstream conditional generator remains usable;
- counterfactual post-only trajectory signals can rank some suppressed routes usefully on a real 8M Transformer;
- scanner-selected candidate sets can causally reopen real suppressed routes;
- repair strength can be bounded without using the base checkpoint, e.g. through a KL stopping threshold;
- reconvergence is useful but incomplete: not every surviving route returns to the default trajectory.

Not established:

- exact reconstruction of a missing pretrained checkpoint;
- reliable historical attribution of every candidate route to a particular unknown pretraining state;
- universal recovery across arbitrary models, post-training recipes, or capabilities;
- efficient scaling to large language models;
- a guarantee that a coherent post-only alternative was historically present before post-training.

## Model and license

The experiments use the public `roneneldan/TinyStories-8M` checkpoint. Model licensing remains governed by its upstream repository.

Project code and original documentation in this repository are licensed under **Apache-2.0**. See [LICENSE](LICENSE).

## Research status

This is an exploratory public research repository. Results and interfaces may change as experiments continue. Pull requests that improve reproduction, compute efficiency, alternative route metrics, or cross-model validation are welcome.
