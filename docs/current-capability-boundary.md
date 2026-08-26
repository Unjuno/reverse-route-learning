# Current TinyStories-8M capability boundary

This page records what the current public proof of concept can and cannot do on the present controlled TinyStories-8M setup. It is deliberately not a claim of universal recovery.

## Fixed protocol

For each condition we:

1. start from the same pinned TinyStories-8M base checkpoint;
2. create a controlled A-only post-training checkpoint;
3. expose only the post-trained checkpoint and the supplied prefix to the scanner;
4. inspect the top-32 subdominant competitors;
5. select the union of the top-3 reconvergence candidates and top-3 intrinsic-normality candidates;
6. reopen only this post-only selected set under an entry-distribution KL budget of approximately 0.05;
7. use the base model only afterward for oracle evaluation.

The table therefore measures **operational reopening under a fixed post-only heuristic at a supplied prefix**, not exact reconstruction or arbitrary location discovery.

## Metric definitions

For prefix `x` and controlled alternative token `B`:

- **closure** = `p_base(B | x) / p_post(B | x)`. Values above 1 mean post-training suppressed B at the entry.
- **forced suffix support** = the geometric-mean likelihood ratio `post/base` on a base-generated continuation after B is forcibly inserted. A value of 1 means equal conditional support; values above 1 are possible and mean the post-trained model assigns the forced suffix more probability than the base model.
- **oracle recovery** = `(p_final(B | x) - p_post(B | x)) / (p_base(B | x) - p_post(B | x))`. A value of 1 corresponds to recovering the base-vs-post entry gap; values can exceed 1 if repair overshoots.
- **entry KL** = `D_KL(p_post(. | x) || p_final(. | x))`, used as a post-only intervention budget.

These oracle metrics use the base checkpoint only after candidate selection / repair for controlled evaluation.

| Route | class | SFT steps | closure | forced suffix support | reconv. rank | intrinsic rank | B selected? | oracle recovery | entry KL |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| town → village | shallow_reconvergent | 8 | 3.6x | 1.85 | 2 | 1 | yes | 73.1% | 0.050 |
| park → slide | shallow_reconvergent | 8 | 155.5x | 0.76 | 24 | 6 | no | 4.1% | 0.055 |
| play → meet | shallow_divergent | 8 | 142.6x | 1.47 | 28 | 7 | no | 1.9% | 0.051 |
| little → big | shallow_divergent | 8 | 14.5x | 0.83 | 26 | 1 | yes | 29.6% | 0.063 |
| little → big | deep_eroded | 20 | 66.7x | 0.42 | 32 | 1 | yes | 18.1% | 0.054 |
| he → it | deep_eroded | 20 | 52.6x | 0.69 | 3 | 2 | yes | 2.4% | 0.058 |
| see → be | deep_eroded | 20 | 1187.1x | 0.50 | 23 | 5 | no | 0.7% | 0.055 |
| boy → girl | shallow_reconvergent | 8 | 52.0x | 1.68 | 3 | 1 | yes | 15.9% | 0.057 |
| Lily → Lucy | shallow_reconvergent | 8 | 29.9x | 2.84 | 5 | 3 | yes | 8.2% | 0.051 |

The labels `shallow_*` and `deep_eroded` describe controlled experimental regimes, not a post-only ground-truth classifier available to the method.

## What currently works

- Some strongly suppressed Transformer routes are discoverable and reopenable without using the base checkpoint for candidate selection or the repair objective.
- `town → village` is the clearest current example: the fixed protocol selected the hidden branch and recovered about **73%** of the base-vs-post entry gap at KL ≈ 0.05.
- Reconvergent routes such as `village` and `girl` are often ranked well by post-only trajectory geometry.
- Divergent coherent routes can sometimes be recovered through the intrinsic-normality branch of the detector (`little → big`).
- A KL trust region gives a base-free way to bound how far the intervention moves the post-trained model.

## What currently fails

- The fixed detector is not universal. `park → slide` and `play → meet` retained substantial forced downstream capability but were not selected by this Top-3 union protocol.
- Deep erosion cannot be identified reliably from semantic plausibility alone. Candidates can still look natural and even be selected while oracle recovery under the same KL budget is small.
- A coherent candidate being easy to strengthen does not prove that it is a historically preserved pre-post-training route.
- We do not currently recover exact pretrained probabilities, training history, arbitrary suppressed capabilities, or their locations automatically.

## Interpretation

The present evidence supports a narrower claim: **at supplied prefixes, some post-training accessibility barriers can be found and partially reopened from the post-trained Transformer itself.** The current search rule has meaningful false negatives and cannot, by itself, establish historical recoverability.

This is the intended boundary of the v0.1 proof of concept. See `docs/reproducibility.md` for the pinned checkpoint and runtime assumptions.
