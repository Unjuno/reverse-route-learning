# Limitations and Non-Claims

This project is intentionally narrow about what the current evidence supports.

## Not exact checkpoint reconstruction

The experiments do not reconstruct the exact weights or complete behavior of a missing pretrained checkpoint. “Reopening” means increasing accessibility to a candidate route that is coherent in the final model.

## Controlled evaluation still uses a base oracle

The base checkpoint is needed in the experiments to create known post-training conditions and to measure whether the known suppressed branch was recovered. It is not used by the blind ranking or candidate-set repair objective.

## Historical provenance is not generally identified

A final checkpoint may admit multiple coherent alternatives that can be locally strengthened. The current post-only signals do not always determine whether a route is specifically inherited from an unknown pre-post-training checkpoint or is simply easy to learn or reinforce now.

Historical fingerprints that did not separate the tested conditions reliably include:

- intrinsic forced-trajectory normality;
- tiny local repair gain;
- gradient alignment;
- candidate causal-graph centrality;
- cross-context hidden-direction consistency;
- cross-context micro-repair transfer.

## Reconvergence is incomplete

Return toward the default/post trajectory is a strong signal for some routes, but other coherent routes remain on a distinct manifold. Extending the horizon blindly also increases false positives.

## Small model and controlled post-training

The real-model results currently use TinyStories-8M and controlled asymmetric SFT. They do not establish behavior on large language models, RLHF/DPO pipelines, arbitrary domain fine-tuning, or production post-training stacks.

## Search compute is not optimized

Counterfactual candidate rollouts can be expensive. The repository prioritizes demonstrating the phenomenon and the intervention principle, not minimizing FLOPs. Search-cost scaling and efficient approximations remain unresolved.

## Repair can overshoot

Aggressive candidate-set repair can move the entry distribution substantially beyond the base oracle. A KL stopping threshold helps control intervention size but does not guarantee recovery to the historical base state.

## Current false negatives

Under the fixed v0.1 protocol, routes such as `park -> slide` and `play -> meet` retain measurable forced downstream capability but are not selected by the present Top-3 union detector. These cases are part of the documented capability boundary rather than excluded from the results.

## Safety and deployment

This is mechanistic research code, not a production model-repair system. Users are responsible for evaluating downstream model behavior and applicable deployment constraints.
