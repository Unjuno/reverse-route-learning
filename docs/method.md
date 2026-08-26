# Method

## Core distinction

Reverse Route Learning separates two quantities that ordinary task-level evaluation conflates:

- **accessibility**: whether the model naturally enters a candidate semantic route;
- **conditional capability**: whether the model can continue coherently after that route is externally entered.

A route can have very low entry probability after post-training while retaining a useful downstream conditional generator.

## Controlled experimental setup

A base TinyStories-8M checkpoint is used to create a controlled post-trained checkpoint through asymmetric A-only SFT. For a semantic branch `(A, B)` at a fixed prefix, training reinforces the A continuation and suppresses B indirectly.

The base checkpoint is retained for **oracle measurement only**: closure factors, suffix-retention ratios, and recovery relative to the original checkpoint. Blind ranking and repair selection operate on the post-trained model.

## Counterfactual boundary crossing

At a candidate prefix:

1. obtain the post-trained model's next-token distribution;
2. retain a set of subdominant competitors rather than searching only the extreme logit tail;
3. force each candidate token;
4. roll the model forward for a short or adaptive horizon;
5. measure downstream distribution and representation geometry.

### Distribution reconvergence

The main portable metric studied so far is Jensen-Shannon divergence between next-token distributions along two trajectories. In controlled settings, a suppressed route can initially move away from the default trajectory and later move back toward its distributional / representational manifold.

A simple adaptive score is the largest early-to-late decrease in a JS-distance curve.

Reconvergence is **not universal**. A coherent surviving route may remain on a distinct manifold. For those routes, intrinsic forced-trajectory normality can provide candidate recall, but it is not sufficient evidence of historical preservation.

## Local reopening

After ranking candidates, a local repair objective raises the aggregate probability mass of a small scanner-selected candidate set. The repair objective does not need the identity of the oracle B token.

Naive high-mass repair can overshoot. A post-only KL stopping threshold relative to the post-trained entry distribution provides a base-free intervention budget.

## Multiple barriers

Controlled synthetic experiments show that post-training can produce more than one accessibility barrier along a trajectory. Repairing barriers exactly once in a fixed order can re-close earlier barriers. The robust control loop is therefore:

`observe -> select deficient boundary -> repair -> re-observe -> repeat`.

## Operational vs historical claims

The method can test **operational reopenability**: whether a coherent candidate route can be found and reopened with bounded collateral drift from the final checkpoint.

It does not, by itself, prove **historical recoverability**: that the candidate route is specifically a surviving behavior of an unknown base checkpoint. Several attempted final-checkpoint fingerprints failed to distinguish shallow Hidden routes from more deeply eroded but still coherent alternatives.
