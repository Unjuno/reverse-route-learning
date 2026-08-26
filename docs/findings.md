# Consolidated Findings

## 1. Accessibility can fail before downstream capability

In multiple TinyStories semantic branches, A-only SFT drove the B-entry probability down by large factors while a forced-B continuation retained substantial support. Stronger SFT increased reopening cost even when the forced suffix remained usable.

## 2. Suppression can be sparse and multi-barrier

Controlled trajectory analyses found a strong primary entry barrier plus non-contiguous secondary valleys. This motivated receding-horizon detection and adaptive repair rather than a one-shot intervention.

## 3. Known-route repair can be much more supervision-efficient than full-sequence SFT

Synthetic controls showed that repairing only the closed entry boundary can require far fewer supervised tokens than retraining the already-functional suffix. In a controlled three-barrier system, adaptive boundary repair also beat full-sequence SFT on the experiment's compute proxy.

These results are mechanism controls, not a claim that all real-model repair is cheaper than SFT.

## 4. Blind candidate discovery is possible in controlled systems

The strongest controlled candidate filters were based on counterfactual trajectory behavior after crossing a subdominant candidate boundary. Extreme low-logit search was not useful.

## 5. Distribution reconvergence generalized better than exact token equality

In controlled synthetic tests, next-token JS reconvergence selected the true candidate more reliably than exact surface-token reconvergence or downstream likelihood. Delayed-surface tests showed that the informative quantity can be **movement toward** a manifold rather than immediate similarity.

## 6. TinyStories fresh Transformer PoC

For fresh TinyStories-8M A-only SFT runs:

- `Lily -> Lucy`: closure 86.5x; known B late-JS rank 11/64.
- `woods -> forest`: closure 2.0x; rank 14/64.
- `town -> village`: closure 8.1x; rank 2/64.
- `boy -> girl`: closure 83.6x; rank 3/64.

The top-3 blind candidate set reopened the true B route in `village` and `girl`, but a naive fixed mass target overshot the base oracle. This established causal reopening while also motivating conservative stopping.

## 7. Base-free intervention strength can be bounded

Using only KL drift from the post-trained entry distribution as a stopping criterion produced a graded reopening curve. The same KL budget had different effects on different routes, so KL is an intervention budget rather than a calibrated recovery estimator.

## 8. Suppression depth predicts repair cost

Across 12 TinyStories conditions, closure factor correlated strongly with the number of local repair steps required for 90% oracle recovery: Spearman rho about 0.849, p about 4.8e-4.

## 9. Reconvergence has a documented failure class

Some surviving candidate routes (`meet`, `big` in tested contexts) retained coherent forced continuations but did not rank well by return-to-default reconvergence. These are **divergent-but-capable** routes.

## 10. Intrinsic normality finds coherent alternatives, not historical truth

A two-mode intrinsic trajectory score (high-confidence/low-NLL or high-entropy/diverse) ranked held-out semantic alternatives well, including divergent routes. But it also ranked deeply eroded controls highly.

Therefore intrinsic normality is a useful candidate generator, not a historical-recoverability detector.

## 11. Historical recoverability may be underidentified from the final checkpoint

Cross-context hidden-direction consistency and cross-context repair transfer remained strong after deeper SFT. Tiny micro-repair was also easy for deeper eroded alternatives. These negative results suggest that semantic geometry and trainability can persist without uniquely encoding the unknown model's training history.

The public project therefore focuses on **operational route discovery and reopening**.
