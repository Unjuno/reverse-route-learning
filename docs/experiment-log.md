# Experiment Log

The internal experiment numbers (`TBxx`) are kept here so raw result files and research notes can be cross-referenced.

| Range | Question | Main result |
| --- | --- | --- |
| TB9–TB20 | Does semantic A-only SFT close competing routes while downstream capability remains? | Yes, in multiple TinyStories branches; closure can precede suffix erosion. |
| TB21–TB38 | Can closed routes be bypassed / locally repaired? | Known candidate forcing and targeted repair can reopen routes; multi-hole repair benefits from sequential/adaptive treatment. |
| TB39–TB66 | Can post-only features distinguish Hidden vs Forgotten / rank route candidates? | Partial success; fresh-branch generalization exposed limits of fixed detectors. |
| TB67–TB72 | What is the topology and cost of multiple barriers? | Sparse multi-valley structure; adaptive repair can outperform full-sequence SFT in controlled systems. |
| TB73–TB85 | Can blind candidate search be compressed and false positives reduced? | Candidate space can be sharply reduced in controlled systems; micro-normality improves recall but precision remains difficult. |
| TB86–TB96 | Is trajectory reconvergence a better causal fingerprint? | Distribution/hidden reconvergence is strong in controlled tests; delayed and adaptive-horizon variants handle surface-form changes. |
| TB97 | Does reconvergence already appear in existing TinyStories logs? | Yes; retrospective real-weight features add signal beyond entry-only features. |
| TB98–TB99 | Fresh TinyStories Transformer PoC | Post-only scan ranks suppressed routes; scanner-selected candidate repair reopens true routes, initially with overshoot. |
| TB100–TB106 | Strength, repair cost, KL stopping, fresh branches | Closure correlates with repair cost; base-free KL budget controls intervention strength; reconvergence has divergent-route failures. |
| TB107–TB111 | Can intrinsic normality recover divergent routes and distinguish recoverability? | Finds coherent alternatives, but fails as historical-recoverability discriminator. |
| TB112–TB113 | Can cross-context internal traces identify historical preservation? | No; direction consistency and repair transfer remain strong even after deeper erosion. |
| TB114 | What can the current public heuristic actually do under one fixed protocol? | Establishes the v0.1 capability boundary: clear, partial, false-negative, and deep-eroded cases under the same post-only selection and KL-bounded repair rule. |

Selected raw JSONs are included under `results/tinystories/`. The repository does not include every exploratory artifact from the research session. Numerical comparisons should use results produced under the same protocol; TB114 is the primary fixed-protocol v0.1 benchmark.
