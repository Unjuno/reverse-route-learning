# Reproducibility Notes

This page records the concrete assumptions behind the v0.1 TinyStories experiments.

## Checkpoint

The public scripts target:

- repository: `roneneldan/TinyStories-8M`
- pinned revision: `b5c14392fcdc61157a3cf4ab6944e9335e7ad6b3`
- file: `pytorch_model.bin`
- expected SHA256: `22c355bfabebc1f6c861b3f5d7a801e96c7f6da4af4bb0f7780096ab82ea6716`

`scripts/download_tinystories.py` pins this revision and verifies the SHA256 by default. The model weights are not redistributed in this repository.

## Runtime

The repository includes a small direct-PyTorch GPT-Neo runtime rather than depending on `transformers` for the core experiments. TinyStories-8M alternates global and local attention; its local window is 256 tokens. For total sequence lengths at or below 256, the causal mask used by the minimal runtime is equivalent for these experiments. The runtime now rejects longer sequences rather than silently producing a different computation.

The published branch definitions are pre-tokenized, so the core reproduction scripts do not require tokenizer files.

## Reference execution environment

The public runs were performed on CPU with four PyTorch intra-op threads. A representative environment for the published run was:

- Python 3.13.5
- PyTorch 2.10.0+cpu
- CPU: AMD EPYC 9V74
- `torch.set_num_threads(4)` in the scan / benchmark scripts

The package metadata intentionally allows a wider supported range (`Python >=3.10`, `torch >=2.2`). Exact wall-clock times are therefore not portable and are not treated as scientific outcomes.

## Protocol comparability

The repository contains both exploratory results and a later fixed-protocol benchmark. Do not compare closure factors or recovery percentages from different TB runs as if all hyperparameters were identical.

For the current v0.1 capability claim, use:

- code: `experiments/tinystories/capability_boundary.py`
- raw results: `results/tinystories/tb114_operational_boundary.json`
- summary: `docs/current-capability-boundary.md`

TB98/TB99 are retained as earlier evidence showing fresh Transformer candidate discovery and causal reopening, but they use different exploratory settings.

## Oracle use

The base checkpoint is used to create the controlled post-training condition and for evaluation quantities such as closure, forced-suffix support, and oracle recovery. It is not used to rank candidates or choose the repair target set in the post-only procedures.

## Randomness

The current experiments are mostly deterministic CPU computations over fixed prefixes and greedy continuations. No claim is made that all PyTorch versions, devices, or parallel backends will reproduce every floating-point value bit-for-bit. Reproductions should compare qualitative rankings and reported metrics within normal floating-point tolerance.
