# Replication report template

Please include enough information for another person to reproduce your result.

## Environment
- Model / checkpoint + exact revision:
- Hardware:
- OS:
- Python / PyTorch versions:

## Post-training setup
- Training objective:
- Number of steps / epochs:
- Learning rate / optimizer:
- Data or semantic branch definition:

## Search setup
- Candidate pool size:
- Lookahead horizon:
- Route score / ranking rule:
- Repair rule and KL budget, if any:

## Result
- Was the suppressed route present in the candidate pool?
- Candidate rank:
- Was it selected for repair?
- Entry probability before / after repair:
- Collateral KL or other drift measure:
- Raw result file or code link:

## Interpretation
- Reproduced / contradicted / partially reproduced:
- Failure mode or notable difference:

Negative results are explicitly welcome.
