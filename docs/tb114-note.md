# TB114 note

The current fixed-protocol benchmark is deliberately small. Its purpose is to expose the present success/failure boundary of the public method, not to optimize a detector.

Observed under the same post-only selection and KL-bounded repair rule:

- clear success: `town -> village` (~73% oracle recovery at entry KL ~0.05);
- partial success: `little -> big` shallow (~30%), `boy -> girl` (~16%), `Lily -> Lucy` (~8%);
- current false negatives despite residual forced capability: `park -> slide`, `play -> meet`;
- deep-eroded controls show that semantic plausibility and local trainability do not establish historical recoverability.

This is the intended v0.1 boundary for external replication.
