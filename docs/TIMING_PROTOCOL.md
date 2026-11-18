# Timing protocol

Report wall time and hardware for seven nonoverlapping stages: SCCM data generation, CS measurement-matrix construction, CS fitting, DNN training, RQMC surrogate evaluation, MCMC sampling, and postprocessing.

The q=23 measurement matrices are random because their rows are evaluations at the random training points. They are reproducible conditional on the registered trial seeds. Cache construction is an offline CS cost; fitting on nested row sub-samples is the online cost. Report both separately and also report their sum.

For MCMC, follow [the matched float32 protocol](MATCHED_MCMC_TIMING.md): both surrogates advance two concurrent ensembles, using the same microbatch scheduler and float32 model arithmetic. The primary clock is observed elapsed time for the complete paired workload; one paired transition advances both ensembles once. Per-ensemble productive clocks are reported separately. DNN and CS must use the same 32-walker callback contract and the same chain lengths. Record one-time DNN compilation separately from steady sampling. CS exact-zero pruning preserves the frozen expansion before floating-point evaluation; record the selected support size and checksum. SCCM timing belongs to the CPU allocation used for SCCM production and should not be extrapolated from the local workstation.

Only checksum-committed atomic blocks count as productive resumed-run time. Setup, checkpoint I/O, caught partial-block loss, and hard-loss upper bounds remain separate fields.
