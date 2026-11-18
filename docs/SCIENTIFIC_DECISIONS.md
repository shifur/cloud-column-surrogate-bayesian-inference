# Scientific decisions and open manuscript choices

## Fixed

ACC is excluded from both likelihoods. The two cases answer different questions: `radiation_only` uses OLR and OSR; `full_except_acc` uses PCP, LWP, IWP, OLR, and OSR.

The primary manuscript posterior comparison uses `full_except_acc`. The `radiation_only` case is a separately identified secondary comparison.

## Figure 7

Generate Figure 7 for `full_except_acc` as the primary result and a separately labeled `radiation_only` result. The authors must decide whether the secondary case belongs in the main text or appendix and state the likelihood mask in each caption.

## Posterior variability across Figure 1 fits

The clean comparison is 20 DNN models and 20 CS models at (m=5000), each trained on its registered Figure 1 trial. Begin with `full_except_acc`. Whether to repeat all 40 model-specific MCMC calculations for `radiation_only` should be decided after inspecting the selected-model radiation-only result and the cost on Derecho.

## Still requiring author review

- final Figure 7 placement and caption;
- whether the 40-model posterior-variability study is a main figure or appendix;
- whether that study is extended to `radiation_only`;
- final SCCM/DNN/CS walltime and resource requests after Derecho timing calibration.

## Matched FP32 timing experiment

The separately configured matched FP32 experiment reuses frozen models and observations, with float32 surrogate inference, float64 likelihood and sampler arithmetic, and two concurrent independent ensembles per method. It uses a separate chain root and paired elapsed timing ledgers. The original production template remains available. See [MATCHED_MCMC_TIMING.md](MATCHED_MCMC_TIMING.md). The incoming 0.001 observation-standard-deviation CS prediction guard is retained as an execution check; posterior diagnostics remain required. These new chains are timing-only and must not replace accepted scientific posterior results. Only Figure 5 adopts the matched elapsed metric; the previous CS/DNN 1.64 timing ratio is obsolete.
