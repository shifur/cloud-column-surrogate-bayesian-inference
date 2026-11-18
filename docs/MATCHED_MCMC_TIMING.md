# Matched float32 MCMC timing handoff

This version runs DNN and CS with the same two concurrent, independently initialized 32-walker ensembles. Each method has one persistent GPU evaluator and the same two-request microbatch broker, with a 2,000 microsecond rendezvous window and bounded singleton fallback. DNN and CS use the GPU in separate intervals; their order alternates between rounds. DE and DE-snooker updates retain their configured 80:20 mixture. The common observation, likelihood masks, parameter box, initialization, seeds, 30,000 burn-in transitions, and 300,000 retained transitions are preserved.

Both surrogates evaluate in float32. CS uses the explicit `cupy-cached-sparse-float32-cuda` backend; the existing float64 backend remains available for Figure 1 and numerical reference checks. Frozen model files and fitted coefficients are not rewritten. The likelihood, box-logit transformation, sampler coordinates, and stored chains remain float64. DNN disables TF32. CS does not enable fast math. The compiled DNN is constructed and evaluated on its broker worker because CUDA graph state has thread ownership.

## Timing-only production integration

For the authorized timing-only rerun in this production repository, pass `--timing-only` to the driver. This keeps the sampling/elapsed policy above and skips posterior export and diagnostics. Use only `full_except_acc`. The original production template and manuscript settings remain unchanged; runtime upgrade uses the separate `mcmc.matched-fp32.template.toml`. Figure 5 uses `configs/manuscript.matched-fp32.toml`, with isolated figure/table destinations. Run `python scripts/finalize_manuscript.py --config configs/manuscript.matched-fp32.toml figure5` only. The previous 1.64 CS/DNN ratio is obsolete.

## Reuse an existing production study

Install this source version alongside the existing study artifacts, preserving their paths under `common/`. Keep the original materialized `configs/mcmc.toml`, parameter-box configuration, observation and manifest, frozen models, and fit/training artifacts. No retraining or new SCCM observation is needed. Run these commands from `common/` inside a GPU allocation, using the existing study environment:

```bash
python scripts/prepare_mcmc.py upgrade-runtime \
  --config configs/mcmc.toml \
  --output configs/mcmc.matched-fp32.toml

python scripts/run_mcmc_surrogates.py \
  --config configs/mcmc.matched-fp32.toml \
  --case full_except_acc \
  --target-burn-in 30000 --target-production 300000
```

The upgrade verifies registered artifacts, preserves their bindings, and selects the fresh root `outputs/mcmc/matched_fp32_v2/chains`. It refuses an existing destination config or chain root. It does not overwrite the old chains or timings. Use the same run command to resume this new run. For the secondary likelihood, change `--case` to `radiation_only`.

The existing PBS wrapper accepts the upgraded configuration:

```bash
qsub -A PROJECT -q QUEUE \
  -v CASE=full_except_acc,CONFIG=configs/mcmc.matched-fp32.toml \
  environment/pbs/mcmc_surrogate_case.pbs
```

Submit from `common/`. Use separate submissions for the two likelihood cases. Each requests one GPU. A single case's driver runs both surrogate methods. The default advance processes up to fifty 1,000-transition blocks per ensemble before switching methods.

For a short smoke test, use a separate output root and omit retained sampling:

```bash
python scripts/run_mcmc_surrogates.py \
  --config configs/mcmc.matched-fp32.toml \
  --case full_except_acc --output-directory outputs/mcmc/smoke_matched_fp32 \
  --target-burn-in 2000 --target-production 0 --maximum-block-rounds 1
```

A fresh study follows the original workflow with the updated template. Use the concurrent driver for surrogate timing; advancing isolated units with `mcmc.py` does not implement this comparison.

## Timing outputs

Under each case, `matched_timing/summary.json` contains the observed elapsed seconds for each method's complete two-ensemble workload and `milliseconds_per_paired_transition`. If both ensembles advance through N transitions and their combined workload takes T seconds, the plotted value is 1000 T/N. One paired transition advances both ensembles once, comprising 64 walker proposals. There is no extra factor of two in this normalization. For the full run N=330,000, including burn-in.

Each method's ledger (`matched_timing/dnn.json` or `pdr.json`) stores interval clocks, before/after transition counts, committed block hashes, config and source hashes, precision/backend policy, package versions, and GPU identity. The elapsed clock surrounds the complete concurrent advance, including per-session checks, sampling, progress logging, checkpoint writing, and any measured retries. It excludes initial model loading, GPU compilation and prewarming, idle periods between invocations, NetCDF export, and diagnostics. Separate `setup_*.json` records retain construction and prewarming costs, machine/software details, actual microbatch counters, and backend metadata.

The summary also includes the mean productive milliseconds per ensemble transition for comparison with the historical convention. The original per-ensemble productive sampler, likelihood-evaluation, checkpoint-I/O, and interruption clocks remain in each `checkpoint.json`. These clocks can overlap between concurrent ensembles and must not be summed or averaged to infer the actual two-ensemble elapsed time.

Completed elapsed intervals survive restart. A caught interruption records its elapsed cost, while only uncommitted chain work is discarded. An uncaught hard termination leaves an explicit unknown elapsed interval. Sampling can resume, but a complete elapsed-time result is withheld for that timing replicate. Use a fresh root for a complete timing comparison after such a loss. Changing source, precision, batch-window policy, model/config bindings, package versions, GPU model, or advance size requires a fresh timing root. A case-level lock prevents two drivers from modifying the same run concurrently.

## Figure 5 and numerical checks

The Figure 5 finalizer now uses the matched elapsed ledgers and produces one logarithmic transition-time panel for DNN and CS. It requires the full 330,000 transitions in each ensemble and complete elapsed coverage. SCCM is omitted because the existing independent CPU-unit clocks do not measure the same paired elapsed workload. The historical hours panel is removed. The two concurrent ensembles share one elapsed clock, so they do not supply two independent timing replicates for error bars.

For an upgraded study, point `mcmc_config` in `configs/manuscript.toml` to `configs/mcmc.matched-fp32.toml` and set `chain_root` to the upgraded root. Then run:

```bash
python scripts/finalize_manuscript.py figure5
```

CS construction checks float32 predictions against the frozen expansion evaluated on CPU in float64: the original construction probe plus 256 deterministic interior points and 64 corners. It records the maximum prediction discrepancy in observation-standard-deviation units and the maximum absolute log-likelihood discrepancy for the selected mask. It rejects a prediction discrepancy of 0.001 observation standard deviations or more. This is a numerical guard, not a posterior convergence test. Final posterior diagnostics must still be reviewed after the full run; float32 and altered batch shapes need not produce identical individual MCMC paths.

The local short checks validate execution and restart behavior. Timings for the collaborator's full run are generated on that machine and are not populated from the earlier local benchmark.
