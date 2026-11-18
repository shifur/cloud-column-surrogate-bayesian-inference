# Derecho installation

`derecho.yml` is a clean project environment, not a snapshot of the local development environment. From `common/`, run `bash environment/bootstrap_derecho.sh`. The bootstrap installs the package from the repository root, builds SCCM from source, and runs CPU tests. In a GPU allocation, run `python environment/verify.py --require-gpu` before Figure 1 or surrogate MCMC.

The PBS files are templates. Pass the allocation and queue to `scripts/run_study.py submit`, and revise their 12-hour requests only after target-node timing measurements. SCCM uses CPU-only jobs. DNN and CS use GPU jobs but retain separate method timings.
