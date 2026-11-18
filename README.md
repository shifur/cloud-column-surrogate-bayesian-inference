# Sparse Polynomial and Neural Network Surrogates for Bayesian Cloud Microphysics Inference

Development repository: [`shifur/cloud-column-surrogate-bayesian-inference`](https://github.com/shifur/cloud-column-surrogate-bayesian-inference).

This source package implements study-owned surrogate training and evaluation,
RQMC metrics, CS/SR-LASSO, formal MCMC analysis and diagnostics, matched timing,
and the manuscript redraws. The companion scientific data record carries
the registered SCCM-derived arrays and the archived analysis inputs.

- RELEASE VERSION: AUTHOR DECISION REQUIRED
- SOFTWARE LICENSE: AUTHOR DECISION REQUIRED
- SOFTWARE DOI: NOT YET ASSIGNED
- DATA DOI: NOT YET ASSIGNED
- DATA REPOSITORY DOI: NOT YET ASSIGNED

## Install and run

Create and activate the tested CPU environment.

```sh
conda env create -f common/environment/plot-environment.yml
conda activate cloud-column-plot
```

Run the scripts directly from the repository root. The source package layout
is described in `pyproject.toml`. The exact resolved CPU packages are recorded
in `common/environment/resolved-cpu-packages.csv`. The recorded production
environment is `common/environment/derecho.yml`.

Set `DATA_ROOT` to the unpacked companion record root, containing
`PUBLICATION_BINDING.json`, `configs/`, `data/`, `formal_mcmc/`, and
`figures/`. Each redraw command requires a fresh output directory.

| Figure | Analysis | Command |
|---|---|---|
| 1 | Forward surrogate accuracy | `python common/scripts/redraw_release.py 1 --archive "$DATA_ROOT" --output-dir "redraw/figure-1"` |
| 2 | CS coefficient structure | `python common/scripts/redraw_release.py 2 --archive "$DATA_ROOT" --output-dir "redraw/figure-2"` |
| 3 | One-dimensional sensitivity | `python common/scripts/redraw_release.py 3 --archive "$DATA_ROOT" --output-dir "redraw/figure-3"` |
| 4 | Measured training and fitting costs | `python common/scripts/redraw_release.py 4 --archive "$DATA_ROOT" --output-dir "redraw/figure-4"` |
| 5 | Matched surrogate timing and SCCM context | `python common/scripts/redraw_release.py 5 --archive "$DATA_ROOT" --output-dir "redraw/figure-5"` |
| 6 | Posterior energy distances | `python common/scripts/redraw_release.py 6 --archive "$DATA_ROOT" --output-dir "redraw/figure-6"` |
| 7 | Posterior marginals and smoothed contours | `python common/scripts/redraw_release.py 7 --archive "$DATA_ROOT" --output-dir "redraw/figure-7"` |
| A1 | Candidate posterior variability | `python common/scripts/redraw_release.py A1 --archive "$DATA_ROOT" --output-dir "redraw/figure-A1"` |
| B1 | Formal convergence diagnostics | `python common/scripts/redraw_release.py B1 --archive "$DATA_ROOT" --output-dir "redraw/figure-B1"` |

Figures 6 and 7 read the six finalized full-except-ACC chains. The data record
also contains the six radiation-only chains and the saved diagnostics for both
masks. The redraw uses the stored scientific results and runs no SCCM, MCMC,
or surrogate training. Reproducing the original simulations and fitting runs
is a separate workflow with its own computational and external dependencies.

SCCM is an external numerical-model dependency used to generate model
outputs for the study. The repository contains study-owned interfaces,
configuration, preprocessing and postprocessing, surrogate, MCMC, and
analysis software. SCCM source code and binaries are outside the repository.
For authorized external runs, set `SCCM_INSTALL_DIR` to the directory with
`cloud_column_model.x`, `namelist_3h_t30-180.f90`, and `trp.dat`.
Exact SCCM source, version, access, citation, and redistribution terms remain
subject to author and model-owner confirmation.

The historical internal path and method identifier `pdr` means manuscript CS.

## Repository organization

- `common/src/cloud_column_reproduction/` contains the study package.
- `common/scripts/` contains the analysis and figure entry points.
- `common/configs/` and `common/environment/` hold portable study settings
  and tested environment specifications.
- `common/tests/` and `docs/` hold test coverage and workflow documentation.

## Tests

```sh
python -m pytest -q common/tests
CLOUD_COLUMN_DATA_ARCHIVE="$DATA_ROOT" python -m pytest -q common/tests
```

With the companion archive supplied, the suite also checks cross-package
bindings. GPU and external SCCM checks skip when the relevant capability is
unavailable. See `PUBLICATION_BINDING.json` for canonical figure hashes and
the formal chain checksum-index digest.


## Citation

Cite the manuscript and the frozen software and data records once their
citations are assigned. `CITATION.cff.template` records the unresolved
software citation fields. Article DOI: NOT YET ASSIGNED.

## License

SOFTWARE LICENSE: AUTHOR DECISION REQUIRED.
See `LICENSE_DECISION.md` for the author-review choices.
