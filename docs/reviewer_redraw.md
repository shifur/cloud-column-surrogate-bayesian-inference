# Reviewer redraw

From the software root, activate the CPU environment defined in
`common/environment/plot-environment.yml`. Set `DATA_ROOT` to the unpacked
scientific-data record. Then run:

```sh
python common/scripts/redraw_release.py 6 --archive "$DATA_ROOT" --output-dir redraw/figure-6
```

Use selectors 1 through 7, A1, or B1. Each selector requires a fresh output
directory outside the protected data subdirectories. See the root README
for the complete figure list. Formal config TOMLs remain in the data record,
and the adapter reads formal chains directly without making duplicate pools.

The DNN and CS bars in Figure 5 use active elapsed time for a concurrent
ensemble pair. Historical SCCM timing provides single-ensemble productive
clock context. The clocks measure different workloads.

SCCM ACCESS / REDISTRIBUTION TERMS: AUTHOR DECISION REQUIRED
