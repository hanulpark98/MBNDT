# Result artifacts

`tables/` contains compact aggregate data used for the paper rather than raw
datasets, checkpoints, or full Optuna histories. The main CSVs include more
datasets than the final paper; analyses must filter to the 21 IDs in
`configs/paper_datasets.txt`.

`figures/leaf_budget_tradeoff.*` is regenerated with:

```bash
python scripts/make_leaf_budget_plot.py
```

The script gives each paper dataset equal weight, matching Table VI's
aggregation. It does not pool outer splits across datasets.
