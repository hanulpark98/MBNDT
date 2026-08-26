# Experiment entry points

The scripts in this directory preserve the method-specific protocols used for
the paper. Install this repository first so `mbndt` is importable.

- `run_mbndt_hpo.py`: MBNDT nested evaluation and HPO.
- `run_cart_hpo.py`: CART baseline.
- `run_xgboost_hpo.py`: XGBoost reference.
- `run_gradtree_hpo.py`: GradTree baseline; also requires TensorFlow and the
  [official GradTree implementation](https://github.com/s-marton/GradTree) to
  provide `from GradTree import GradTree`.
- `run_split_hpo.py`: SPLIT baseline; requires the
  [official SPLIT-ICML implementation](https://github.com/VarunBabbar/SPLIT-ICML)
  to provide `from split import SPLIT`.

The GradTree and SPLIT third-party repositories are intentionally not vendored.
Their exact upstream revisions still need to be recorded before the public
release. Run `python experiments/<script>.py --help` for method-specific flags.
