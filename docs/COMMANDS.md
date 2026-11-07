# Validation commands

Run from the code candidate root after installation:

```bash
python -m pytest -q
python -m pip check
python -m polish.cli audit --metadata /path/to/dataset/splits/trajectory_disjoint/metadata.csv --output strict_audit.json
python scripts/gpu_validation.py --dataset /path/to/dataset --base-model /path/to/svd-xt --output outputs/integration
```

The audit evidence directory records actual integration commands and return codes in `commands.json`. Historical audit failure is expected when the strict independence criterion is enabled. CPU training command and bounded settings are documented in the README. Test selection, hyperparameters and checkpoints must be frozen before computing final test scores.
