# ICF capsule polishing: synthetic data tooling

Tooling for the synthetic slip-sweep data used in *Physics-Conditioned Video Diffusion with Kinematic Priors for Fusion Capsule Polishing* (NeurIPS 2026), by Shashank Galla, Abhishek Hanchate, Monika Biener, Suhas Bhandarkar, and Satish Bukkapatnam.

It covers generating the synthetic videos, checking dataset integrity, training a small model on them, and running inference/evaluation. The accompanying dataset is on Hugging Face: [shashankgalla/synthetic-capsule-polishing](https://huggingface.co/datasets/shashankgalla/synthetic-capsule-polishing).

## Install

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements-cpu.lock.txt --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
pip install --no-build-isolation --no-deps -e .
pytest -q
```

For a GPU environment, install a CUDA build of `torch` yourself and skip `requirements-cpu.lock.txt`.

## Get the dataset

```bash
python -m polish.cli download \
  --dataset-id shashankgalla/synthetic-capsule-polishing \
  --revision 4249c2a4a127bca81f66e0dc0b9f8e61714398f0 \
  --output data

python -m polish.cli audit --metadata data/splits/trajectory_disjoint/metadata.csv --output audit.json
```

The dataset has two splits: `historical_temporal` (used in the paper's early ablations) and `trajectory_disjoint` (whole simulations held out, no leakage across train/val/test). Prefer `trajectory_disjoint` for new work; use `--allow_historical_protocol` if you need the other one.

## Train

You'll need your own copy of the Stable Video Diffusion base model (`stabilityai/stable-video-diffusion-img2vid-xt`), which isn't included here.

```bash
export BASE_MODEL=/path/to/stable-video-diffusion-img2vid-xt
python -m polish.train \
  --method_profile paper_declared \
  --metadata_csv data/splits/trajectory_disjoint/metadata.csv \
  --pretrained_model "$BASE_MODEL" \
  --output_dir outputs/run \
  --epochs 100 --batch_size 2 --lora_rank 32 --num_frames 30 \
  --output_width 480 --output_height 48
```

`paper_declared` matches the physics formulation described in the paper. `legacy_supplied` is the earlier implementation kept for comparison; it uses different normalization and doesn't reproduce the paper's reported numbers.

Resume a run with `--resume_from outputs/run/epoch_001`.

## Generate and evaluate

```bash
python -m polish.cli infer \
  --checkpoint outputs/run/final --pretrained-model "$BASE_MODEL" \
  --image data/frames/seq_000_rpm021_sc0p10/frame_000000.png \
  --rpm 21 --slip 0.1 --frames 30 --steps 25 --seed 42 \
  --output outputs/example.mp4

python -m polish.evaluate \
  --metadata data/splits/trajectory_disjoint/metadata.csv --split test \
  --generated outputs/test --method-profile paper_declared \
  --output outputs/test_metrics.json
```

`evaluate` currently reports paired PSNR/SSIM. It doesn't include the paper's full detector-based slip/speed rank pipeline, LPIPS, FID, or FVD.

## Rendering new simulations

`generation/render_blender.py` runs the procedural simulator that generated the dataset (requires Blender). `generation/preprocess.py` applies the fixed crop/resize used to export frames.

```bash
python scripts/gpu_validation.py --dataset data --base-model "$BASE_MODEL" --output outputs/integration
```

This runs a quick end-to-end sanity check (train a couple of epochs, resume, run inference) rather than a full benchmark. See `docs/` for more detail on the dataset splits and known caveats.

## Citation

```bibtex
@inproceedings{galla2026physcond,
  title={Physics-Conditioned Video Diffusion with Kinematic Priors for Fusion Capsule Polishing},
  author={Galla, Shashank and Hanchate, Abhishek and Biener, Monika and Bhandarkar, Suhas and Bukkapatnam, Satish},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```

## License

MIT. See `LICENSE`.
