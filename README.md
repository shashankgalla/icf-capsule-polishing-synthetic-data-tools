# Capsule polishing: synthetic release candidate

Local candidate for *Physics-Conditioned Video Diffusion with Kinematic Priors for Fusion Capsule Polishing*, by Shashank Galla, Abhishek Hanchate, Monika Biener, Suhas Bhandarkar and Satish Bukkapatnam. Publication details and permanent paper URL remain to be supplied.

This package contains synthetic data tooling only. The owner has confirmed code and data rights; code is released under MIT (see `LICENSE`). The accompanying dataset is public at [huggingface.co/datasets/shashankgalla/synthetic-capsule-polishing](https://huggingface.co/datasets/shashankgalla/synthetic-capsule-polishing). Full paper benchmark validation remains open. See [audit](docs/AUDIT.md) and [checklist](docs/RELEASE_CHECKLIST.md).

## Installation

Python 3.13 on Linux was tested. Python 3.11 and 3.12 are not verified. For the tested CPU environment:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.lock.txt --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
python -m pip install --no-build-isolation --no-deps -e .
python -m pytest -q
python -m pip check
```

`requirements.txt` pins direct dependencies. CUDA execution was tested with the existing PyTorch 2.10.0 CUDA 12.8 environment on one RTX 4090. A fresh CUDA installation and multiple GPUs are not verified. The trainer currently requires one process. Small execution checks used rank 2, four frames, 64 by 32 pixels and two training clips. Full paper settings require separate memory and benchmark validation.

## Dataset

The sibling `huggingface_dataset` package contains 20 simulations, 16,000 RGB frames and 540 clip records. Set `DATASET` to its absolute location. All media paths inside its manifests are relative. No private data or model weights are included.

```bash
export DATASET=/path/to/huggingface_dataset
python -m polish.cli audit --metadata "$DATASET/splits/trajectory_disjoint/metadata.csv" --output audit.json
```

A future published dataset can be downloaded and verified from this package:

```bash
python -m polish.cli download --dataset-id OWNER/DATASET --revision APPROVED_40_CHARACTER_COMMIT --output data
```

The ID and revision are required placeholders, not an existing public resource. See the dataset card for schema, provenance and split limitations.

`historical_temporal` preserves original split membership and intervals: 400 train, 40 validation, 100 test. Simulations and seeds are shared across these splits. `trajectory_disjoint` assigns whole simulations: 378 train, 54 validation, 108 test, with 14/2/4 simulation groups and split seed 2026. This is a new protocol and requires new results. It cannot validate the historical paper scores. There is only one simulation for each RPM/slip combination, so holding out simulations also holds out condition combinations.

## Training

Provide a separately licensed local SVD-xt base model. Frozen base weights and detector weights are not distributed here. The following is a short execution check, not the paper experiment:

```bash
export BASE_MODEL=/path/to/stable-video-diffusion-img2vid-xt
python -m polish.train --method_profile paper_declared --metadata_csv "$DATASET/splits/trajectory_disjoint/metadata.csv" --pretrained_model "$BASE_MODEL" --output_dir outputs/smoke --epochs 1 --batch_size 2 --grad_accum 1 --data_fraction .006 --lora_rank 2 --lora_alpha 2 --num_frames 4 --output_width 64 --output_height 32 --num_workers 0 --sample_every_epoch 999 --eval_every_epoch 999 --physics_warmup_epoch -1 --physics_ramp_epochs 1 --lambda_ratio .1 --deterministic
```

For CPU execution add `--mixed_precision no` and set `CUDA_VISIBLE_DEVICES=''`. The trainer checks data integrity before loading weights. Historical splits require the explicit `--allow_historical_protocol` override and retain their limitations.

Choose a method profile explicitly. `legacy_supplied` preserves the supplied physics equations and encoder behavior, with data-loader and reliability fixes. It is not byte-identical historical training. `paper_declared` uses physical RPM normalization, fixed Fourier frequencies, no cross-attention Q adapters, and the manuscript physics losses. These changes affect results. Neither profile has reproduced the reported paper benchmarks.

Checkpoints include adapters, encoder, run configuration, optimizer, AMP scaler and random states. Frozen base weights are omitted. Resume with `--resume_from outputs/run/epoch_001` and the same training configuration and total epoch count. Epoch boundaries support exact resume; interrupted mid-epoch runs do not. `best` means lowest training loss, not validation-selected performance. Test data must never be used to tune a run or select its checkpoint.

## Inference

```bash
python -m polish.cli infer --checkpoint outputs/smoke/final --pretrained-model "$BASE_MODEL" --image "$DATASET/frames/seq_000_rpm021_sc0p10/frame_000000.png" --rpm 21 --slip .1 --frames 4 --width 64 --height 32 --steps 2 --seed 42 --output outputs/example.mp4
```

Expected outputs are an MP4 preview, a lossless RGB uint8 NPY array and JSON settings. Add `--device cpu` for CPU inference. For a full configured run, use the matching frame count and resolution and the declared 25 DDIM steps and CFG 3. Supplied historical VAE pixel conventions are retained and require further scientific verification.

## Evaluation

Save generated arrays as `sample_id.npy`, one per manifest record, then run:

```bash
python -m polish.evaluate --metadata "$DATASET/splits/trajectory_disjoint/metadata.csv" --split test --generated outputs/test --method-profile paper_declared --output outputs/test_metrics.json
```

Evaluation requires complete split coverage and exact dimensions. It computes paired PSNR and SSIM only. It does not implement the paper's full detector, LPIPS, FID or FVD evaluation. The supplied source called an r3d_18 feature metric FVD and could silently substitute flow histograms. That implementation is excluded rather than relabeled as a verified paper metric.

## Generation and validation

`generation/render_blender.py` preserves the supplied procedural simulator, with an explicit `SIMULATION_SEED` environment setting. Blender is not installed in the audited environment, so rendering is not verified. Set a new `OUTPUT_ROOT`; never point it at existing research data. `generation/preprocess.py` provides an explicit fixed crop without fitting crop statistics on validation or test frames. The exported frames already contain the historical crop.

```bash
python scripts/gpu_validation.py --dataset "$DATASET" --base-model "$BASE_MODEL" --output outputs/integration
```

This bounded check trains two small epochs, resumes from the first epoch, compares adapter and encoder tensors, performs repeated fixed-seed inference and computes paired smoke metrics. It does not reproduce a benchmark. Tests and integrity reports are described in the audit.

## Citation and licensing

Cite the paper title and authors above until an approved BibTeX record and permanent URL are supplied. No license grant is made by this candidate. See `LICENSE_STATUS.md`. Before publication, fill in approved GitHub, Hugging Face, paper and checkpoint URLs consistently in this README and the dataset card.
