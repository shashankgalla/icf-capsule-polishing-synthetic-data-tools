"""Deterministic paired synthetic evaluation. PSNR/SSIM only; no substitute FID/FVD."""
import argparse
import json
from pathlib import Path
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def paired_metrics(reference, generated):
    if reference.shape != generated.shape or reference.dtype != np.uint8 or generated.dtype != np.uint8:
        raise ValueError('Expected matching uint8 arrays [T,H,W,3]; no silent resize or padding')
    if reference.ndim != 4 or reference.shape[-1] != 3:
        raise ValueError('Expected [T,H,W,3] RGB arrays')
    psnr = [float(peak_signal_noise_ratio(a, b, data_range=255)) for a, b in zip(reference, generated)]
    ssim = [float(structural_similarity(a, b, channel_axis=-1, data_range=255)) for a, b in zip(reference, generated)]
    return {'psnr_mean': float(np.mean(psnr)), 'ssim_mean': float(np.mean(ssim)), 'n_frames': len(reference)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--metadata', type=Path, required=True)
    p.add_argument('--split', choices=['val', 'test'], required=True)
    p.add_argument('--generated', type=Path, required=True, help='Directory with sample_id.npy arrays')
    p.add_argument('--method-profile', choices=['legacy_supplied', 'paper_declared'], required=True)
    p.add_argument('--frames', type=int, default=30)
    p.add_argument('--width', type=int, default=480)
    p.add_argument('--height', type=int, default=48)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Evaluation output already exists')
    from .dataset import PolishingVideoDataset
    norm = {'rpm_column': 'raw_rpm', 'rpm_min': 21., 'rpm_max': 108.} if args.method_profile == 'paper_declared' else {}
    dataset = PolishingVideoDataset(args.metadata, split=args.split, augment=False, num_frames=args.frames,
        output_width=args.width, output_height=args.height, **norm)
    metrics, missing = [], []
    for sample in dataset:
        sid = sample['sample_id']
        path = args.generated / f'{sid}.npy'
        if not path.is_file():
            missing.append(sid)
            continue
        reference = (sample['video'].permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8)
        generated = np.load(path, allow_pickle=False)
        metrics.append({'sample_id': sid, **paired_metrics(reference, generated)})
    report = {'split': args.split, 'method_profile': args.method_profile,
              'status': 'PASS' if not missing and metrics else 'FAIL',
              'coverage': len(metrics) / len(dataset), 'missing_samples': missing,
              'per_clip': metrics,
              'psnr_mean': float(np.mean([r['psnr_mean'] for r in metrics])) if metrics else None,
              'ssim_mean': float(np.mean([r['ssim_mean'] for r in metrics])) if metrics else None,
              'limits': 'No LPIPS, FID-I3D, FVD, or detector-based motion score is computed. This does not reproduce the paper evaluation suite.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ['per_clip', 'missing_samples']}))
    if report['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
