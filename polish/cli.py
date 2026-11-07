"""Synthetic release commands. No upload or publication command is provided."""
import argparse
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(prog='python -m polish.cli')
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('audit')
    a.add_argument('--metadata', type=Path, required=True)
    a.add_argument('--historical', action='store_true')
    a.add_argument('--output', type=Path, required=True)
    d = sub.add_parser('download')
    d.add_argument('--dataset-id', required=True)
    d.add_argument('--revision', required=True, help='Pinned approved dataset commit, not main')
    d.add_argument('--output', type=Path, required=True)
    g = sub.add_parser('infer')
    g.add_argument('--checkpoint', type=Path, required=True)
    g.add_argument('--pretrained-model', required=True)
    g.add_argument('--image', type=Path, required=True)
    g.add_argument('--rpm', type=float, required=True)
    g.add_argument('--slip', type=float, required=True)
    g.add_argument('--seed', type=int, default=42)
    g.add_argument('--frames', type=int, default=30)
    g.add_argument('--steps', type=int, default=25)
    g.add_argument('--width', type=int, default=480)
    g.add_argument('--height', type=int, default=48)
    g.add_argument('--cfg', type=float, default=3.)
    g.add_argument('--device', default='cuda')
    g.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.command == 'audit':
        from .data_integrity import audit
        report = audit(args.metadata, strict=not args.historical)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'status': report['status'], 'counts': report['counts']}))
        if report['status'] != 'PASS':
            raise SystemExit(1)
    elif args.command == 'download':
        import re
        if not re.fullmatch(r'[0-9a-f]{40}', args.revision):
            raise ValueError('Use an approved full 40-character dataset commit')
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=args.dataset_id, repo_type='dataset', revision=args.revision, local_dir=args.output)
        from .data_integrity import sha256
        root = args.output.resolve()
        for line in (root / 'SHA256SUMS').read_text().splitlines():
            expected, relative = line.split('  ', 1)
            from .dataset import inside
            if sha256(inside(root, relative)) != expected:
                raise ValueError('Downloaded dataset checksum mismatch')
        print('Downloaded files match the pinned checksum manifest')
    else:
        import numpy as np
        import torch
        from .generate import load_model, generate_video_manual, save_video
        if args.output.exists() or args.output.with_suffix('.npy').exists():
            raise FileExistsError('Refusing to overwrite generated output')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        dtype = torch.float16 if args.device.startswith('cuda') else torch.float32
        torch.manual_seed(args.seed)
        pipeline, encoder = load_model(args.checkpoint, args.pretrained_model, device=args.device, dtype=dtype)
        with torch.inference_mode():
            frames = generate_video_manual(pipeline.unet, pipeline.vae, pipeline.image_encoder,
                pipeline.feature_extractor, encoder, pipeline.conditioning_mode, pipeline.rpm_film,
                str(args.image), args.rpm, num_frames=args.frames, num_inference_steps=args.steps,
                output_width=args.width, output_height=args.height, device=args.device, dtype=dtype,
                cfg_scale=args.cfg, slip_ratio=args.slip)
        np.save(args.output.with_suffix('.npy'), frames)
        save_video(frames, args.output, fps=100)
        args.output.with_suffix('.json').write_text(json.dumps({**vars(args), 'checkpoint': str(args.checkpoint), 'image': str(args.image), 'output': str(args.output), 'shape': list(frames.shape), 'status': 'execution check only; not a paper benchmark'}, indent=2))


if __name__ == '__main__':
    main()
