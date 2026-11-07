#!/usr/bin/env python3
"""Package only the declared synthetic derivative without modifying its source."""
import argparse
import csv
import hashlib
import json
import random
import shutil
from pathlib import Path


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True, help='Prepared synthetic q4 directory')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--split-seed', type=int, default=2026)
    args = p.parse_args()
    source, out = args.source.resolve(), args.output.resolve()
    if source == out or out.exists():
        raise FileExistsError('Output must be a new directory; original data are never overwritten')
    rows = list(csv.DictReader((source / 'metadata.csv').open(newline='')))
    if not rows or any(r.get('regime') != 'controlled_c_slip' for r in rows):
        raise ValueError('Input must contain only synthetic C-slip metadata')
    sim_ids = sorted({r['sim_id'] for r in rows})
    records, metadata = [], {}
    for sid in sim_ids:
        if Path(sid).name != sid:
            raise ValueError('Invalid simulation ID')
        seq = source / 'controlled_c_slip' / sid
        meta = json.loads((seq / f'{sid}_metadata.json').read_text())
        if meta['parameters'].get('regime') != 'controlled_c_slip':
            raise ValueError('Unexpected simulation metadata')
        cleaned = {key: meta[key] for key in ['run_id', 'description', 'parameters', 'geometry', 'derived_kinematics', 'dataset_stats']}
        prep = meta.get('preprocessing', {})
        cleaned['preprocessing'] = {k: prep[k] for k in ['crop_x0_px', 'crop_y0_px', 'crop_width_px', 'crop_height_px', 'q4_resize'] if k in prep}
        metadata[sid] = cleaned
    if len({m['parameters']['seed'] for m in metadata.values()}) != len(sim_ids):
        raise ValueError('Simulation seed reuse requires grouping by seed before release')
    # Explicit new protocol. All clips from one simulation stay in one split.
    # This changes the scientific question and does not reproduce historical results.
    order = sim_ids.copy()
    random.Random(args.split_seed).shuffle(order)
    ntrain, nval = int(len(order) * .7), max(1, int(len(order) * .1))
    assignment = {sid: ('train' if i < ntrain else 'val' if i < ntrain + nval else 'test') for i, sid in enumerate(order)}
    if len(order) < 3 or not 0 < ntrain < ntrain + nval < len(order):
        raise ValueError('At least three simulation groups are required')
    out.mkdir(parents=True)
    for sid in sim_ids:
        seq = source / 'controlled_c_slip' / sid
        dest = out / 'frames' / sid
        shutil.copytree(seq / 'frames', dest)
        (out / 'simulation_metadata').mkdir(exist_ok=True)
        (out / 'simulation_metadata' / f'{sid}.json').write_text(json.dumps(metadata[sid], indent=2) + '\n')
        # CSV coordinates were scaled in q4 preparation; NPZ was copied unscaled.
        # Exclude inconsistent NPZ rather than distribute competing coordinates.
        gt = seq / f'{sid}_gt.csv'
        if gt.exists():
            (out / 'ground_truth').mkdir(exist_ok=True)
            shutil.copy2(gt, out / 'ground_truth' / f'{sid}.csv')
    for i, original in enumerate(sorted(rows, key=lambda r: (r['sim_id'], int(r['start_frame'])))):
        sid = original['sim_id']
        row = {k: original[k] for k in ['start_frame', 'end_frame', 'rpm', 'raw_rpm', 'surface_type', 'num_shells', 'fps', 'sim_id', 'regime', 'slip_type', 'slip', 'sc_target', 'sc_mean', 'split']}
        row = {'clip_id': i, 'sample_id': hashlib.sha256(f"{sid}:{row['start_frame']}:{row['end_frame']}".encode()).hexdigest()[:24],
               'source_dir': f'frames/{sid}', 'simulation_seed': metadata[sid]['parameters']['seed'], **row}
        records.append(row)
    info = {'format_version': 1, 'synthetic_only': True, 'simulation_count': len(sim_ids),
            'frame_size': [480, 48], 'clip_length': 30, 'fps': 100,
            'normalization': {'legacy_supplied': {'column': 'rpm', 'min': 10, 'max': 40}, 'paper_declared': {'column': 'raw_rpm', 'min': 21, 'max': 108}},
            'split_seed': args.split_seed, 'release_license': 'UNDECIDED: owner approval required',
            'protocols': {'historical_temporal': 'Original frame intervals and split assignments preserved. Shared simulations and seeds across splits.', 'trajectory_disjoint': 'New split: whole simulation groups, 70/10/20 by group count. New results required.'},
            'generation_slips': sorted({float(r['slip']) for r in records}),
            'excluded': ['Private experimental data', 'Model and detector weights', 'Stale unscaled GT NPZ', 'Absolute source paths']}
    (out / 'dataset_info.json').write_text(json.dumps(info, indent=2) + '\n')
    for protocol in ['historical_temporal', 'trajectory_disjoint']:
        protocol_rows = [dict(r, split=assignment[r['sim_id']]) if protocol == 'trajectory_disjoint' else dict(r) for r in records]
        directory = out / 'splits' / protocol
        write_csv(directory / 'metadata.csv', protocol_rows)
        for split in ['train', 'val', 'test']:
            write_csv(directory / (split + '.csv'), [r for r in protocol_rows if r['split'] == split])
    digest = hashlib.sha256()
    with (out / 'SHA256SUMS').open('w') as listing:
        for path in sorted(out.rglob('*')):
            if path.is_file() and path.name != 'SHA256SUMS':
                h = hashlib.sha256(path.read_bytes()).hexdigest()
                listing.write(f'{h}  {path.relative_to(out).as_posix()}\n')
                digest.update(h.encode())
    print(json.dumps({'simulations': len(sim_ids), 'clips': len(records), 'content_digest': digest.hexdigest(), 'output': str(out)}, indent=2))


if __name__ == '__main__':
    main()
