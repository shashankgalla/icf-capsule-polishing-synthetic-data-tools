"""Read-only, fail-closed audit of synthetic release manifests and frame content."""
import csv
import hashlib
import json
from pathlib import Path
from .dataset import inside, dataset_root


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def audit(metadata_csv, strict=True, hash_frames=True):
    metadata_csv = Path(metadata_csv).resolve()
    root = dataset_root(metadata_csv)
    rows, issues = {}, []
    for split in ['train', 'val', 'test']:
        path = metadata_csv.parent / (split + '.csv')
        if not path.is_file():
            raise FileNotFoundError(f'Required split missing: {path.name}')
        rows[split] = list(csv.DictReader(path.open(newline='')))
        if not rows[split]:
            issues.append({'severity': 'critical', 'type': 'empty_split', 'split': split})
    interval_overlaps, shared_sims, shared_seeds, duplicate_content = [], [], [], []
    frame_sets, sim_sets, seed_sets = {}, {}, {}
    intervals = {}
    hashes = {}
    for split, records in rows.items():
        frame_sets[split], sim_sets[split], seed_sets[split] = set(), set(), set()
        sample_ids = set()
        for row in records:
            if row.get('regime') != 'controlled_c_slip' or row.get('surface_type') != 'synthetic_c_slip':
                raise ValueError('Non-synthetic record rejected')
            sid = row['sim_id']
            sim_sets[split].add(sid)
            seed_sets[split].add(row['simulation_seed'])
            if row['sample_id'] in sample_ids:
                issues.append({'severity': 'critical', 'type': 'duplicate_sample_id', 'split': split})
            sample_ids.add(row['sample_id'])
            start, end = int(row['start_frame']), int(row['end_frame'])
            if start < 0 or end <= start:
                raise ValueError('Invalid half-open frame interval')
            if not 0 <= float(row['slip']) <= 1:
                raise ValueError('Invalid slip')
            directory = inside(root, row['source_dir'])
            intervals.setdefault(sid, []).append((split, start, end))
            for frame in range(start, end):
                key = (sid, frame)
                frame_sets[split].add(key)
                path = directory / f'frame_{frame:06d}.png'
                if not path.is_file():
                    issues.append({'severity': 'critical', 'type': 'missing_frame', 'sample': row['sample_id'], 'frame': frame})
                elif hash_frames and key not in hashes:
                    hashes[key] = sha256(path)
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        common = frame_sets[a] & frame_sets[b]
        if common:
            interval_overlaps.append({'splits': [a, b], 'count': len(common)})
        common_sims = sim_sets[a] & sim_sets[b]
        if common_sims:
            shared_sims.append({'splits': [a, b], 'sim_ids': sorted(common_sims)})
        common_seeds = seed_sets[a] & seed_sets[b]
        if common_seeds:
            shared_seeds.append({'splits': [a, b], 'seeds': sorted(common_seeds)})
        if hash_frames:
            ha = {hashes[k] for k in frame_sets[a] if k in hashes}
            hb = {hashes[k] for k in frame_sets[b] if k in hashes}
            if ha & hb:
                duplicate_content.append({'splits': [a, b], 'distinct_sha256': len(ha & hb)})
    for kind, findings in [('frame_overlap', interval_overlaps), ('duplicate_frame_content', duplicate_content)]:
        if findings:
            issues.append({'severity': 'critical', 'type': kind, 'details': findings})
    if strict:
        for kind, findings in [('shared_simulation', shared_sims), ('shared_simulation_seed', shared_seeds)]:
            if findings:
                issues.append({'severity': 'critical', 'type': kind, 'details': findings})
    report = {'protocol': metadata_csv.parent.name, 'strict': strict,
              'status': 'PASS' if not issues else 'FAIL',
              'counts': {s: len(v) for s, v in rows.items()},
              'frame_overlap': interval_overlaps, 'shared_simulations': shared_sims,
              'shared_seeds': shared_seeds, 'duplicate_frame_content': duplicate_content,
              'hashed_unique_frames': len(hashes), 'issues': issues,
              'limits': ['SHA-256 checks exact duplicates, not perceptual similarity.',
                         'Seed metadata is checked, but simulator provenance is not independently certified.',
                         'No conclusions about private experimental data are made.']}
    return report
