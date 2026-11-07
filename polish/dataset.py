"""Strict synthetic-only video loading with explicit splits and fixed normalization."""
import csv
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def dataset_root(csv_path):
    path = Path(csv_path).resolve()
    for parent in path.parents:
        if (parent / 'dataset_info.json').is_file():
            return parent
    return path.parent


def inside(root, relative):
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Dataset paths must be relative and cannot contain parent traversal')
    resolved = (Path(root) / path).resolve()
    if not resolved.is_relative_to(Path(root).resolve()):
        raise ValueError('Dataset path or symlink escapes the declared data root')
    return resolved


class PolishingVideoDataset(Dataset):
    RPM_MIN = 10.0
    RPM_MAX = 40.0

    def __init__(self, metadata_csv, crop_width=736, crop_height=184,
                 output_width=0, output_height=0, resize_factor=.25,
                 no_crop=True, clips_per_sample=1, augment=False,
                 temporal_subsample=1, num_frames=30, split=None,
                 crop_size=None, output_size=None, rpm_column='rpm',
                 rpm_min=10., rpm_max=40.):
        csv_path = Path(metadata_csv).resolve()
        if split is not None:
            if split not in {'train', 'val', 'test'}:
                raise ValueError('split must be train, val, or test')
            csv_path = csv_path.parent / f'{split}.csv'
        if not csv_path.is_file():
            raise FileNotFoundError(f'Required metadata or split is missing: {csv_path}')
        if min(clips_per_sample, temporal_subsample, num_frames) <= 0 or resize_factor <= 0:
            raise ValueError('Sample counts, frame counts, and resize factor must be positive')
        if not rpm_max > rpm_min:
            raise ValueError('rpm_max must exceed rpm_min')
        if augment and split != 'train':
            raise ValueError('Augmentation is permitted only on the explicit training split')
        self.root = dataset_root(csv_path)
        self.rpm_min, self.rpm_max = float(rpm_min), float(rpm_max)
        self.crop_width = crop_size or crop_width
        self.crop_height = crop_size or crop_height
        self.output_width = output_size or output_width
        self.output_height = output_size or output_height
        self.resize_factor, self.no_crop = resize_factor, no_crop
        self.augment, self.temporal_subsample, self.num_frames = augment, temporal_subsample, num_frames
        self.csv_path, self.split = csv_path, split
        self.clips = []
        with csv_path.open(newline='') as stream:
            for row in csv.DictReader(stream):
                if row.get('regime') != 'controlled_c_slip' or row.get('surface_type') != 'synthetic_c_slip':
                    raise ValueError('Only explicitly identified synthetic C-slip records are permitted')
                clip = dict(row)
                for field in ['start_frame', 'end_frame']:
                    clip[field] = int(row[field])
                for field in ['slip', 'fps']:
                    clip[field] = float(row[field])
                clip['rpm'] = float(row[rpm_column])
                if not all(math.isfinite(clip[k]) for k in ['rpm', 'slip', 'fps']):
                    raise ValueError('Non-finite physical label')
                if not 0 <= clip['slip'] <= 1 or clip['fps'] <= 0:
                    raise ValueError('Slip must be in [0,1] and fps must be positive')
                if not self.rpm_min <= clip['rpm'] <= self.rpm_max:
                    raise ValueError('RPM outside the declared fixed normalization range')
                if clip['start_frame'] < 0 or len(range(clip['start_frame'], clip['end_frame'], temporal_subsample)) < num_frames:
                    raise ValueError('Clip cannot supply the requested number of frames')
                if split is not None and row.get('split', split) != split:
                    raise ValueError('Split label disagrees with the selected split file')
                clip['source_dir'] = str(inside(self.root, row['source_dir']))
                self.clips.append(clip)
        if not self.clips:
            raise ValueError('Empty split')
        self.samples = [clip for clip in self.clips for _ in range(clips_per_sample)]
        if self.output_width <= 0 or self.output_height <= 0:
            first = self.clips[0]
            img = cv2.imread(str(Path(first['source_dir']) / f"frame_{first['start_frame']:06d}.png"))
            if img is None:
                raise FileNotFoundError('First synthetic frame is missing')
            self.output_width = self._align_to_vae_grid(img.shape[1] * resize_factor)
            self.output_height = self._align_to_vae_grid(img.shape[0] * resize_factor)
        if self.output_width % 16 or self.output_height % 16:
            raise ValueError('Output dimensions must be multiples of 16')

    @staticmethod
    def _align_to_vae_grid(x):
        return max(16, int(x) // 16 * 16)

    def __len__(self):
        return len(self.samples)

    def _normalize_rpm(self, rpm):
        return (rpm - self.rpm_min) / (self.rpm_max - self.rpm_min)

    def _load_frames(self, source_dir, start, end):
        frames = []
        for idx in list(range(start, end, self.temporal_subsample))[:self.num_frames]:
            path = Path(source_dir) / f'frame_{idx:06d}.png'
            image = cv2.imread(str(path))
            if image is None:
                raise FileNotFoundError(f'Missing or unreadable synthetic frame: {path}')
            frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return np.stack(frames)

    def __getitem__(self, idx):
        clip = self.samples[idx]
        frames = self._load_frames(clip['source_dir'], clip['start_frame'], clip['end_frame'])
        if not self.no_crop:
            h, w = frames.shape[1:3]
            ch, cw = min(h, self.crop_height), min(w, self.crop_width)
            x = random.randint(0, w - cw) if self.augment else (w - cw) // 2
            y = random.randint(0, h - ch) if self.augment else (h - ch) // 2
            frames = frames[:, y:y + ch, x:x + cw]
        frames = np.stack([cv2.resize(f, (self.output_width, self.output_height), interpolation=cv2.INTER_LINEAR) for f in frames])
        if self.augment:
            if random.random() > .5:
                frames = frames[:, :, ::-1].copy()
            if random.random() > .5:
                frames = np.clip(frames * random.uniform(.9, 1.1), 0, 255).astype(np.uint8)
        video = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2).float() / 255.
        return {'video': video, 'cond_image': video[0],
                'physics_params': torch.tensor([self._normalize_rpm(clip['rpm']), clip['slip']], dtype=torch.float32),
                'rpm': clip['rpm'], 'surface_type': clip['surface_type'], 'sample_id': clip.get('sample_id', str(idx))}
