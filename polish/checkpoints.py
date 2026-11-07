"""Checkpoint metadata and optimizer/RNG state without copying frozen base weights."""
import json
import random
from pathlib import Path
import numpy as np
import torch


def save_training_state(directory, optimizer, epoch, global_step, best_loss, args, scaler=None):
    directory = Path(directory)
    np_state = np.random.get_state()
    state = {'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict() if scaler is not None else None, 'next_epoch': epoch + 1,
             'global_step': global_step, 'best_loss': best_loss,
             'epoch_complete': getattr(args, 'max_steps', None) is None,
             'python_rng': random.getstate(),
             'numpy_rng': [np_state[0], np_state[1].tolist(), np_state[2], np_state[3], np_state[4]],
             'torch_rng': torch.get_rng_state(),
             'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    temporary = directory / 'trainer_state.pt.tmp'
    torch.save(state, temporary)
    temporary.replace(directory / 'trainer_state.pt')
    (directory / 'run_config.json').write_text(json.dumps(vars(args), indent=2) + '\n')
    config = {k: getattr(args, k) for k in ['conditioning_mode', 'sc_encoding', 'num_frequencies_sc', 'physics_encoding', 'method_profile', 'lora_rank', 'lora_alpha', 'output_width', 'output_height', 'num_frames']}
    (directory / 'conditioning_config.json').write_text(json.dumps(config, indent=2) + '\n')


def restore_training_state(directory, optimizer, scaler=None):
    path = Path(directory) / 'trainer_state.pt'
    if not path.is_file():
        raise FileNotFoundError('Exact resume requires trainer_state.pt; weights-only loading is not resume')
    state = torch.load(path, map_location='cpu', weights_only=True)
    if not state.get('epoch_complete', False):
        raise ValueError('Mid-epoch integration checkpoints are not valid for exact resume')
    optimizer.load_state_dict(state['optimizer'])
    if state.get('scaler') is not None:
        if scaler is None:
            raise ValueError('Checkpoint requires the matching mixed-precision scaler')
        scaler.load_state_dict(state['scaler'])
    random.setstate(state['python_rng'])
    n = state['numpy_rng']
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state['torch_rng'])
    if state['cuda_rng']:
        if len(state['cuda_rng']) != torch.cuda.device_count():
            raise ValueError('Exact resume requires the same number of CUDA devices')
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state
