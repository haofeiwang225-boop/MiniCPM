"""Persist training-only calibration indices once; reload and verify at every probe."""
import hashlib
import json
from pathlib import Path


def write_calibration_indices(path, dataset, batch_size, num_batches, source):
    if batch_size < 1 or num_batches < 1:
        raise ValueError('Calibration counts must be positive')
    groups = [list(range(i, min(i+batch_size, len(dataset))))
              for i in range(0, len(dataset), batch_size)][:num_batches]
    if len(groups) != num_batches:
        raise ValueError('Training dataset has too few distinct calibration batches; no cycling allowed')
    payload = dict(split='train', dataset_source=str(source), dataset_length=len(dataset),
                   selection='first_batches_no_shuffle', batch_size=batch_size,
                   num_calibration_batches=num_batches, batch_indices=groups)
    path = Path(path)
    with path.open('x') as stream:
        stream.write(json.dumps(payload, indent=2)+'\n')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_calibration_batches(path, expected_sha256, dataset, collator):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError('Calibration manifest changed during trajectory')
    manifest = json.loads(data)
    if manifest['split'] != 'train' or manifest['dataset_length'] != len(dataset):
        raise ValueError('Calibration must match the original training dataset')
    indices = [i for group in manifest['batch_indices'] for i in group]
    if len(set(indices)) != len(indices) or any(i < 0 or i >= len(dataset) for i in indices):
        raise ValueError('Invalid calibration indices')
    return [collator([dataset[i] for i in group]) for group in manifest['batch_indices']]
