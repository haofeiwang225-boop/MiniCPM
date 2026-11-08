"""Standard-library-only protocol contracts. Never reads adapter tensors.

These functions do not execute training, import torch, or create output paths.
"""
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

VERSION = 'altlora-from-zero-2000-v2'
PLAN_SHA256 = 'a7fe38aef2b656b989a312bf75075c74297076d77775b1362604d8abe98661f6'
HISTORY_SHA256 = '432d8b33b87bbf2c16905dc142e180b8f3177c034bcfdf6acbee9b1685d8be07'
TRAIN_SHA256 = 'ab70c3ac6632d111dbdbfab7558d8c4e68ab2d6a0a605583fb2a7382f42032cc'
STAGES = (0, 200, 400, 800, 1200, 1600, 2000)
CHECKPOINT_STEPS = tuple(range(0, 2001, 200))
ARMS = ('vanilla', 'altlora_false', 'altlora_true')


def canonical_hash(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_rank_inventory(plan_path, expected_sha256=PLAN_SHA256, *, historical_config=None):
    """Only JSON inventory is read; the inventory's adapter path is never opened."""
    raw = Path(plan_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('Historical rank inventory SHA256 mismatch')
    if historical_config is None:
        raise ValueError('Trusted historical run_config is required for budget verification')
    history_raw = Path(historical_config).read_bytes()
    if hashlib.sha256(history_raw).hexdigest() != HISTORY_SHA256:
        raise ValueError('Historical configuration SHA256 mismatch')
    history = json.loads(history_raw)
    if history.get('dense_lora_param_budget') != 480256:
        raise ValueError('Historical budget mismatch')
    inventory = json.loads(raw)['common']['inventory']
    rows = inventory['modules']
    expected = {f'model.layers.{i}.self_attn.{kind}_proj'
                for i in range(42) for kind in ('q', 'v')}
    if len(rows) != 84 or {r['module'] for r in rows} != expected:
        raise ValueError('Missing, extra, or duplicate target module')
    pattern = {}
    histograms = {'q': Counter(), 'v': Counter()}
    total = 0
    for row in rows:
        name = row['module']; kind = name.rsplit('.', 1)[-1][0]
        rank = row['rank']; din = 2048; dout = 2048 if kind == 'q' else 256
        if type(rank) is not int or not 1 <= rank <= 4:
            raise ValueError(f'Invalid rank: {name}')
        if (row['d_in'], row['d_out']) != (din, dout):
            raise ValueError(f'Unexpected MiniCPM dimensions: {name}')
        if row['A_shape'] != [rank, din] or row['B_shape'] != [dout, rank]:
            raise ValueError(f'Factor shape mismatch: {name}')
        cost = din + dout
        if row['cost_per_added_rank'] != cost or row['old_parameters'] != rank * cost:
            raise ValueError(f'Parameter cost mismatch: {name}')
        if row['scaling'] != 4.0 / rank:
            raise ValueError(f'Scaling mismatch: {name}')
        pattern[name] = dict(rank=rank, A_shape=[rank, din], B_shape=[dout, rank],
                             scaling=row['scaling'], parameters=rank * cost)
        histograms[kind][rank] += 1
        total += rank * cost
    if histograms != {'q': Counter({1: 7, 2: 21, 3: 12, 4: 2}),
                       'v': Counter({1: 41, 2: 1})}:
        raise ValueError('Historical rank histogram mismatch')
    if total != 480000 or inventory['old_total_parameters'] != total:
        raise ValueError('Actual parameter count mismatch')
    return dict(pattern=pattern, actual_parameters=total, budget_limit=history['dense_lora_param_budget'],
                source_sha256=expected_sha256, rank_map_sha256=canonical_hash(pattern),
                origin='Historical rank metadata only; no historical factors or optimizer loaded')


def training_order(train_path, seed=42):
    """One new deterministic permutation, NOT historical trajectory replay."""
    raw = Path(train_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != TRAIN_SHA256:
        raise ValueError('Training dataset SHA256 mismatch')
    samples = json.loads(raw)
    if not isinstance(samples, list) or not samples:
        raise ValueError('Training records must be a nonempty list')
    if seed != 42:
        raise ValueError('This protocol is restricted to seed42')
    indices = []
    rng = random.Random(seed)
    while len(indices) < 2000:
        epoch = list(range(len(samples)))
        rng.shuffle(epoch)
        indices.extend(epoch)
    indices = indices[:2000]
    manifest = dict(seed=seed, dataset_sha256=TRAIN_SHA256,
                    order_algorithm='Python Random(seed), shuffled epochs truncated to 2000; NOT PyTorch DataLoader replay',
                    dataset_records=len(samples),
                    historical_replay=False, optimizer_steps=2000,
                    samples=[dict(step=i+1, index=index,
                                  sample_sha256=canonical_hash(samples[index]))
                             for i, index in enumerate(indices)])
    return dict(manifest, manifest_sha256=canonical_hash(manifest))


def learning_rate(completed_updates):
    """LR for the NEXT update, matching historical LambdaLR(1-step/steps)."""
    if type(completed_updates) is not int or not 0 <= completed_updates <= 2000:
        raise ValueError('Completed update count must be 0..2000')
    return 5e-5 * max(0.0, 1.0 - completed_updates / 2000)


def expected_side(arm, next_update):
    """Audit expectation only; runtime must call existing OfficialSchedule."""
    if arm not in ARMS or type(next_update) is not int or not 1 <= next_update <= 2000:
        raise ValueError('Invalid arm/update')
    if arm != 'altlora_true' or next_update < 10:
        return 'both'
    return 'A' if (next_update // 10) % 2 else 'B'


def validate_resume_identity(saved, current):
    required = ('arm', 'pair_id', 'rank_map_sha256', 'initial_adapter_sha256',
                'train_order_sha256', 'base_hashes', 'code_hashes', 'protocol')
    if current.get('arm') not in ARMS:
        raise ValueError('Invalid treatment identity')
    for key in required:
        if key not in saved or key not in current or saved[key] != current[key]:
            raise ValueError(f'Resume identity mismatch: {key}')


def require_fresh_output(path):
    if Path(path).exists():
        raise FileExistsError(f'Refusing to overwrite: {path}')
