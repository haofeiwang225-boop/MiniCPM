"""RSC uses disjoint indices of the active TRAIN dataset; no selection data path."""
import hashlib
import json
from pathlib import Path
import torch
from .data_provenance import file_sha256


def tensor_sample_sha256(sample):
    h = hashlib.sha256()
    for key, value in sorted(sample.items()):
        if not torch.is_tensor(value):
            raise ValueError('Expected tensor-only training sample')
        value = value.detach().cpu().contiguous()
        h.update(str((key, str(value.dtype), tuple(value.shape))).encode())
        h.update(value.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def training_source(source, synthetic):
    if synthetic:
        return 'synthetic_training', None
    path = Path(source).resolve()
    # Fail closed: support explicit train split files, never silently accept dev/test.
    if path.name not in ('train.json', 'train.jsonl') or any(
            part.lower() in ('benchmarks', 'dev', 'test', 'validation', 'math-500') for part in path.parts):
        raise ValueError('RSC accepts only the active train.json/train.jsonl, never dev/test/benchmark data')
    return str(path), file_sha256(path)


def write_rsc_manifest(path, dataset, args):
    source, digest = training_source(args.train_data_path, args.tiny_model)
    calibration_bytes = Path(args.calibration_indices_path).read_bytes()
    if hashlib.sha256(calibration_bytes).hexdigest() != args.calibration_indices_sha256:
        raise ValueError('Original calibration manifest changed')
    original = json.loads(calibration_bytes)
    if original['split'] != 'train' or original['dataset_length'] != len(dataset) or original['batch_size'] != 1:
        raise ValueError('RSC requires the existing per-sample training calibration')
    if str(original['dataset_source']) != source:
        raise ValueError('Proposal source is not the active training data')
    proposal = [i for group in original['batch_indices'] for i in group]
    selection = args.rsc_selection_indices
    if selection is None:
        selection = [i for i in range(len(dataset)) if i not in set(proposal)][:args.rsc_selection_samples]
        if len(selection) != args.rsc_selection_samples:
            raise ValueError('Insufficient distinct training samples for Selection')
    for indices in (proposal, selection):
        if not indices or len(set(indices)) != len(indices) or any(type(i) is not int or i < 0 or i >= len(dataset) for i in indices):
            raise ValueError('Invalid/duplicate RSC sample indices')
    if set(proposal) & set(selection):
        raise ValueError('Proposal and Selection indices overlap')
    hashes = {str(i): tensor_sample_sha256(dataset[i]) for i in proposal+selection}
    if {hashes[str(i)] for i in proposal} & {hashes[str(i)] for i in selection}:
        raise ValueError('Proposal and Selection contain duplicate encoded samples')
    manifest = dict(split='train', source=source, source_sha256=digest, dataset_length=len(dataset),
                    synthetic=args.tiny_model, proposal_indices=proposal, selection_indices=selection,
                    sample_sha256=hashes, proposal_manifest_sha256=args.calibration_indices_sha256,
                    loss_definition='arithmetic_mean_of_individual_sample_model_losses_batch_size_1')
    with Path(path).open('x') as stream:
        json.dump(manifest, stream, indent=2)
    return file_sha256(path)


def load_rsc_samples(dataset, collator, args):
    path = Path(args.rsc_manifest_path)
    if file_sha256(path) != args.rsc_manifest_sha256:
        raise ValueError('RSC manifest changed')
    manifest = json.loads(path.read_text())
    source, digest = training_source(args.train_data_path, args.tiny_model)
    if (manifest['split'] != 'train' or manifest['source'] != source or manifest['source_sha256'] != digest
            or manifest['dataset_length'] != len(dataset) or manifest['synthetic'] != args.tiny_model):
        raise ValueError('RSC manifest does not match active TRAIN dataset')
    proposal, selection = manifest['proposal_indices'], manifest['selection_indices']
    if set(proposal) & set(selection):
        raise ValueError('Proposal and Selection overlap')
    for index in proposal+selection:
        if tensor_sample_sha256(dataset[index]) != manifest['sample_sha256'][str(index)]:
            raise ValueError(f'RSC training sample changed: {index}')
    return ([collator([dataset[i]]) for i in proposal],
            [collator([dataset[i]]) for i in selection], manifest)
