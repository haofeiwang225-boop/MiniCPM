"""Written for review; not executed in this development stage."""
import copy
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from altlora_from_zero.protocol import (
    read_rank_inventory, file_hash, learning_rate, expected_side,
    validate_resume_identity, require_fresh_output,
)


class ProtocolTests(unittest.TestCase):
    def inventory(self):
        ranks_q = [1]*7 + [2]*21 + [3]*12 + [4]*2
        ranks_v = [1]*41 + [2]
        rows = []
        for kind, ranks, dout in [('q', ranks_q, 2048), ('v', ranks_v, 256)]:
            for i, r in enumerate(ranks):
                rows.append(dict(module=f'model.layers.{i}.self_attn.{kind}_proj',
                    rank=r, d_in=2048, d_out=dout, A_shape=[r,2048], B_shape=[dout,r],
                    scaling=4/r, cost_per_added_rank=2048+dout, old_parameters=r*(2048+dout)))
        return dict(common=dict(inventory=dict(modules=rows, old_total_parameters=480000)))

    def test_inventory_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'plan.json'
            path.write_text(json.dumps(self.inventory()))
            history=Path(directory)/'history.json'
            history.write_text(json.dumps({'dense_lora_param_budget':480256}))
            patched=patch('altlora_from_zero.protocol.HISTORY_SHA256',file_hash(history))
            patched.start();self.addCleanup(patched.stop)
            result = read_rank_inventory(path, file_hash(path), historical_config=history)
            self.assertEqual(result['actual_parameters'], 480000)
            self.assertEqual(len(result['pattern']), 84)
            with self.assertRaises(ValueError): read_rank_inventory(path, '0'*64, historical_config=history)
            for key, value in [('rank', 5), ('A_shape', [1, 256]), ('scaling', 99)]:
                bad = self.inventory(); bad['common']['inventory']['modules'][0][key] = value
                path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError): read_rank_inventory(path, file_hash(path), historical_config=history)

    def test_all_2000_schedule_boundaries(self):
        counter = 0; update_a = True; side = 'both'
        for step in range(1, 2001):
            counter += 1
            if counter % 10 == 0:
                side = 'A' if update_a else 'B'; update_a = not update_a
            self.assertEqual(expected_side('altlora_true', step), side)
            self.assertEqual(expected_side('altlora_false', step), 'both')
            self.assertEqual(expected_side('vanilla', step), 'both')

    def test_historical_lr_endpoints(self):
        self.assertEqual(learning_rate(0), 5e-5)
        self.assertEqual(learning_rate(2000), 0)
        self.assertGreater(learning_rate(1999), 0)
        with self.assertRaises(ValueError): learning_rate(2001)

    def test_cross_treatment_resume_rejected(self):
        identity = dict(arm='vanilla', pair_id='p', rank_map_sha256='r',
            initial_adapter_sha256='a', train_order_sha256='o', base_hashes={},
            code_hashes={}, protocol={'version': 1})
        validate_resume_identity(identity, copy.deepcopy(identity))
        for arm in ('altlora_false', 'altlora_true'):
            with self.assertRaises(ValueError):
                validate_resume_identity(identity, dict(identity, arm=arm))

    def test_existing_output_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError): require_fresh_output(directory)
