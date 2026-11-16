"""Mixed-storage conversion preserves decoder values and catches FP16 overflow."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]


class Compact(unittest.TestCase):
    def test_storage_contract_and_anchor_sentinel(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source.pt'
            out = root / 'mixed.pt'
            state = {'backbone.weight': torch.tensor([.1234567], dtype=torch.float32),
                     'encoder.weight': torch.tensor([1.001], dtype=torch.float32),
                     'decoder.weight': torch.tensor([.1234567], dtype=torch.float32),
                     'decoder.anchors': torch.tensor([0., float('inf')], dtype=torch.float32),
                     'decoder.valid_mask': torch.tensor([True, False])}
            torch.save({'model': state}, source)
            subprocess.run([sys.executable, 'tools/compact_weights.py', '--checkpoint', str(source),
                            '--output', str(out)], cwd=ROOT, check=True, capture_output=True)
            got = torch.load(out, weights_only=False)['model']
            self.assertEqual(got['backbone.weight'].dtype, torch.float16)
            self.assertEqual(got['decoder.weight'].dtype, torch.float32)
            for key in ['decoder.weight', 'decoder.anchors', 'decoder.valid_mask']:
                self.assertTrue(torch.equal(got[key], state[key]))
            report = json.loads(out.with_suffix('.storage.json').read_text())
            self.assertFalse(Path(str(out) + '.gz').exists())
            self.assertEqual(report['checkpoint_bytes'], out.stat().st_size)
            self.assertTrue(report['decoder_byte_values_preserved'])
            self.assertFalse(report['accuracy_validated'])

    def test_overflow_refuses_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, out = Path(tmp) / 'source.pt', Path(tmp) / 'mixed.pt'
            torch.save({'model': {'backbone.weight': torch.tensor([70000.])}}, source)
            p = subprocess.run([sys.executable, 'tools/compact_weights.py', '--checkpoint', str(source),
                                '--output', str(out)], cwd=ROOT, capture_output=True)
            self.assertNotEqual(p.returncode, 0)
            self.assertFalse(out.exists())

    def test_v2_preserves_context_projections_and_small_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, out = Path(tmp) / 'source.pt', Path(tmp) / 'mixed.pt'
            state = {'backbone.context_scale': torch.tensor([1e-12]),
                     'backbone.context.weight': torch.tensor([1e-12]),
                     'backbone.projections.0.weight': torch.tensor([1e-12]),
                     'backbone.body.weight': torch.tensor([.1])}
            torch.save({'model': state}, source)
            subprocess.run([sys.executable, 'tools/compact_weights.py', '--checkpoint', str(source),
                            '--output', str(out)], cwd=ROOT, check=True, capture_output=True)
            got = torch.load(out, weights_only=False)['model']
            for key in state:
                if key != 'backbone.body.weight':
                    self.assertEqual(got[key].dtype, torch.float32)
                    self.assertTrue(torch.equal(got[key], state[key]))

    def test_v2_rejects_storage_underflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, out = Path(tmp) / 'source.pt', Path(tmp) / 'mixed.pt'
            torch.save({'model': {'backbone.context_scale': torch.tensor([.1]),
                                  'backbone.body.weight': torch.tensor([1e-12])}}, source)
            result = subprocess.run([sys.executable, 'tools/compact_weights.py', '--checkpoint', str(source),
                                     '--output', str(out)], cwd=ROOT, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b'underflow', result.stderr)
            self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
