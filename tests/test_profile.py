"""Arithmetic checks for the FLOPs reporting convention."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from profile_model import bilinear_flops


class Profile(unittest.TestCase):
    def test_bilinear_counts_every_output_channel(self):
        self.assertEqual(bilinear_flops((2, 8, 20, 20), (2, 5, 3, 2), 0, 0, False),
                         2 * 8 * 5 * 3 * 7)

    def test_unsupported_sampler_is_not_silently_counted(self):
        with self.assertRaises(ValueError):
            bilinear_flops((1, 1, 2, 2), (1, 1, 1, 2), 1, 0, False)

    def test_matmul_and_attention_count(self):
        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from torch.utils.flop_counter import FlopCounterMode
        a, b = torch.ones(2, 3), torch.ones(3, 4)
        q = torch.zeros(1, 2, 5, 4)
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            with FlopCounterMode(display=False) as counter:
                torch.mm(a, b)
                torch.nn.functional.scaled_dot_product_attention(q, q, q)
        # 2x3 @ 3x4 plus two attention products over two 5-token heads.
        self.assertEqual(counter.get_total_flops(), 2 * 2 * 3 * 4 + 2 * (2 * 1 * 2 * 5 * 5 * 4))


if __name__ == '__main__':
    unittest.main()
