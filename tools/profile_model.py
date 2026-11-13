"""Profile fixed-640 inference arithmetic on CPU; no weights or dataset required.

Conv/linear/matmul/attention use 2 FLOPs per multiply-accumulate. Bilinear
sampling adds 7 FLOPs per sampled output element (4 products + 3 sums).
Normalization, activations, softmax, indexing/top-k, coordinate arithmetic,
preprocessing and memory traffic are excluded. This is an arithmetic estimate,
not a latency benchmark. The exact 640 input is never stride-extrapolated.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import platform
import time

from common import config, digest


def bilinear_flops(input_shape, grid_shape, interpolation_mode, padding_mode,
                   align_corners, out_shape=None, **kwargs):
    if interpolation_mode != 0 or padding_mode != 0:
        raise ValueError('Only bilinear/zero-padding sampling is covered')
    n, c, _, _ = input_shape
    _, h, w, _ = grid_shape
    return n * c * h * w * 7

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.utils.flop_counter import FlopCounterMode

    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    torch.backends.mha.set_fastpath_enabled(False)
    cfg = config(args.config)
    if cfg.yaml_cfg.get('DeepStageHybrid', {}).get('weights_path'):
        raise ValueError('Use a release config with empty weights_path; profiling needs no checkpoint')
    model = cfg.model.eval().cpu()
    full_params = sum(p.numel() for p in model.parameters())
    components = {name: sum(p.numel() for p in getattr(model, name).parameters())
                  for name in ['backbone', 'encoder', 'decoder']}
    backbone = model.backbone
    components['pretrained_cnn'] = sum(p.numel() for p in backbone.body.parameters())
    components['context'] = sum(p.numel() for p in backbone.context.parameters())
    components['pyramid_projections'] = sum(p.numel() for p in backbone.projections.parameters())
    components['detail'] = sum(p.numel() for p in backbone.detail.parameters()) + backbone.detail_scale.numel()
    state = model.state_dict()
    state_elements = Counter()
    state_bytes = Counter()
    for value in state.values():
        state_elements[str(value.dtype)] += value.numel()
        state_bytes[str(value.dtype)] += value.numel() * value.element_size()
    del state
    model.deploy()
    deploy_params = sum(p.numel() for p in model.parameters())


    # Force SDPA to its matmul decomposition so CPU flash kernels cannot go uncounted.
    mapping = {torch.ops.aten.grid_sampler_2d: bilinear_flops}
    image = torch.zeros(1, 3, 640, 640)
    start = time.time()
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        with FlopCounterMode(display=False, custom_mapping=mapping) as counter:
            predictions = model(image)
    if not all(torch.isfinite(predictions[key]).all() for key in ['pred_logits', 'pred_boxes']):
        raise RuntimeError('Nonfinite initialized-model outputs')
    counts = counter.get_flop_counts()
    operations = {str(k): int(v) for k, v in counts['Global'].items()}
    total = counter.get_total_flops()
    if not any('bmm' in k for k in operations):
        raise RuntimeError('Attention matmuls missing from counter')
    if not any('grid_sampler' in k for k in operations):
        raise RuntimeError('Deformable bilinear sampling missing from counter')
    report = {
        'config': args.config, 'config_sha256': digest(args.config),
        'input': [1, 3, 640, 640], 'classes': int(cfg.yaml_cfg['num_classes']),
        'torch': torch.__version__, 'python': platform.python_version(), 'device': 'cpu',
        'threads': args.threads, 'weights': 'seed42 initialization; architecture-only measurement',
        'precision': 'FP32 CPU; arithmetic counts independent of AMP storage policy',
        'full_parameters': full_params, 'deployed_parameters': deploy_params,
        'parameter_components': components, 'full_state_elements_by_dtype': dict(state_elements),
        'full_state_bytes_by_dtype': dict(state_bytes),
        'flops': total, 'gflops': total / 1e9, 'operations': operations,
        'module_flops': {name: sum(ops.values()) for name, ops in counts.items()},
        'convention': '2 FLOPs per MAC; bilinear 7 per output element; exact640 deployed eval graph',
        'excluded': ['normalization', 'activation', 'softmax', 'indexing and top-k',
                     'coordinate arithmetic', 'preprocessing', 'memory traffic', 'criterion', 'backward'],
        'profile_seconds_not_latency': time.time() - start,
        'accuracy_or_latency_claim': False,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ['config', 'full_parameters', 'deployed_parameters', 'gflops']}))


if __name__ == '__main__':
    main()
