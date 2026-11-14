"""Store FP16 backbone/neck and original FP32 decoder; load into an FP32 model.

Rounding non-decoder tensors is lossy. Validate each converted checkpoint on
held-out data before distributing it. Does not change inference arithmetic.
"""
import argparse
import json
from pathlib import Path
from common import digest, weights


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    import torch
    source, ck = weights(a.checkpoint)
    out = Path(a.output)
    if out.exists():
        raise FileExistsError('Refusing to overwrite an existing checkpoint')
    if any(v.is_floating_point() and v.dtype != torch.float32 for v in source.values()):
        raise ValueError('Conversion requires the original FP32 EMA checkpoint')
    converted = {}
    elements = {'fp16': 0, 'fp32': 0, 'other': 0}
    for key, value in source.items():
        value = value.detach().cpu()
        if value.is_floating_point():
            target = value.float() if key.startswith('decoder.') else value.half()
            # Invalid anchors intentionally use +Inf and are masked by valid_mask.
            # Preserve that decoder buffer exactly; never permit new nonfinite values.
            if key == 'decoder.anchors':
                if target.dtype != torch.float32 or torch.isnan(target).any():
                    raise FloatingPointError('Invalid source anchor buffer')
            elif not torch.isfinite(target).all():
                raise FloatingPointError('Nonfinite source or storage conversion: ' + key)
            elements['fp32' if target.dtype == torch.float32 else 'fp16'] += target.numel()
        else:
            target = value.clone()
            elements['other'] += target.numel()
        if key.startswith('decoder.') and not torch.equal(value, target):
            raise RuntimeError('Decoder changed: ' + key)
        converted[key] = target
    metadata = dict(ck.get('metadata', {}))
    metadata['storage_conversion'] = {'source_sha256': digest(a.checkpoint),
        'format': 'FP16 backbone/context/neck; original FP32 decoder',
        'load_into': 'FP32 parameters; never model.half()', 'accuracy_validated': False,
        'elements': elements}
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': converted, 'last_epoch': ck.get('last_epoch', -1),
                'metadata': metadata}, out)
    # Verify decoder tensors after serialization.
    saved = torch.load(out, map_location='cpu', weights_only=False)['model']
    assert set(saved) == set(source)
    for key in source:
        assert saved[key].shape == source[key].shape
        if key.startswith('decoder.'):
            assert saved[key].dtype == source[key].dtype and torch.equal(saved[key], source[key])
    report = dict(metadata['storage_conversion'], checkpoint_sha256=digest(out),
        checkpoint_bytes=out.stat().st_size, decoder_byte_values_preserved=True)
    out.with_suffix('.storage.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
