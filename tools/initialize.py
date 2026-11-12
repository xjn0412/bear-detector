"""Build a detector checkpoint from a DINOv3 backbone or random smoke-test weights."""
import argparse
import json
from pathlib import Path
from common import config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--pretrained', action='store_true', help='Download the matching timm DINOv3 backbone')
    group.add_argument('--backbone-weights', help='Local timm-compatible backbone safetensors/pth')
    group.add_argument('--random-init', action='store_true', help='Smoke tests only; no useful detection accuracy')
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    import torch
    import timm
    torch.set_num_threads(4)
    torch.manual_seed(42)
    cfg = config(a.config)
    cfg.yaml_cfg['DeepStageHybrid']['weights_path'] = a.backbone_weights or ''
    model = cfg.model
    if a.pretrained:
        source = timm.create_model(model.backbone.variant + '.dinov3_lvd1689m', pretrained=True, num_classes=0)
        source.head = torch.nn.Identity()
        model.backbone.body.load_state_dict(source.state_dict(), strict=True)
        del source
    out.parent.mkdir(parents=True, exist_ok=True)
    metadata = {'seed': 42, 'variant': model.backbone.variant, 'num_classes': cfg.yaml_cfg['num_classes'],
                'initialization': 'DINOv3 pretrained backbone; fresh detector' if not a.random_init else 'random smoke-test initialization',
                'detector_trained': False}
    torch.save({'model': model.state_dict(), 'last_epoch': -1, 'metadata': metadata}, out)
    print(json.dumps(dict(metadata, checkpoint=str(out))))


if __name__ == '__main__':
    main()
