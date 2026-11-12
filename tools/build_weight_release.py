"""Prepare checksum-verified release assets on a CPU-only GitHub runner."""
import concurrent.futures
import gzip
import hashlib
import json
import shutil
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def download(item):
    u = urllib.parse.urlsplit(item['url'])
    if u.scheme != 'https' or u.hostname != 'mcp.sentinelx.app' or not u.path.startswith('/files/d/'):
        raise ValueError('Unexpected source host/path')
    path = ROOT / 'incoming' / item['name']
    path.parent.mkdir(exist_ok=True)
    req = urllib.request.Request(item['url'], method='POST')
    with urllib.request.urlopen(req, timeout=1800) as response, open(path, 'wb') as f:
        if response.status != 200:
            raise RuntimeError('Download failed')
        shutil.copyfileobj(response, f, 8 << 20)
    if path.stat().st_size != item['bytes'] or digest(path) != item['sha256']:
        raise ValueError('Source file size/checksum mismatch for ' + item['model'])
    print('Downloaded and SHA256 verified: ' + item['name'], flush=True)
    return path


def compress(source, target):
    with open(source, 'rb') as src, open(target, 'wb') as raw:
        with gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0, compresslevel=6) as dst:
            shutil.copyfileobj(src, dst, 8 << 20)
    h = hashlib.sha256()
    with gzip.open(target, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    if h.hexdigest() != digest(source):
        raise ValueError('Compression round-trip mismatch')


def main():
    import torch
    torch.set_num_threads(2)
    request = json.loads((ROOT / 'validation/release-request.json').read_text())
    out = ROOT / 'dist'
    out.mkdir(exist_ok=False)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        files = list(pool.map(download, request['assets']))
    records = []
    for item, source in zip(request['assets'], files):
        ck = torch.load(source, map_location='cpu', weights_only=True)
        state = ck['model']
        compact = {k: v.half() if v.dtype == torch.float32 and not k.startswith('decoder.') else v
                   for k, v in state.items()}
        for key, value in compact.items():
            if key.startswith('decoder.'):
                assert torch.equal(value, state[key]), key
            elif value.dtype == torch.float16:
                assert torch.isfinite(value).all(), key
        meta = dict(ck['metadata'], storage='FP16 backbone/context/neck; unchanged FP32 decoder',
                    full_coco_val2017_AP50_95=None, reference_full_COCO_AP=item['reference_AP'],
                    accuracy_status='experimental; rounded weights; full COCO and TensorRT parity not validated')
        compact_path = source.with_name('bear-' + item['model'].lower() + '-coco-compact-experimental.pt')
        torch.save(dict(model=compact, last_epoch=ck['last_epoch'], metadata=meta), compact_path)
        check = torch.load(compact_path, map_location='cpu', weights_only=True)
        assert set(check['model']) == set(state)
        assert all(torch.equal(v, check['model'][k]) for k, v in compact.items())
        del check, compact, state, ck
        for path, experimental in [(source, False), (compact_path, True)]:
            archive = out / (path.name + '.gz')
            compress(path, archive)
            # Test the documented CLI with the exact checkpoint stored in each archive.
            variant = item['model']
            subprocess.run(['python', str(ROOT / 'tools/predict.py'), '--config', str(ROOT / f'configs/{variant}.yml'),
                            '--checkpoint', str(path), '--classes', str(ROOT / 'configs/coco-classes.json'),
                            '--image', str(ROOT / 'assets/bear-input.jpg'), '--device', 'cpu',
                            '--output', str(ROOT / f'incoming/{variant}-{experimental}-prediction.json')],
                           cwd=ROOT, check=True)
            records.append(dict(filename=archive.name, bytes=archive.stat().st_size,
                                sha256=digest(archive), decompressed_sha256=digest(path),
                                model=variant, selected_epoch=item['best_epoch'], weights='EMA',
                                experimental=experimental, full_COCO_AP=None if experimental else item['reference_AP'],
                                strict_CLI_CPU_inference_passed=True))
        shutil.copy2(ROOT / f'configs/{item["model"]}.yml', out / f'bear-{item["model"].lower()}.yml')
    shutil.copy2(ROOT / 'configs/coco-classes.json', out / 'coco-classes.json')
    for name in ['LICENSE.md', 'THIRD_PARTY.md']:
        shutil.copy2(ROOT / name, out / name)
    (out / 'weights-manifest.json').write_text(json.dumps({'input_size': 640, 'assets': records}, indent=2))
    (out / 'SHA256SUMS').write_text(''.join(digest(p) + '  ' + p.name + '\n' for p in sorted(out.iterdir()) if p.is_file()))
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
