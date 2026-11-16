"""Download official COCO2017 and initialize v2 from DINOv3 backbones."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
import time
import urllib.request
import zipfile


def parallel_download(url, target, workers=8):
    """Verified HTTP ranges, resumable parts, atomic final-file replacement."""
    with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=30) as response:
        size = int(response.headers['Content-Length'])
    if target.exists() and target.stat().st_size == size:
        return
    chunk = 64 << 20
    parts = target.with_suffix(target.suffix + '.parts')
    parts.mkdir(exist_ok=True)
    # Preserve the prefix downloaded by an earlier single-connection attempt.
    if target.exists():
        with target.open('rb') as source:
            for i in range((target.stat().st_size + chunk - 1) // chunk):
                content = source.read(chunk)
                part = parts / str(i)
                if not part.exists():
                    part.write_bytes(content)

    def fetch(i):
        start, end = i * chunk, min(size, (i + 1) * chunk) - 1
        part = parts / str(i)
        for attempt in range(5):
            have = part.stat().st_size if part.exists() else 0
            if have == end - start + 1:
                return
            request = urllib.request.Request(url, headers={'Range': f'bytes={start + have}-{end}'})
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    expected = f'bytes {start + have}-{end}/{size}'
                    if response.status != 206 or response.headers.get('Content-Range') != expected:
                        raise RuntimeError('Server did not honor requested range')
                    with part.open('ab') as output:
                        shutil.copyfileobj(response, output, 1 << 20)
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(1)
        if part.stat().st_size != end - start + 1:
            raise RuntimeError('Incomplete download part')

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(fetch, range((size + chunk - 1) // chunk)))
    temporary = target.with_suffix(target.suffix + '.complete')
    with temporary.open('wb') as output:
        for i in range((size + chunk - 1) // chunk):
            with (parts / str(i)).open('rb') as source:
                shutil.copyfileobj(source, output, 1 << 20)
    assert temporary.stat().st_size == size
    temporary.replace(target)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workdir', required=True)
    a = p.parse_args()
    work = Path(a.workdir).resolve()
    data = work / 'coco'
    downloads = work / 'downloads'
    for directory in [work, data, downloads, work / 'weights', work / 'logs']:
        directory.mkdir(parents=True, exist_ok=True)
    os.environ['HF_HOME'] = str(work / 'hf-cache')
    os.environ['TORCH_HOME'] = str(work / 'torch-cache')
    root = Path(__file__).resolve().parents[1]

    def download(name):
        url = 'https://s3.amazonaws.com/images.cocodataset.org/' + ('annotations/' if name.startswith('annotations') else 'zips/') + name
        target = downloads / name
        marker = downloads / (name + '.extracted')
        if marker.exists():
            return
        parallel_download(url, target)
        with zipfile.ZipFile(target) as archive:
            for info in archive.infolist():
                destination = (data / info.filename).resolve()
                if data not in destination.parents:
                    raise ValueError('Unsafe archive member: ' + info.filename)
            archive.extractall(data)
        marker.write_text(url + '\n')
        print('COCO_READY', name, flush=True)

    def initialize():
        # Serialize large CPU model creation while image downloads proceed.
        for size in ['L', 'B', 'S', 'T']:
            output = work / 'weights' / ('v2-' + size.lower() + '-initial.pt')
            if output.exists():
                continue
            with (work / 'logs' / ('initialize-' + size + '.log')).open('a') as log:
                subprocess.run([sys.executable, str(root / 'tools/initialize.py'), '--config',
                                str(root / 'configs' / ('V2-' + size + '.yml')), '--pretrained',
                                '--output', str(output)], cwd=root, stdout=log, stderr=log, check=True)
            print('INITIAL_READY', size, flush=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(download, name) for name in
                   ['annotations_trainval2017.zip', 'val2017.zip', 'train2017.zip']]
        futures.append(pool.submit(initialize))
        for future in futures:
            future.result()
    report = {'source': 'official COCO2017; DINOv3 ConvNeXt initialization', 'detection_training': 'train2017 only'}
    for split, expected in [('train', 118287), ('val', 5000)]:
        annotations = data / 'annotations' / ('instances_' + split + '2017.json')
        metadata = json.loads(annotations.read_text())
        assert len(metadata['images']) == expected and len(metadata['categories']) == 80
        missing = [im['file_name'] for im in metadata['images'] if not (data / (split + '2017') / im['file_name']).is_file()]
        assert not missing, missing[:10]
        report[split] = {'images': expected, 'annotations': str(annotations), 'directory': str(data / (split + '2017'))}
    (work / 'assets-ready.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
