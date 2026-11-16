"""CRC-checked completed ZIP entries for timing while the full download proceeds."""
import argparse
import json
from pathlib import Path
import struct
import zlib


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workdir', required=True)
    a = p.parse_args()
    work = Path(a.workdir).resolve()
    destination = work / 'benchmark-data'
    destination.mkdir(exist_ok=True)
    report = dict(scope='Fixed timing subsets of official COCO train/val ZIP prefixes; never used as the formal training set',
                  formal_train_images=118287, formal_val_images=5000)
    for split, wanted in [('train', 1024), ('val', 512)]:
        extracted = set()
        archive = work / 'downloads' / (split + '2017.zip')
        with archive.open('rb') as source:
            while len(extracted) < wanted:
                header = source.read(30)
                if len(header) < 30:
                    break
                signature, version, flags, method, _, _, crc, compressed, size, nname, nextra = struct.unpack('<4s5H3I2H', header)
                if signature != b'PK\x03\x04' or flags & 8 or method not in [0, 8]:
                    raise ValueError('Unsupported/incomplete ZIP prefix')
                name = source.read(nname).decode('utf-8')
                source.read(nextra)
                payload = source.read(compressed)
                if len(payload) != compressed:
                    break
                if name.endswith('/'):
                    continue
                decoded = zlib.decompress(payload, -15) if method == 8 else payload
                if len(decoded) != size or zlib.crc32(decoded) & 0xffffffff != crc:
                    raise ValueError('ZIP member failed CRC: ' + name)
                target = (destination / name).resolve()
                if destination not in target.parents or not name.endswith('.jpg'):
                    raise ValueError('Unexpected ZIP member')
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(decoded)
                extracted.add(target.name)
        if len(extracted) < wanted:
            raise RuntimeError(f'Need more downloaded bytes for {split}: {len(extracted)}/{wanted}')
        full = json.loads((work / 'coco/annotations' / f'instances_{split}2017.json').read_text())
        full['images'] = [im for im in full['images'] if im['file_name'] in extracted]
        ids = {im['id'] for im in full['images']}
        full['annotations'] = [ann for ann in full['annotations'] if ann['image_id'] in ids]
        path = destination / f'{split}.json'
        path.write_text(json.dumps(full))
        report[split] = dict(images=len(ids), annotations=str(path), directory=str(destination / (split+'2017')))
    (work / 'benchmark-assets-ready.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
