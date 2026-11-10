"""Build an immutable incremental manifest on the computer holding full images."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import collections
import csv
import hashlib
import io
import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import PurePosixPath

import numpy as np
from PIL import Image
from data.incremental_gsv import FrozenPlaceGuard, extension_plans, file_sha, assert_append_only
from utils.io import read_jsonl, write_jsonl, digest, config


def inspect_image(path):
    path = Path(path)
    content = path.read_bytes()
    with Image.open(io.BytesIO(content)) as image:
        image.verify()
    with Image.open(io.BytesIO(content)) as image:
        pixels = np.asarray(image.convert('L').resize((9, 8)))
    return {'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content),
            'dhash64': int.from_bytes(np.packbits(pixels[:, 1:] > pixels[:, :-1]).tobytes(), 'big')}


def safe_inspect_image(path):
    try:
        return inspect_image(path)
    except (OSError, ValueError) as error:
        return {'error': str(error)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-root', required=True)
    ap.add_argument('--base-index', required=True)
    ap.add_argument('--baseline-config', required=True)
    ap.add_argument('--remote-dataset-root', default='/root/autodl-tmp/datasets/GSVCities')
    ap.add_argument('--output', required=True)
    a = ap.parse_args()
    root, out = Path(a.dataset_root), Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    baseline = config(a.baseline_config)
    base_rows = read_jsonl(a.base_index)
    source_root = Path(__file__).resolve().parents[1]
    source_names = ['text_generation/vllm_pipeline.py', 'text_generation/batch_tasks.py',
                    'text_generation/prompts.py', 'text_generation/place_fusion.py',
                    'text_generation/quality_filter.py', 'utils/vllm_client.py',
                    'utils/generation_cache.py', 'data/retrieval_protocols.py']
    generator_code = {n: file_sha(source_root / n) for n in source_names}
    identity = digest(['gsv-extension-all-v1', file_sha(a.base_index), baseline, a.remote_dataset_root,
                       generator_code,
                       {p.name: file_sha(p) for p in sorted((root / 'Dataframes').glob('*.csv'))}])
    ready = out / 'manifest.json'
    if ready.exists():
        manifest = json.loads(ready.read_text())
        if manifest['identity'] != identity:
            raise ValueError('Existing extension identity changed; do not overwrite it')
        for name, meta in manifest['files'].items():
            if file_sha(out / name) != meta['sha256']:
                raise ValueError('Frozen extension file changed: ' + name)
        print(json.dumps({'status': 'already_prepared', **manifest['counts']}), flush=True)
        return
    cache = sqlite3.connect(out / 'local_image_audit.sqlite3')
    cache.execute('CREATE TABLE IF NOT EXISTS images (path TEXT PRIMARY KEY, size INTEGER, mtime INTEGER, value TEXT)')
    def cached(paths):
        results, todo = {}, []
        for path in paths:
            stat = path.stat()
            value = cache.execute('SELECT size,mtime,value FROM images WHERE path=?', (str(path),)).fetchone()
            if value and value[:2] == (stat.st_size, stat.st_mtime_ns):
                results[str(path)] = json.loads(value[2])
            else:
                todo.append((path, stat))
        pool = ThreadPoolExecutor(max_workers=8) if len(todo) > 1 else None
        try:
            values = pool.map(safe_inspect_image, [x[0] for x in todo]) if pool else map(safe_inspect_image, [x[0] for x in todo])
            for (path, stat), value in zip(todo, values):
                results[str(path)] = value
                cache.execute('INSERT OR REPLACE INTO images VALUES(?,?,?,?)',
                              (str(path), stat.st_size, stat.st_mtime_ns, json.dumps(value)))
        finally:
            if pool:
                pool.shutdown()
        cache.commit()
        return results
    base_paths = [root / 'Images' / im['path'].replace('\\', '/').split('/')[-2] / im['path'].replace('\\', '/').split('/')[-1]
                  for row in base_rows for im in row['images']]
    print(json.dumps({'stage': 'audit_original_images', 'images': len(base_paths)}), flush=True)
    base_values = cached(base_paths)
    if any('error' in value for value in base_values.values()):
        raise ValueError('An original 15k image is unreadable; preserve and inspect the baseline')
    guard = FrozenPlaceGuard(base_rows, [v['dhash64'] for v in base_values.values()])
    rows, rejected, errors = [], [], []
    official_eligible = 0
    started = time.time()
    for frame in sorted((root / 'Dataframes').glob('*.csv')):
        groups = {}
        with frame.open(encoding='utf-8-sig', newline='') as stream:
            for entry in csv.DictReader(stream):
                if entry['city_id'] != frame.stem:
                    raise ValueError('Official CSV city mismatch')
                groups.setdefault(entry['place_id'], []).append(entry)
        candidates = []
        for pid, entries in sorted(groups.items(), key=lambda item: int(item[0])):
            if len(entries) < 4:
                continue
            official_eligible += 1
            place_id = f'{frame.stem}:{int(pid):07d}'
            reason = guard.metadata_reason(place_id, [[float(e['lat']), float(e['lon'])] for e in entries],
                                           [e['panoid'] for e in entries])
            if reason:
                rejected.append({'place_id': place_id, 'reason': reason})
                continue
            candidates.append((pid, place_id, entries))
        warm = []
        for pid, place_id, entries in candidates:
            for entry in entries[:4]:
                stem = f"{frame.stem}_{int(pid):07d}_{int(entry['year']):04d}_{int(entry['month']):02d}_{int(entry['northdeg']):03d}"
                path = root / 'Images' / frame.stem / (stem + f"_{entry['lat']}_{entry['lon']}_{entry['panoid']}.jpg")
                if path.is_file():
                    warm.append(path)
        # One transaction per city avoids a disk sync and a thread pool per JPEG.
        cached(warm)
        # Image parsing uses exactly official row order and at most the first four valid images.
        for pid, place_id, entries in candidates:
            images = []
            for entry in entries:
                if len(images) == 4:
                    break
                stem = f"{frame.stem}_{int(pid):07d}_{int(entry['year']):04d}_{int(entry['month']):02d}_{int(entry['northdeg']):03d}"
                path = root / 'Images' / frame.stem / (stem + f"_{entry['lat']}_{entry['lon']}_{entry['panoid']}.jpg")
                try:
                    if not path.is_file():
                        matches = [p for p in path.parent.glob(stem + '*') if p.stem.endswith('_' + entry['panoid'])]
                        if len(matches) != 1:
                            raise FileNotFoundError(str(path))
                        path = matches[0]
                    meta = cached([path])[str(path)]
                    if 'error' in meta:
                        raise OSError(meta['error'])
                    rel = path.relative_to(root).as_posix()
                    remote = str(PurePosixPath(a.remote_dataset_root) / rel)
                    images.append({'image_id': 'I_' + digest(remote)[:24], 'path': remote,
                                   'gps': [float(entry['lat']), float(entry['lon'])], 'year': int(entry['year']),
                                   'month': int(entry['month']), 'northdeg': int(entry['northdeg']),
                                   'panoid': entry['panoid'], **meta})
                except (OSError, ValueError) as error:
                    errors.append({'place_id': place_id, 'path': str(path), 'reason': str(error)})
            reason = 'fewer_than_four_readable_images' if len(images) < 4 else guard.image_reason([i['dhash64'] for i in images])
            if reason:
                rejected.append({'place_id': place_id, 'reason': reason})
                continue
            rows.append({'place_id': place_id, 'dataset': 'GSV-Cities', 'images': images,
                         'image_paths': [i['path'] for i in images], 'gps': [i['gps'] for i in images],
                         'metadata': {'city': frame.stem, 'official_place_id': pid}, 'split': 'train'})
            if len(rows) % 1000 == 0:
                print(json.dumps({'stage': 'audit_new_places', 'new_candidates': len(rows),
                                  'elapsed_s': time.time() - started}), flush=True)
        print(json.dumps({'stage': 'scan_city', 'city': frame.stem, 'eligible_so_far': official_eligible,
                          'new_candidates': len(rows), 'elapsed_s': time.time() - started}), flush=True)
    assert_append_only(base_rows, rows)
    plans, protocol_report = extension_plans(rows, baseline['protocol'])
    eligible = {p['place_id'] for p in plans}
    rows = [row for row in rows if row['place_id'] in eligible]
    transfer = [{'path': str(PurePosixPath(im['path']).relative_to(a.remote_dataset_root)),
                 'bytes': im['bytes'], 'sha256': im['sha256'], 'place_id': row['place_id']}
                for row in rows for im in row['images']]
    if len({i['path'] for i in transfer}) != len(transfer):
        raise ValueError('New image path aliases detected')
    write_jsonl(out / 'places.jsonl', rows)
    write_jsonl(out / 'plans.jsonl', plans)
    write_jsonl(out / 'image_manifest.jsonl', transfer)
    write_jsonl(out / 'excluded_places.jsonl', rejected)
    write_jsonl(out / 'image_errors.jsonl', errors)
    (out / 'protocol_report.json').write_text(json.dumps(protocol_report, indent=2))
    counts = {'original_places': len(base_rows), 'official_places_with_at_least_four_rows': official_eligible,
              'remaining_before_audit': official_eligible - len(base_rows), 'added_places': len(rows),
              'added_images': len(transfer), 'added_image_bytes': sum(i['bytes'] for i in transfer),
              'combined_selected_places': len(base_rows) + len(rows), 'image_errors': len(errors),
              'exclusions': dict(collections.Counter(i['reason'] for i in rejected)),
              'protocol_rejected_places': protocol_report['input_places'] - protocol_report['eligible_places']}
    names = ['places.jsonl', 'plans.jsonl', 'image_manifest.jsonl', 'excluded_places.jsonl', 'image_errors.jsonl', 'protocol_report.json']
    manifest = {'identity': identity, 'counts': counts, 'baseline_config': baseline,
                'generator_code_sha256': generator_code,
                'base_index_sha256': file_sha(a.base_index), 'remote_dataset_root': a.remote_dataset_root,
                'training_only_extension': True, 'original_records_replanned': False,
                'images_per_place': 4, 'protocol': baseline['protocol'],
                'files': {n: {'sha256': file_sha(out / n), 'bytes': (out / n).stat().st_size} for n in names}}
    ready.write_text(json.dumps(manifest, indent=2))
    cache.close()
    print(json.dumps({'status': 'prepared', **counts}), flush=True)


if __name__ == '__main__':
    main()
