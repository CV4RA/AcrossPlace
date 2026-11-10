"""Append-only GSV extension helpers; the original place records are read-only."""
import copy
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from utils.io import digest


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


class HammingIndex:
    """Exact radius <=4 search using the pigeonhole principle on five blocks."""
    def __init__(self):
        self.buckets = [defaultdict(list) for _ in range(5)]
        self.values = set()

    @staticmethod
    def parts(value):
        return [(value >> shift) & ((1 << width) - 1)
                for shift, width in [(0, 13), (13, 13), (26, 13), (39, 13), (52, 12)]]

    def add(self, value):
        value = int(value)
        if value in self.values:
            return
        self.values.add(value)
        for bucket, part in zip(self.buckets, self.parts(value)):
            bucket[part].append(value)

    def near(self, value, radius=4):
        if not 0 <= radius <= 4:
            raise ValueError('Five-block exact index supports Hamming radius 0..4')
        value = int(value)
        return any((other ^ value).bit_count() <= radius
                   for bucket, part in zip(self.buckets, self.parts(value))
                   for other in bucket.get(part, ()))


def xyz(points):
    ll = np.radians(np.asarray(points, dtype=np.float64))
    return np.stack([np.cos(ll[:, 0]) * np.cos(ll[:, 1]),
                     np.cos(ll[:, 0]) * np.sin(ll[:, 1]), np.sin(ll[:, 0])], axis=1)


class FrozenPlaceGuard:
    def __init__(self, base_rows, base_hashes, buffer_m=25, duplicate_hamming=4):
        self.ids = {r['place_id'] for r in base_rows}
        if len(self.ids) != len(base_rows):
            raise ValueError('Duplicate original place IDs')
        self.panos = {im['panoid'] for r in base_rows for im in r['images']}
        heldout = [gps for r in base_rows if r['split'] in ['val', 'test'] for gps in r['gps']]
        self.tree = cKDTree(xyz(heldout)) if heldout else None
        self.radius = 2 * np.sin(buffer_m / 6371000 / 2)
        self.hashes = HammingIndex()
        for value in base_hashes:
            self.hashes.add(value)
        self.duplicate_hamming = duplicate_hamming

    def metadata_reason(self, pid, gps, panos):
        if pid in self.ids:
            return 'already_in_original_15000'
        if self.tree is not None and gps:
            distance, _ = self.tree.query(xyz(gps), k=1)
            if np.any(distance <= self.radius):
                return 'within_25m_of_frozen_val_or_test'
        if self.panos.intersection(panos):
            return 'panorama_already_in_original_15000'
        return None

    def image_reason(self, values):
        if any(self.hashes.near(v, self.duplicate_hamming) for v in values):
            return 'near_duplicate_of_original_image'
        return None


def extension_plans(rows, protocol):
    """Same source choice and global source/gallery audit as plan_protocol.

    Incoming rows were already screened against ALL original image views. Original
    plans are never replanned. Hashes are precomputed from original JPEG bytes.
    """
    if protocol.get('protocol', 'B') != 'B':
        raise ValueError('This extension is pinned to original Protocol B')
    rng = random.Random(protocol.get('seed', 42))
    plans, issues = [], []
    seen = set()
    for row in rows:
        pid = row['place_id']
        if pid in seen or row['split'] != 'train':
            raise ValueError('Extension requires unique train-only places')
        seen.add(pid)
        groups = {}
        for image in row['images']:
            groups.setdefault(image['panoid'], []).append(image)
        names = sorted(groups)
        rng.shuffle(names)
        if len(names) < 2:
            issues.append({'place_id': pid, 'reason': 'insufficient_independent_panoramas'})
            continue
        n = min(protocol.get('source_panorama_groups', 2), len(names) - 1)
        chosen = set(names[:max(1, n)])
        source = [im for im in row['images'] if im['panoid'] in chosen][:protocol.get('max_source_views', 2)]
        gallery = [im for im in row['images'] if im['panoid'] not in chosen]
        plans.append({'place': row, 'place_id': pid, 'split': 'train', 'protocol': 'B',
                      'source_images': source, 'gallery_images': gallery})
    hashes = HammingIndex()
    paths, panos = set(), set()
    for plan in plans:
        for image in plan['source_images']:
            hashes.add(image['dhash64'])
            paths.add(image['path'])
            panos.add(image['panoid'])
    output = []
    removed = 0
    for plan in plans:
        gallery = []
        for image in plan['gallery_images']:
            if (image['path'] in paths or image['panoid'] in panos or
                    hashes.near(image['dhash64'], protocol.get('duplicate_hamming', 4))):
                issues.append({'place_id': plan['place_id'], 'image_id': image['image_id'],
                               'reason': 'source_or_near_duplicate_gallery'})
                removed += 1
            else:
                gallery.append(image)
        if gallery:
            output.append(dict(plan, gallery_images=gallery))
        else:
            issues.append({'place_id': plan['place_id'], 'reason': 'no_valid_source_or_independent_gallery'})
    return output, {'input_places': len(rows), 'eligible_places': len(output),
                    'near_duplicate_gallery_removed': removed, 'issues': issues,
                    'protocol': 'B', 'seed': protocol.get('seed', 42)}


def assert_append_only(base_rows, new_rows):
    base = {r['place_id'] for r in base_rows}
    new = {r['place_id'] for r in new_rows}
    if len(new) != len(new_rows) or base & new:
        raise ValueError('New place IDs duplicate original or each other')
    if any(r['split'] != 'train' for r in new_rows):
        raise ValueError('Frozen validation/test cannot receive new places')


def frozen_inventory(project):
    project = Path(project)
    names = ['outputs/scale_15k', 'outputs/vllm_15k', 'outputs/retrieval_15k/data']
    files = {}
    for name in names:
        for path in sorted((project / name).rglob('*')):
            if path.is_file():
                files[path.relative_to(project).as_posix()] = {'bytes': path.stat().st_size, 'sha256': file_sha(path)}
    for name in ['configs/vllm_15k.yaml', 'configs/dataset_15k.yaml']:
        path = project / name
        files[name] = {'bytes': path.stat().st_size, 'sha256': file_sha(path)}
    return {'version': 'gsv-original-15k-readonly-v1', 'files': files, 'fingerprint': digest(files)}


def verify_inventory(project, inventory):
    project = Path(project)
    for name, meta in inventory['files'].items():
        path = project / name
        if not path.is_file() or path.stat().st_size != meta['bytes'] or file_sha(path) != meta['sha256']:
            raise ValueError('Original 15k artifact changed: ' + name)


def append_file(source, addition, target):
    """Preserve original byte prefix; never write to either input."""
    import shutil
    source, target = Path(source), Path(target)
    addition = Path(addition) if addition else None
    if target.resolve() in {source.resolve(), *( [addition.resolve()] if addition else [])}:
        raise ValueError('Append target aliases an input')
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + '.tmp')
    with temporary.open('wb') as out:
        with source.open('rb') as stream:
            shutil.copyfileobj(stream, out)
        if addition is not None:
            if source.stat().st_size:
                with source.open('rb') as stream:
                    stream.seek(-1, 2)
                    if stream.read() != b'\n':
                        out.write(b'\n')
            with addition.open('rb') as stream:
                shutil.copyfileobj(stream, out)
    temporary.replace(target)
