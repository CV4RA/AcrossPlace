import copy
import json
import random
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from data.incremental_gsv import (HammingIndex, FrozenPlaceGuard, extension_plans,
                                  assert_append_only, append_file, frozen_inventory, verify_inventory)
from data.retrieval_protocols import plan_protocol
from eval.protocols import dhash


def test_exact_radius_hash_search():
    rng = random.Random(8)
    values = [rng.getrandbits(64) for _ in range(500)]
    index = HammingIndex()
    for value in values:
        index.add(value)
    queries = [rng.getrandbits(64) for _ in range(100)]
    for value in values[:20]:
        for radius in range(5):
            mask = sum(1 << p for p in rng.sample(range(64), radius))
            queries.append(value ^ mask)
    for query in queries:
        for radius in range(5):
            assert index.near(query, radius) == any((query ^ v).bit_count() <= radius for v in values)
    with pytest.raises(ValueError):
        index.near(0, 5)


def test_frozen_guard_and_append_protection(tmp_path):
    base = [{'place_id': 'old', 'split': 'test', 'gps': [[0., 0.]], 'images': [{'panoid': 'p0'}]}]
    frozen = copy.deepcopy(base)
    guard = FrozenPlaceGuard(base, [0])
    assert guard.metadata_reason('old', [[10, 10]], ['new'])
    assert guard.metadata_reason('near', [[0, .0001]], ['new']) == 'within_25m_of_frozen_val_or_test'
    assert guard.metadata_reason('far', [[10, 10]], ['p0']) == 'panorama_already_in_original_15000'
    assert guard.metadata_reason('new', [[10, 10]], ['new']) is None
    assert guard.image_reason([15]) is not None
    assert guard.image_reason([(1 << 64) - 1]) is None
    assert base == frozen
    assert_append_only(base, [{'place_id': 'new', 'split': 'train'}])
    with pytest.raises(ValueError):
        assert_append_only(base, base)
    with pytest.raises(ValueError):
        assert_append_only(base, [{'place_id': 'new', 'split': 'test'}])
    first, second, target = [tmp_path / n for n in ['first', 'second', 'target']]
    first.write_bytes(b'original\n'); second.write_bytes(b'new\n')
    append_file(first, second, target)
    assert first.read_bytes() == b'original\n'
    assert target.read_bytes() == b'original\nnew\n'
    with pytest.raises(ValueError):
        append_file(first, second, first)


def test_source_gallery_matches_existing_algorithm(tmp_path):
    rows = []
    for place in range(8):
        images = []
        for view in range(4):
            path = tmp_path / f'{place}_{view}.jpg'
            image = np.random.default_rng(place * 100 + view).integers(0, 256, (80, 120, 3), dtype=np.uint8)
            Image.fromarray(image).save(path)
            images.append({'path': str(path), 'panoid': f'{place}_{view}', 'image_id': f'I{place}_{view}',
                           'dhash64': int.from_bytes(dhash(path), 'big')})
        rows.append({'place_id': str(place), 'split': 'train', 'images': images})
    c = {'protocol': 'B', 'seed': 42, 'source_panorama_groups': 2, 'max_source_views': 2, 'duplicate_hamming': 4}
    expected, original_report = plan_protocol(copy.deepcopy(rows), c)
    result, report = extension_plans(rows, c)
    assert result == expected
    assert report['eligible_places'] == original_report['eligible_places']
    assert all(len(p['source_images']) == len(p['gallery_images']) == 2 for p in result)
    # A second place's gallery identical to a source is removed globally.
    source = result[0]['source_images'][0]
    gallery = result[1]['gallery_images'][0]
    for im in rows[1]['images']:
        if im['path'] == gallery['path']:
            im['dhash64'] = source['dhash64']
    result, report = extension_plans(rows, c)
    assert report['near_duplicate_gallery_removed'] >= 1


def test_original_inventory_detects_mutation(tmp_path):
    for name in ['configs/vllm_15k.yaml', 'configs/dataset_15k.yaml', 'outputs/vllm_15k/annotations/original.jsonl']:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('original')
    snapshot = frozen_inventory(tmp_path)
    verify_inventory(tmp_path, snapshot)
    (tmp_path / 'outputs/vllm_15k/annotations/original.jsonl').write_text('modified')
    with pytest.raises(ValueError, match='Original 15k artifact changed'):
        verify_inventory(tmp_path, snapshot)


def test_combined_formal_keeps_all_evaluation_bytes(tmp_path):
    from scripts.run_gsv_extension import merge_formal
    from data.incremental_gsv import file_sha
    from utils.io import write_jsonl
    base = tmp_path / 'base'
    original = {'place_id': 'old', 'split': 'train', 'texts': []}
    base.mkdir()
    files = {}
    names = ['valid.jsonl', 'valid_plus_uncertain_same_places.jsonl', 'train.jsonl', 'skipped_places.jsonl',
             'train_t2i_queries.jsonl', 'train_i2t_gallery.jsonl', 'train_t2i_gallery.jsonl', 'train_i2t_queries.jsonl',
             'val.jsonl', 'val_t2i_queries.jsonl', 'test.jsonl', 'test_t2i_gallery.jsonl']
    for name in names:
        write_jsonl(base / name, [original] if name != 'skipped_places.jsonl' else [])
        files[name] = {'sha256': file_sha(base / name), 'rows': int(name != 'skipped_places.jsonl')}
    manifest = {'fingerprint': 'frozen', 'files': files, 'quality_policy': 'valid only',
                'splits': {s: {'places': 1, 'texts': 1, 'gallery_images': 1} for s in ['train', 'val', 'test']}}
    (base / 'manifest.json').write_text(json.dumps(manifest))
    addition = tmp_path / 'delta.jsonl'
    row = {'place_id': 'new', 'split': 'train', 'source_image_paths': ['s'], 'target_image_paths': ['g'],
           'images': [{'path': 's', 'image_id': 's'}, {'path': 'g', 'image_id': 'g'}],
           'texts': [{'text_id': 'new_T0', 'text': 'brick house', 'quality_status': 'valid', 'accepted': True}]}
    write_jsonl(addition, [row])
    before = {name: (base / name).read_bytes() for name in names}
    out = tmp_path / 'combined'
    result = merge_formal(base, addition, out)
    assert result['splits']['train']['places'] == 2
    for name in names:
        assert (base / name).read_bytes() == before[name]
        if name.startswith(('val.', 'val_', 'test.', 'test_')):
            assert (out / name).read_bytes() == before[name]
        else:
            assert (out / name).read_bytes().startswith(before[name])
