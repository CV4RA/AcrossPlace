"""Single-instance append-only A800 extension using the unchanged 15k generator."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import copy
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import requests
import yaml
from data.incremental_gsv import file_sha, frozen_inventory, verify_inventory, assert_append_only, append_file
from data.formal_retrieval import compact
from data.crossmodal_builder import build_crossmodal_dataset
from text_generation.vllm_pipeline import ConcurrentPlacePipeline
from utils.io import read_jsonl, write_jsonl, digest, config


def merge_formal(base, delta_annotations, out):
    """Append train rows/queries only; copy every validation/test file byte-for-byte."""
    base, out = Path(base), Path(out)
    original = json.loads((base / 'manifest.json').read_text())
    for name, meta in original['files'].items():
        if file_sha(base / name) != meta['sha256']:
            raise ValueError('Frozen formal baseline changed: ' + name)
    original_rows = read_jsonl(base / 'valid.jsonl')
    additions = read_jsonl(delta_annotations)
    assert_append_only(original_rows, additions)
    strict, expanded, skipped = [], [], []
    for row in additions:
        retained = [t for t in row['texts'] if t.get('accepted', True)]
        valid = [t for t in retained if t.get('quality_status') == 'valid']
        if not valid or not row['target_image_paths']:
            skipped.append({'place_id': row['place_id'], 'split': 'train', 'reason': 'no_valid_texts_or_gallery'})
            continue
        strict.append(compact(row, valid))
        expanded.append(compact(row, retained))
    out.mkdir(parents=True, exist_ok=True)
    if out.resolve() == base.resolve():
        raise ValueError('Combined formal output aliases original')
    deltas = out / 'delta'
    write_jsonl(deltas / 'valid.jsonl', strict)
    write_jsonl(deltas / 'valid_plus_uncertain_same_places.jsonl', expanded)
    write_jsonl(deltas / 'train.jsonl', strict)
    write_jsonl(deltas / 'skipped_places.jsonl', skipped)
    texts, images = [], []
    for row in strict:
        common = {'place_id': row['place_id'], 'split': 'train', 'positive_place_ids': [row['place_id']], 'protocol': 'B'}
        texts.extend(dict(t, **common) for t in row['texts'])
        images.extend(dict(im, **common) for im in row['images'] if im['path'] in row['target_image_paths'])
    for name, rows in [('train_t2i_queries', texts), ('train_i2t_gallery', texts),
                       ('train_t2i_gallery', images), ('train_i2t_queries', images)]:
        write_jsonl(deltas / (name + '.jsonl'), rows)
    files = {}
    for name, meta in original['files'].items():
        addition = deltas / name
        append_file(base / name, addition if addition.exists() else None, out / name)
        files[name] = {'rows': meta['rows'] + (sum(1 for _ in addition.open(encoding='utf-8')) if addition.exists() else 0),
                       'sha256': file_sha(out / name)}
        if name.startswith(('val.', 'val_', 'test.', 'test_')) and files[name]['sha256'] != meta['sha256']:
            raise ValueError('Combined formal evaluation file changed')
    splits = copy.deepcopy(original['splits'])
    splits['train']['places'] += len(strict)
    splits['train']['texts'] += len(texts)
    splits['train']['gallery_images'] += len(images)
    manifest = {'version': 'formal-valid-append-v1',
                'fingerprint': digest(['formal-valid-append-v1', original['fingerprint'], file_sha(delta_annotations), files]),
                'base_fingerprint': original['fingerprint'], 'base_directory': str(base),
                'source_annotation_delta': str(delta_annotations), 'source_sha256': file_sha(delta_annotations),
                'quality_policy': original['quality_policy'], 'protocol': 'B', 'splits': splits, 'files': files,
                'validation_and_test_byte_identical': True, 'new_valid_training_places': len(strict),
                'near_duplicate_audit': 'Frozen extension plans screened globally and against all original views',
                'human_audit_completed': False}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def merge_annotations(project, delta, combined):
    base = project / 'outputs/vllm_15k'
    for path in sorted((base / 'annotations').glob('*.jsonl')):
        addition = delta / 'annotations' / path.name
        if not addition.exists():
            raise ValueError('Missing delta annotation: ' + path.name)
        append_file(path, addition, combined / 'annotations' / path.name)
    for split in ['train', 'val', 'test']:
        old = json.loads((base / 'splits' / (split + '_places.json')).read_text())
        new = json.loads((delta / 'splits' / (split + '_places.json')).read_text())
        if split != 'train' and new:
            raise ValueError('Extension changes evaluation split')
        if set(old) & set(new):
            raise ValueError('Merged split has duplicate places')
        target = combined / 'splits' / (split + '_places.json')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(old + new, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--extension-root', default='outputs/gsv_extension_all')
    ap.add_argument('--wait-transfer', action='store_true')
    a = ap.parse_args()
    project = Path(__file__).resolve().parents[1]
    os.chdir(project)
    root = Path(a.extension_root).resolve()
    allowed = root == Path('/root/gsv_extension_all') or root.is_relative_to(project / 'outputs/gsv_extension_all')
    if not allowed:
        raise ValueError('Unsafe extension output')
    logs = root / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (root / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def status(stage, **fields):
            value = {'stage': stage, 'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'pid': os.getpid(), **fields}
            temporary = root / 'status.tmp'
            temporary.write_text(json.dumps(value, indent=2))
            temporary.replace(root / 'status.json')
            print(json.dumps(value), flush=True)
        try:
            preparation = root / 'preparation'
            manifest = json.loads((preparation / 'manifest.json').read_text())
            for name, meta in manifest['files'].items():
                if file_sha(preparation / name) != meta['sha256']:
                    raise ValueError('Extension preparation changed: ' + name)
            baseline = config('configs/vllm_15k.yaml')
            for name, expected in manifest['generator_code_sha256'].items():
                if file_sha(project / name) != expected:
                    raise ValueError('Original generation code differs: ' + name)
            if baseline != manifest['baseline_config'] or file_sha('outputs/scale_15k/data/places.jsonl') != manifest['base_index_sha256']:
                raise ValueError('Original generation configuration or place index changed')
            snapshot_path = root / 'original_15k_snapshot.json'
            if snapshot_path.exists():
                snapshot = json.loads(snapshot_path.read_text())
                verify_inventory(project, snapshot)
            else:
                status('snapshot_original_artifacts')
                snapshot = frozen_inventory(project)
                snapshot_path.write_text(json.dumps(snapshot, indent=2))
            done = root / 'transfer_complete.json'
            status('waiting_for_incremental_images', expected_images=manifest['counts']['added_images'])
            while not done.exists():
                if not a.wait_transfer:
                    raise RuntimeError('Transfer is not complete')
                failed = root / 'transfer_failed.json'
                if failed.exists():
                    raise RuntimeError('Image upload failed; inspect transfer_failed.json')
                time.sleep(15)
            transfer = json.loads(done.read_text())
            if transfer['preparation_identity'] != manifest['identity']:
                raise ValueError('Transfer belongs to another extension')
            status('verify_all_incremental_image_bytes')
            image_rows = read_jsonl(preparation / 'image_manifest.jsonl')
            data_root = Path(manifest['remote_dataset_root']).resolve()
            def verify_image(row):
                path = (data_root / row['path']).resolve()
                if not path.is_relative_to(data_root) or path.is_symlink() or path.stat().st_size != row['bytes'] or file_sha(path) != row['sha256']:
                    raise ValueError('Uploaded image hash/path mismatch: ' + row['path'])
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(verify_image, image_rows))
            new_places = read_jsonl(preparation / 'places.jsonl')
            assert_append_only(read_jsonl('outputs/scale_15k/data/places.jsonl'), new_places)
            if len(image_rows) != manifest['counts']['added_images']:
                raise ValueError('Image manifest count mismatch')
            # Copy the exact frozen generation configuration, changing input/output only.
            c = copy.deepcopy(baseline)
            delta = root / 'delta'
            c['dataset'] = {'index': str(preparation / 'places.jsonl'), 'max_places': len(new_places),
                            'plan_file': str(preparation / 'plans.jsonl')}
            c['output']['directory'] = str(delta)
            (root / 'generation_config.yaml').write_text(yaml.safe_dump(c, sort_keys=False))
            session = requests.Session()
            session.trust_env = False
            def ready():
                try:
                    r = session.get(c['vllm']['base_url'].rstrip('/') + '/models', timeout=5)
                    return r.ok and c['vllm']['served_model'] in [v['id'] for v in r.json()['data']]
                except (requests.RequestException, ValueError, KeyError):
                    return False
            if not ready():
                if shutil.disk_usage(data_root).free < 2 * 1024 ** 3:
                    raise RuntimeError('Insufficient data disk headroom for inference output')
                if shutil.disk_usage(root).free < 8 * 1024 ** 3:
                    raise RuntimeError('Insufficient extension disk headroom for cache and merged output')
                status('starting_original_vllm_server')
                with (logs / 'vllm_server.log').open('a') as output:
                    server = subprocess.Popen(['bash', 'scripts/start_vllm_a800.sh'], stdout=output, stderr=subprocess.STDOUT,
                                              start_new_session=True)
                (root / 'server.pid').write_text(str(server.pid))
                deadline = time.monotonic() + 1800
                while not ready():
                    if server.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError('Original vLLM server failed to become ready')
                    time.sleep(5)
            status('generate_new_places_only', places=len(new_places), original_places_untouched=15000)
            pipeline = ConcurrentPlacePipeline(c)
            try:
                plans = pipeline.prepare()
                fused = pipeline.extract(plans)
                raw = pipeline.generate(fused)
                quality = pipeline.filter(raw)
                retrieval = build_crossmodal_dataset({'annotations': str(pipeline.ann / 'place_texts_filtered.jsonl'),
                                                     'output': str(delta), 'protocol': 'B', 'duplicate_hamming': 4,
                                                     'spatial_buffer_m': 25})
                summary = {'status': 'complete_with_failures' if pipeline.failures else 'complete',
                           'quality': quality, 'retrieval': retrieval, 'metrics': pipeline.stats,
                           'original_generation_reused': False}
                (pipeline.reports / 'completion.json').write_text(json.dumps(summary, indent=2))
            finally:
                pipeline.close()
            status('merge_without_overwriting_originals')
            combined = root / 'combined'
            merge_annotations(project, delta, combined)
            formal = merge_formal(project / 'outputs/retrieval_15k/data', delta / 'annotations/place_texts_filtered.jsonl', combined / 'formal')
            verify_inventory(project, snapshot)
            result = {'status': summary['status'], 'original_15000_artifacts_unchanged': True,
                      'preparation_identity': manifest['identity'], 'input_counts': manifest['counts'],
                      'new_quality': quality, 'new_retrieval': retrieval, 'combined_formal': formal,
                      'human_audit_completed': False, 'rule_valid_is_not_human_ground_truth': True}
            (root / 'completion.json').write_text(json.dumps(result, indent=2))
            status(summary['status'], added_valid_training_places=formal['new_valid_training_places'],
                   combined_training_places=formal['splits']['train']['places'], original_artifacts_unchanged=True)
        except Exception as error:
            status('failed', error=str(error))
            raise


if __name__ == '__main__':
    main()
