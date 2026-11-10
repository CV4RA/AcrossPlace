"""Freeze the runnable retrieval source before launching a study worker."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
paths=sorted([*root.glob('src/**/*.py'),*root.glob('scripts/*.py'),*root.glob('configs/*.yaml'),root/'pyproject.toml'])
files={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
(root/'release_files.json').write_text(json.dumps(files,indent=2)+'\n',encoding='utf-8')
print(f'Frozen {len(files)} source/configuration files in release_files.json')
