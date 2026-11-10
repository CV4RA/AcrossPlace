"""Build immutable rule-valid formal retrieval metadata from generated records."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.formal_retrieval import prepare_formal
from utils.io import config

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    args=p.parse_args()
    print(json.dumps(prepare_formal(config(args.config)),indent=2))
