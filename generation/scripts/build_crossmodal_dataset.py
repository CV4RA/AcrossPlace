import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,json
from utils.io import config
from data.crossmodal_builder import build_crossmodal_dataset

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='configs/retrieval_protocol.yaml'); a=ap.parse_args()
    print(json.dumps(build_crossmodal_dataset(config(a.config)),indent=2))
if __name__=='__main__': main()
