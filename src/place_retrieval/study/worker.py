"""Locked sequential worker with per-job logs, explicit failure and immutable plans."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys

from ..io import digest, load_config, read_json, run_lock, sha256, write_json


def now():
    return datetime.now(timezone.utc).isoformat()


def worker(plan_path, role):
    plan_path = Path(plan_path)
    root = plan_path.parent
    with run_lock(root / "workers" / role):
        plan = read_json(plan_path)
        jobs = [job for job in plan["jobs"] if job["role"] == role]
        state_path = root / "workers" / role / "status.json"
        status = {"role": role, "pid": os.getpid(), "plan_fingerprint": plan["fingerprint"],
                  "started_at": now(), "completed": [], "test_stage": "sealed"}
        source_root = Path(__file__).resolve().parents[3]
        release = read_json(source_root / "release_files.json")
        for name, expected in release.items():
            if sha256(source_root / name) != expected:
                raise ValueError(f"Study source changed after freeze: {name}")
        if shutil.disk_usage(root).free < 25 * 2**30:
            raise RuntimeError("Less than 25 GiB free before first study job")
        for job in jobs:
            cfg = load_config(job["config"])
            if digest(cfg) != job["config_digest"]:
                raise ValueError("Frozen job configuration changed")
            done = Path(job["output"]) / "job_complete.json"
            if done.exists():
                saved = read_json(done)
                if saved["config_digest"] != job["config_digest"] or saved["action"] != job["action"]:
                    raise ValueError("A completed job belongs to another configuration")
                status["completed"].append(job["id"])
                continue
            if shutil.disk_usage(root).free < 20 * 2**30:
                status.update(stage="blocked_disk", active=job["id"], updated_at=now())
                write_json(state_path, status)
                raise RuntimeError("Disk reserve reached; no output or unique checkpoint deleted")
            log_path = root / "logs" / f"{job['id']}.log"
            log_path.parent.mkdir(exist_ok=True)
            status.update(stage="running", active=job["id"], log=str(log_path), updated_at=now())
            write_json(state_path, status)
            with log_path.open("a", buffering=1) as stream:
                process = subprocess.Popen([sys.executable, "-u", "-m", "place_retrieval.study.job",
                                            "--config", job["config"], "--action", job["action"]],
                                           stdout=stream, stderr=subprocess.STDOUT)
                status.update(child_pid=process.pid)
                write_json(state_path, status)
                code = process.wait()
            if code or not done.exists():
                status.update(stage="failed", exit_code=code, updated_at=now())
                write_json(state_path, status)
                raise RuntimeError(f"Job {job['id']} failed; inspect {log_path}")
            status["completed"].append(job["id"])
        status.update(stage="validation_complete_test_sealed", active=None, updated_at=now())
        write_json(state_path, status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--role", required=True, choices=("comparison", "ablation", "auxiliary"))
    args = parser.parse_args()
    worker(args.plan, args.role)
