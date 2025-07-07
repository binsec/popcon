#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import resource
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from configparser import ConfigParser
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import prctl


script = Path(__file__)
script.resolve()
root = script.parent

def warn(*args):
    print("\033[93mWARNING:", *args, '\033[0m', file=sys.stderr)


class TestStatus(Enum):
    SUCCESS = "\033[92mSUCCESS\033[0m"
    TIMEOUT = "\033[93mTIMEOUT\033[0m"
    INVALID = "\033[94mINVALID\033[0m"


def suicide():
    os.setpgrp()
    prctl.set_pdeathsig(signal.SIGINT)


def run(smt2_file: Path, timeout: int, goal: str, exact: bool, relax : int, relax_bfs: bool) -> Dict[Any, Any]:
    with tempfile.TemporaryDirectory() as d:
        statsfile = Path(d) / "stats.json"
        start = time.monotonic()
        invalid = False
        timeouted = False
        args: List[str] = ([
        str(root/"../target/release/popcon"),
        "--timeout", str(timeout*1000),
        "--goal", goal,
        "--stats", str(statsfile),
        "--relax", str(relax),
        ]
        + (["--relax-bfs"] if relax_bfs else [])
        + (["--relax-exact"] if relax and exact else [])
        + [ str(smt2_file) ])
        print("running", " ".join(args))
        process = subprocess.Popen(args , stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=False, preexec_fn=suicide)
        try:
            returncode = process.wait(timeout=timeout + 5)
            stop = time.monotonic()
            if returncode == 42:
                timeouted = True
            elif returncode:
                warn(f"popcon exited with code {returncode}")
                invalid = True
        except subprocess.TimeoutExpired:
            stop = time.monotonic()
            timeouted = True
        finally:
            if process.returncode is None:
                warn(f"killing process {process.pid}...")
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(100)
                except subprocess.TimeoutExpired:
                    pass
            try:
                os.kill(-process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            with statsfile.open("r") as f:
                try:
                    stats = json.load(f)
                except json.JSONDecodeError as e:
                    f.seek(0)
                    content = f.read()
                    warn(f"{statsfile} for test {smt2_file} contains malformed json: {e}: {content}")
                    stats = {}
        except OSError:
            warn(f"popcon did not create stats file {statsfile}")
            raise

        stats["timeout"] = timeouted or stats.get("timeout", False)
        stats["invalid"] = invalid
        stats["time"] = stop - start
        stats["param_relax"] = relax
        stats["param_exact"] = exact
        stats["param_relax_bfs"] = relax_bfs
        stats["param_file"] = str(smt2_file.relative_to(root))
        stats["param_timeout"] = timeout
        stats["param_goal"] = goal

        return stats


if __name__ == "__main__":
    resource.setrlimit(resource.RLIMIT_AS, (7*1024*1024*1024, 7*1024*1024*1024))
    args = []
    timeout = 60*8
    ncpu = max(1, 3*(os.cpu_count() or 1)//4)
    for f in root.glob("**/*.smt2"):
        args.append({"smt2_file": f, "timeout": timeout, "goal": "modelcount", "exact": True, "relax": 0, "relax_bfs": False})
        args.append({"smt2_file": f, "timeout": timeout, "goal": "popcon", "exact": True, "relax": 0, "relax_bfs": False})
        for relax in (3, 6, 9, 12, 15, 18, 20, 23, 30, 50):
            for relax_bfs in (True, False):
                for exact in ((True, False) if not relax_bfs else (False,)):
                    args.append({"smt2_file": f, "timeout": timeout, "goal": "popcon", "exact": exact, "relax": relax, "relax_bfs": relax_bfs})
    print(f"Starting {len(args)} tests on {ncpu} cpus, max duration: {len(args)*timeout/(3600*ncpu)}h")
    subprocess.check_call(["cargo", "build", "--release"])
    with ThreadPoolExecutor(max_workers=ncpu) as t:
        result = list(t.map(lambda args: run(**args), args))
    json.dump(result, (root/"results.json").open("w"))

