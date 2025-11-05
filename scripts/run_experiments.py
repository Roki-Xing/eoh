#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch runner for EOH experiments driven by YAML configuration.

Example config (see config/experiments/sample.yaml):
  experiments:
    - name: baseline
      evolve:
        csv: data/SPY.csv
        symbol: SPY
        train_start: 2016-01-01
        train_end: 2020-12-31
        test_start: 2021-01-01
        test_end: 2023-12-31
        population: 64
        generations: 5
      post_export: true
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml

from eoh_core.utils import ensure_dir, log


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")


def _build_cmd(script: str, options: Dict[str, Any]) -> list[str]:
    cmd = [sys.executable, script]
    for key, value in options.items():
        if value is None:
            continue
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])
    return cmd


def run_experiment(exp: Dict[str, Any], base_outdir: Path, dry_run: bool = False) -> None:
    name = exp.get("name") or f"exp-{_timestamp()}"
    evolve_opts = exp.get("evolve", {})
    if not evolve_opts:
        raise ValueError(f"Experiment '{name}' missing 'evolve' section")

    outdir = ensure_dir(base_outdir / f"{name}-{_timestamp()}")
    evolve_opts = dict(evolve_opts)  # shallow copy
    evolve_opts.setdefault("outdir", str(outdir))

    log(f"[RUN] experiment={name} outdir={outdir}")

    evolve_cmd = _build_cmd("eoh_evolve_main.py", evolve_opts)
    log(f"[CMD] {' '.join(evolve_cmd)}")
    if not dry_run:
        subprocess.run(evolve_cmd, check=True)

    if exp.get("post_export"):
        export_opts = exp.get("post_export_options", {})
        export_opts = dict(export_opts)
        export_opts.setdefault("outdir", str(outdir))
        export_opts.setdefault("csv", evolve_opts.get("csv"))
        export_opts.setdefault("symbol", evolve_opts.get("symbol"))
        export_opts.setdefault("train_start", evolve_opts.get("train_start"))
        export_opts.setdefault("train_end", evolve_opts.get("train_end"))
        export_opts.setdefault("test_start", evolve_opts.get("test_start"))
        export_opts.setdefault("test_end", evolve_opts.get("test_end"))

        export_cmd = _build_cmd("eoh_post_export.py", export_opts)
        log(f"[CMD] {' '.join(export_cmd)}")
        if not dry_run:
            subprocess.run(export_cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/experiments/sample.yaml", help="Path to YAML configuration.")
    parser.add_argument("--outdir", default="runs", help="Base output directory for experiment artefacts.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}

    experiments = config.get("experiments") or []
    if not experiments:
        raise ValueError(f"No experiments defined in {config_path}")

    base_outdir = ensure_dir(args.outdir)
    for exp in experiments:
        run_experiment(exp, base_outdir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
