"""Evaluate several world-model checkpoints on the exact same fixed windows.

This is the P0 shared-evaluation utility from plan9: Ratio1, Ratio2 and baseline
checkpoints must be scored on one common `WindowBatch`, otherwise their NSE
numbers are not comparable.

Example:
  .venv/bin/python scripts/eval_wm_checkpoints.py \
      --checkpoints /path/ratio1_4000.pt /path/ratio2_4000.pt \
      --windows /root/autodl-tmp/data/wm/eval_a_k32.pt \
      --out experiment/koopman-world-model/shared_eval_ratio.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
import tyro
from diag_koopman import _load_windows, _load_wm


def _f(x) -> float | None:
    v = float(x)
    return v if math.isfinite(v) else None


def main(
    checkpoints: tuple[Path, ...],
    windows: Path,
    out: Path = Path("experiment/koopman-world-model/shared_eval.json"),
    device: str = "cuda:0",
) -> None:
    w = _load_windows(windows, device)
    rows = []
    for ckpt in checkpoints:
        wm, meta = _load_wm(ckpt, None, device)
        with torch.no_grad():
            nse = wm.rollout_error(w.states, w.a_exec, w.xi)
        vals = [_f(x) for x in nse.tolist()]
        rows.append(
            {
                "checkpoint": str(ckpt),
                "params": sum(p.numel() for p in wm.parameters()),
                "meta": meta,
                "shape": {"k": int(w.states.shape[0] - 1), "n": int(w.states.shape[1])},
                "E1": vals[0] if len(vals) > 0 else None,
                "E8": vals[7] if len(vals) > 7 else None,
                "E16": vals[15] if len(vals) > 15 else None,
                "E32": vals[31] if len(vals) > 31 else None,
                "nse_curve": vals,
            }
        )
    payload = {"windows": str(windows), "rows": rows, "argv": sys.argv}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    for r in rows:
        print(f"{Path(r['checkpoint']).name:24s} E1={r['E1']} E8={r['E8']} E16={r['E16']} E32={r['E32']}")
    print(f"saved {out}")


if __name__ == "__main__":
    tyro.cli(main)
