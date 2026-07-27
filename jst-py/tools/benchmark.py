#!/usr/bin/env python3
"""Python 基线 与 Rust 实现的同题对比基准.

对每个网格规模跑固定步数,报告 ms/步 与 ns/单元/步,并给出加速比。
两侧读同一份 `config.json` 与同一个网格文件,确保是同一道题。

    uv run python tools/benchmark.py --steps 20
    uv run python tools/benchmark.py --steps 20 --sizes 9x24 17x40 33x80
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.genmesh import gen_o_mesh, write_mesh  # noqa: E402

RUST_BIN = ROOT / "jst-rs" / "target" / "release" / "jst"


def time_python(mesh: Path, steps: int) -> tuple[float, float]:
    """在**子进程**里跑 Python 基线,返回 (总秒数, 最终残差)。

    必须用子进程:基线用的是模块级全局状态,同一进程内跑第二个算例会互相污染。
    """
    code = f"""
import sys, time, json
sys.path.insert(0, {str(ROOT)!r})
import classconfig as cc, meshreading as mr, geometry as geo
import initialize as ini, solvesupple as ss, solvemain as sm
import numpy as np
cc.reset_state()
mr.read_mesh({str(mesh)!r})
geo.calc_cell_vol(); geo.calc_cell_center()
geo.calc_face_direction_tau(); geo.calc_face_direction_n()
geo.calc_most_near_walldistance()
ini.initialization()
cc.shockwave_tau = np.zeros((cc.i_total+cc.IM+1, cc.j_total+1))
cc.shockwave_n  = np.zeros((cc.i_total, cc.j_total+cc.IM+1))
cc.density_table = np.zeros((cc.i_total+1, cc.j_total+1))
ss.formvars_main(); ss.riemann_main(); ss.imagination_mesh_create()
t0 = time.perf_counter()
for s in range(1, {steps}+1):
    sm.RK(s)
res = ss.calc_residual()
print(json.dumps({{"wall": time.perf_counter()-t0, "residual": res}}))
"""
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, check=True
    )
    d = json.loads(out.stdout.strip().splitlines()[-1])
    return d["wall"], d["residual"]


def time_rust(mesh: Path, steps: int, threads: int | None) -> tuple[float, float]:
    cmd = [str(RUST_BIN), "--mesh", str(mesh), "--steps", str(steps),
           "--quiet", "--no-output"]
    if threads:
        cmd += ["--threads", str(threads)]
    t0 = time.perf_counter()
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, check=True)
    wall_total = time.perf_counter() - t0
    solve = re.search(r"wall clock:\s+([0-9.]+) s", out.stdout)
    res = re.search(r"final residual:\s+([0-9.eE+-]+)", out.stdout)
    # 用求解器自报的求解时间(排除进程启动与网格读取)
    return (float(solve.group(1)) if solve else wall_total,
            float(res.group(1)) if res else float("nan"))


def parse_size(s: str) -> tuple[int, int]:
    m = re.fullmatch(r"(\d+)x(\d+)", s)
    if not m:
        raise argparse.ArgumentTypeError(f"bad size {s!r}, expected like 33x80")
    return int(m.group(1)), int(m.group(2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--sizes", type=parse_size, nargs="+",
                    default=[(10, 12), (17, 40), (33, 80), (65, 128), (129, 256)])
    ap.add_argument("--threads", type=int, default=None,
                    help="固定 Rust 线程数(默认由求解器按规模自动选)")
    ap.add_argument("--max-python-cells", type=int, default=20000,
                    help="超过这个单元数就跳过 Python(太慢)")
    args = ap.parse_args()

    if not RUST_BIN.exists():
        sys.exit(f"missing {RUST_BIN}; run `cargo build --release` in jst-rs/ first")

    mesh_dir = ROOT / "meshes"
    rows = []
    for rings, nj in args.sizes:
        cells = (rings - 1) * nj
        mesh = mesh_dir / f"bench_{rings}x{nj}.txt"
        if not mesh.exists():
            write_mesh(mesh, *gen_o_mesh(rings, nj))

        rust_s, rust_res = time_rust(mesh, args.steps, args.threads)
        if cells <= args.max_python_cells:
            py_s, py_res = time_python(mesh, args.steps)
        else:
            py_s, py_res = float("nan"), float("nan")

        rows.append({
            "mesh": f"{rings}x{nj}", "cells": cells,
            "py_ms": py_s / args.steps * 1e3, "rs_ms": rust_s / args.steps * 1e3,
            "py_ns": py_s / args.steps / cells * 1e9,
            "rs_ns": rust_s / args.steps / cells * 1e9,
            "speedup": py_s / rust_s if py_s == py_s else float("nan"),
            "py_res": py_res, "rs_res": rust_res,
        })
        r = rows[-1]
        print(f"{r['mesh']:>9} {r['cells']:>8} cells | "
              f"python {r['py_ms']:>10.3f} ms/step | rust {r['rs_ms']:>8.3f} ms/step | "
              f"speedup {r['speedup']:>7.1f}x")

    print()
    print(f"| mesh | cells | Python ms/step | Rust ms/step | Python ns/cell | "
          f"Rust ns/cell | speedup |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        py_ms = "—" if r["py_ms"] != r["py_ms"] else f"{r['py_ms']:.2f}"
        py_ns = "—" if r["py_ns"] != r["py_ns"] else f"{r['py_ns']:.0f}"
        sp = "—" if r["speedup"] != r["speedup"] else f"**{r['speedup']:.0f}x**"
        print(f"| {r['mesh']} | {r['cells']:,} | {py_ms} | {r['rs_ms']:.3f} | "
              f"{py_ns} | {r['rs_ns']:.0f} | {sp} |")

    print("\n残差一致性检查(相对偏差):")
    for r in rows:
        if r["py_res"] == r["py_res"]:
            rel = abs(r["rs_res"] - r["py_res"]) / abs(r["py_res"])
            flag = "ok" if rel < 1e-6 else "MISMATCH"
            print(f"  {r['mesh']:>9}: python {r['py_res']:.6e}  "
                  f"rust {r['rs_res']:.6e}  rel {rel:.2e}  {flag}")


if __name__ == "__main__":
    main()
