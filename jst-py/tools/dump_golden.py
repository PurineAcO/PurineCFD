#!/usr/bin/env python3
"""从 Python 基线导出 golden 参考数据.

导出的是**分级中间量**而不只是最终解：几何 → 初始化 → 单个 RK 级的四个残差
分项 → N 步后的全场。这样 Rust 侧一旦对不上，可以立刻定位到具体是哪个 kernel
写错了，而不是只知道"最终结果不一样"。

    uv run python tools/dump_golden.py --mesh fangdata.txt --steps 20 \
        -o tests/golden/cyl_10x12.json

JSON 顶层结构::

    {
      "meta":     {mesh, i_total, j_total, steps, warmup, config…},
      "geometry": {cell_vol, cell_x, cell_y, cell_sad, tau_nx…, n_nx…},
      "init":     {rho, p, T, u, v, E, H, c, ma, miubl, U1…U5},
      "terms":    {ugrad_x…, Fc1…Fc5, Fv1…Fv5, Fd1…Fd5, S5, dt}   ← warmup 步之后
      "final":    {steps, residual, totaltime, rho, p, T, u, v, miubl}
    }

所有数组均按 `i` 外循环、`j` 内循环展平（行主序），长度 = (i_total-1)*j_total。
浮点用 Python 的 `repr`（最短往返表示）写出，因此 f64 逐位无损。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import classconfig as cc  # noqa: E402
import geometry as geo  # noqa: E402
import initialize as ini  # noqa: E402
import meshreading as mr  # noqa: E402
import solvemain as sm  # noqa: E402
import solvesupple as ss  # noqa: E402


def _cells():
    """按行主序遍历全部物理单元."""
    for i in range(1, cc.i_total):
        for j in range(1, cc.j_total + 1):
            yield cc.CellList[i][j]


def _field(getter):
    return [float(getter(c)) for c in _cells()]


def dump_geometry() -> dict:
    tau, nfa = [], []
    for i in range(1, cc.i_total + 1):
        for j in range(1, cc.j_total + 1):
            tau.append(cc.Facelist_tau[i][j])
    for i in range(1, cc.i_total):
        for j in range(1, cc.j_total + 1):
            nfa.append(cc.FaceList_n[i][j])
    return {
        "cell_vol": _field(lambda c: c.vol),
        "cell_x": _field(lambda c: c.x),
        "cell_y": _field(lambda c: c.y),
        "cell_sad": _field(lambda c: c.sad),
        "tau_nx": [f.nx for f in tau], "tau_ny": [f.ny for f in tau],
        "tau_mx": [f.mx for f in tau], "tau_my": [f.my for f in tau],
        "n_nx": [f.nx for f in nfa], "n_ny": [f.ny for f in nfa],
        "n_mx": [f.mx for f in nfa], "n_my": [f.my for f in nfa],
    }


def dump_state() -> dict:
    return {
        "rho": _field(lambda c: c.rho), "p": _field(lambda c: c.p),
        "T": _field(lambda c: c.T), "u": _field(lambda c: c.u),
        "v": _field(lambda c: c.v), "E": _field(lambda c: c.E),
        "H": _field(lambda c: c.H), "c": _field(lambda c: c.c),
        "ma": _field(lambda c: c.ma), "miubl": _field(lambda c: c.miubl),
        "U1": _field(lambda c: c.U[1]), "U2": _field(lambda c: c.U[2]),
        "U3": _field(lambda c: c.U[3]), "U4": _field(lambda c: c.U[4]),
        "U5": _field(lambda c: c.U[5]),
    }


def dump_terms() -> dict:
    """跑**一个 RK 级**的全部 kernel,导出四个残差分项与梯度."""
    dt = ss.min_timestep()
    ss.riemann_main()
    ss.imagination_mesh_update()
    ss.calc_convect()
    ss.calc_grad()
    ss.calc_diffusion()
    ss.calc_dissipation()
    ss.calc_source()
    out = {"dt": dt}
    for name, get in (("ugrad", lambda c: c.ugrad), ("vgrad", lambda c: c.vgrad),
                      ("Tgrad", lambda c: c.Tgrad), ("miublgrad", lambda c: c.miublgrad)):
        out[f"{name}_x"] = _field(lambda c, g=get: g(c)[1])
        out[f"{name}_y"] = _field(lambda c, g=get: g(c)[2])
    for k in range(1, 6):
        out[f"Fc{k}"] = _field(lambda c, k=k: c.Fc[k])
        out[f"Fv{k}"] = _field(lambda c, k=k: c.Fv[k])
        out[f"Fd{k}"] = _field(lambda c, k=k: c.Fd[k])
    out["S5"] = _field(lambda c: c.S[5])
    out["chi"] = _field(lambda c: c.chi)
    out["fv1"] = _field(lambda c: c.fv1)
    return out


def build(mesh: str, steps: int, warmup: int = 3) -> dict:
    cc.reset_state()
    mr.read_mesh(mesh)
    geo.calc_cell_vol()
    geo.calc_cell_center()
    geo.calc_face_direction_tau()
    geo.calc_face_direction_n()
    geo.calc_most_near_walldistance()
    ini.initialization()
    cc.shockwave_tau = __import__("numpy").zeros((cc.i_total + cc.IM + 1, cc.j_total + 1))
    cc.shockwave_n = __import__("numpy").zeros((cc.i_total, cc.j_total + cc.IM + 1))
    cc.density_table = __import__("numpy").zeros((cc.i_total + 1, cc.j_total + 1))
    ss.formvars_main()
    ss.riemann_main()
    ss.imagination_mesh_create()

    doc = {
        "meta": {
            "mesh": Path(mesh).name, "i_total": cc.i_total, "j_total": cc.j_total,
            "n_cells": (cc.i_total - 1) * cc.j_total, "steps": steps, "warmup": warmup,
            "gamma": cc.gamma, "R": cc.R, "CFL": cc.CFL, "IM": cc.IM,
            "Ma": cc.Ma, "AOA": cc.AOA, "T_inf": cc.T, "P_inf": cc.P,
            "k2": cc.k2, "k4": cc.k4,
        },
        "geometry": dump_geometry(),
        "init": dump_state(),
    }

    # 先跑 warmup 步再导出残差分项:初始场是均匀的,此时梯度/粘性项/源项在两侧
    # 实现里都只是"零的舍入噪声",逐点比对没有意义。跑几步让流场发展出真实结构后,
    # 这些量才具有可比的量级。
    for step in range(1, warmup + 1):
        sm.RK(step)
    doc["terms"] = dump_terms()

    # dump_terms 只算残差分项、不推进 U,可以直接接着往下跑
    residual = ss.calc_residual()
    done = warmup
    for step in range(warmup + 1, steps + 1):
        sm.RK(step)
        residual = ss.calc_residual()
        done = step
        if residual < cc.targetres:
            break
    doc["final"] = {"steps": done, "residual": residual, "totaltime": cc.totaltime,
                    **{k: v for k, v in dump_state().items()
                       if k in ("rho", "p", "T", "u", "v", "miubl")}}
    return doc


def main() -> None:
    ap = argparse.ArgumentParser(description="dump golden reference data")
    ap.add_argument("--mesh", default="fangdata.txt")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3,
                    help="导出残差分项前先推进的步数,使各场脱离均匀初始态")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    doc = build(args.mesh, args.steps, args.warmup)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(doc, indent=1), encoding="utf-8")
    m = doc["meta"]
    print(f"wrote {args.out}: {m['i_total']}x{m['j_total']} mesh, "
          f"{m['n_cells']} cells, {doc['final']['steps']} steps, "
          f"residual {doc['final']['residual']:.6e}")


if __name__ == "__main__":
    main()
