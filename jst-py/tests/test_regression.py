"""Golden 回归：把 Python 基线钉死，作为 Rust 重写的比对基准.

`tests/golden/*.json` 由 `tools/dump_golden.py` 生成，含四段：几何 / 初始场 /
单个 RK 级的残差分项 / N 步后的全场。Rust 侧读同一份文件做同样的比对
（见 `jst-rs/tests/golden.rs`），因此两边只要有一处 kernel 不一致，就能定位到
具体是几何、梯度、对流、扩散、耗散还是源项。

若确需更新 golden（例如有意改动数值格式），重新运行::

    uv run python tools/dump_golden.py --mesh fangdata.txt --steps 20 \
        -o tests/golden/cyl_10x12.json
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import classconfig as cc
import solvemain as sm
import solvesupple as ss
from conftest import GOLDEN_DIR, ROOT, physical_cells, setup_case

GOLDEN_FILES = sorted(GOLDEN_DIR.glob("*.json"))


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _cmp(got, want, name, rtol, atol=0.0):
    """按**场**比较:容差取 ``rtol·‖want‖∞ + atol``.

    逐元素相对容差在这里是错的 —— 场里总有个别单元的值比场的量级低若干数量级
    (如 AOA=0 时对称面上的 v ≈ 1e-13),对它们要求相对精度等于在比两侧各自的
    舍入噪声。以场的范数归一才是有意义的判据。
    """
    got, want = np.asarray(got, float), np.asarray(want, float)
    assert got.shape == want.shape, f"{name}: shape {got.shape} != {want.shape}"
    scale = np.max(np.abs(want)) if want.size else 0.0
    tol = atol + rtol * scale
    err = np.abs(got - want)
    if err.max(initial=0.0) > tol:
        i = int(np.argmax(err))
        raise AssertionError(
            f"{name}: worst at flat index {i}: got {got[i]!r} want {want[i]!r} "
            f"(abs err {err[i]:.3e} > tol {tol:.3e}; field scale {scale:.3e})"
        )


def _resolve_mesh(meta) -> str:
    name = meta["mesh"]
    for cand in (ROOT / name, ROOT / "meshes" / name):
        if cand.exists():
            return str(cand)
    pytest.skip(f"mesh {name} not found")


@pytest.fixture(params=GOLDEN_FILES, ids=lambda p: p.stem)
def golden(request):
    doc = _load(request.param)
    setup_case(_resolve_mesh(doc["meta"]))
    yield doc
    cc.reset_state()


def _fields(names, getter):
    return {n: [float(getter(c, n)) for c in physical_cells()] for n in names}


def test_meta_matches(golden):
    m = golden["meta"]
    assert (cc.i_total, cc.j_total) == (m["i_total"], m["j_total"])
    assert (cc.i_total - 1) * cc.j_total == m["n_cells"]
    for key, val in (("gamma", cc.gamma), ("R", cc.R), ("CFL", cc.CFL), ("IM", cc.IM),
                     ("Ma", cc.Ma), ("k2", cc.k2), ("k4", cc.k4)):
        assert val == pytest.approx(m[key]), f"config drift on {key}"


def test_geometry_matches(golden):
    """几何量必须逐位重现（纯确定性运算，无迭代）."""
    g = golden["geometry"]
    _cmp([c.vol for c in physical_cells()], g["cell_vol"], "cell_vol", 1e-15)
    _cmp([c.x for c in physical_cells()], g["cell_x"], "cell_x", 1e-14, atol=1e-15)
    _cmp([c.y for c in physical_cells()], g["cell_y"], "cell_y", 1e-14, atol=1e-15)
    _cmp([c.sad for c in physical_cells()], g["cell_sad"], "cell_sad", 1e-15)
    tau = [cc.Facelist_tau[i][j] for i in range(1, cc.i_total + 1)
           for j in range(1, cc.j_total + 1)]
    nfa = [cc.FaceList_n[i][j] for i in range(1, cc.i_total)
           for j in range(1, cc.j_total + 1)]
    for key, seq, attr in (("tau_nx", tau, "nx"), ("tau_ny", tau, "ny"),
                           ("tau_mx", tau, "mx"), ("tau_my", tau, "my"),
                           ("n_nx", nfa, "nx"), ("n_ny", nfa, "ny"),
                           ("n_mx", nfa, "mx"), ("n_my", nfa, "my")):
        _cmp([getattr(f, attr) for f in seq], g[key], key, 1e-14, atol=1e-15)


def test_initial_state_matches(golden):
    ini = golden["init"]
    for name in ("rho", "p", "T", "u", "v", "E", "H", "c", "ma", "miubl"):
        _cmp([getattr(c, name) for c in physical_cells()], ini[name], name, 1e-14, atol=1e-300)
    for k in range(1, 6):
        _cmp([c.U[k] for c in physical_cells()], ini[f"U{k}"], f"U{k}", 1e-14, atol=1e-300)


def test_single_stage_terms_match(golden):
    """单个 RK 级的四个残差分项 —— 定位 kernel 级别差异的关键用例.

    先按 golden 记录的 ``warmup`` 步推进:初始场是均匀的,此时梯度/粘性项/源项
    都只是零的舍入噪声,比对不出任何信息.
    """
    t = golden["terms"]
    for step in range(1, golden["meta"].get("warmup", 0) + 1):
        sm.RK(step)
    dt = ss.min_timestep()
    ss.riemann_main()
    ss.imagination_mesh_update()
    ss.calc_convect()
    ss.calc_grad()
    ss.calc_diffusion()
    ss.calc_dissipation()
    ss.calc_source()

    assert dt == pytest.approx(t["dt"], rel=1e-13)
    for name, attr in (("ugrad", "ugrad"), ("vgrad", "vgrad"),
                       ("Tgrad", "Tgrad"), ("miublgrad", "miublgrad")):
        _cmp([getattr(c, attr)[1] for c in physical_cells()], t[f"{name}_x"], f"{name}_x", 1e-12, atol=1e-280)
        _cmp([getattr(c, attr)[2] for c in physical_cells()], t[f"{name}_y"], f"{name}_y", 1e-12, atol=1e-280)
    _cmp([c.chi for c in physical_cells()], t["chi"], "chi", 1e-13)
    _cmp([c.fv1 for c in physical_cells()], t["fv1"], "fv1", 1e-13)
    for k in range(1, 6):
        for term in ("Fc", "Fv", "Fd"):
            _cmp([getattr(c, term)[k] for c in physical_cells()],
                 t[f"{term}{k}"], f"{term}{k}", 1e-11, atol=1e-280)
    _cmp([c.S[5] for c in physical_cells()], t["S5"], "S5", 1e-12, atol=1e-280)


def test_final_state_matches(golden):
    """跑满 golden 记录的步数后，全场必须重现.

    时间推进会放大舍入差异，所以容差比几何/初始场松，但 1e-9 仍足以抓住任何
    真正的算法性差异（一处索引写错通常带来 O(1) 的相对偏差）。
    """
    f = golden["final"]
    steps = golden["meta"]["steps"]
    for step in range(1, steps + 1):
        sm.RK(step)
        if ss.calc_residual() < cc.targetres:
            break
    residual = ss.calc_residual()
    assert residual == pytest.approx(f["residual"], rel=1e-8)
    assert cc.totaltime == pytest.approx(f["totaltime"], rel=1e-10)
    for name in ("rho", "p", "T", "u", "v", "miubl"):
        _cmp([getattr(c, name) for c in physical_cells()], f[name], name, 1e-9, atol=1e-300)


def test_golden_files_present():
    assert GOLDEN_FILES, "no golden data — run tools/dump_golden.py"
