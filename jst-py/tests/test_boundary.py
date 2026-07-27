"""边界条件：远场黎曼不变量、固壁镜像、周向周期."""

from __future__ import annotations

import math

import numpy as np
import pytest

import boundary as bd
import classconfig as cc
import solvesupple as ss
from conftest import set_uniform


def test_riemann_recovers_freestream_when_interior_is_freestream(case):
    """内部为来流状态时，远场面必须逐位还原来流.

    R⁺ = vₙ,in + 2c_in/(γ−1)、R⁻ = vₙ,∞ − 2c∞/(γ−1)，两者相同 ⇒ vₙ,face = vₙ,∞、
    c_face = c∞，等熵关系再给出 ρ∞、p∞、T∞。这是远场边界最基本的相容性。
    """
    ss.riemann_main()
    for j in range(1, cc.j_total + 1):
        f = cc.Facelist_tau[cc.i_total][j]
        assert f.T == pytest.approx(cc.T, rel=1e-11)
        assert f.rho == pytest.approx(cc.rholl, rel=1e-11)
        assert f.p == pytest.approx(cc.P, rel=1e-11)
        assert f.u == pytest.approx(cc.ull, abs=1e-8 * max(abs(cc.ull), 1.0))
        assert f.v == pytest.approx(cc.vll, abs=1e-8 * max(abs(cc.cll), 1.0))


def test_riemann_supersonic_detection():
    assert bd.ifsupersonic() == (math.hypot(cc.ull, cc.vll) >= cc.cll)


def test_riemann_inflow_gets_freestream_nutilde(case):
    """入流处的 ν̃ 必须取来流值而非 0（B10 回归）."""
    ss.riemann_main()
    inflow_seen = False
    for j in range(1, cc.j_total + 1):
        f = cc.Facelist_tau[cc.i_total][j]
        nvec = np.array([f.nx, f.ny])
        if f.u * nvec[0] + f.v * nvec[1] <= 0:      # 法向速度向内 ⇒ 入流
            inflow_seen = True
            assert f.miubl == pytest.approx(cc.miublll, rel=1e-14)
    assert inflow_seen, "该算例应当存在入流面"


def test_riemann_outflow_extrapolates_nutilde(case):
    """出流处 ν̃ 由内部三点 Lagrange 外插，且必须为正."""
    ss.riemann_main()
    for j in range(1, cc.j_total + 1):
        f = cc.Facelist_tau[cc.i_total][j]
        if f.u * f.nx + f.v * f.ny > 0:
            assert f.miubl > 0


def test_freestream_nutilde_matches_initialization():
    """远场 ν̃∞ 与流场初始化用的是同一个定义（0.1·ν∞），避免边界/内场不自洽."""
    mu = cc.mu0 * (cc.T / cc.T0) ** 1.5 * (cc.T0 + cc.Ts) / (cc.T + cc.Ts)
    assert cc.miublll == pytest.approx(0.1 * mu / cc.rholl, rel=1e-14)


# ── 固壁镜像 ───────────────────────────────────────────────────

def test_wall_ghost_mirrors_velocity(case):
    """壁面 ghost 的速度与湍流量取内层的相反数（无滑移）."""
    ss.imagination_mesh_update()
    for im in range(1, cc.IM + 1):
        for j in range(1, cc.j_total + 1):
            g = cc.CellList[cc.i_total + im - 1][j]
            inner = cc.CellList[im][j]
            assert g.u == pytest.approx(-inner.u, rel=1e-15)
            assert g.v == pytest.approx(-inner.v, rel=1e-15)
            assert g.miubl == pytest.approx(-inner.miubl, rel=1e-15)


def test_wall_ghost_scalars_match_same_layer(case):
    """标量必须取**同一层** (im) 而非恒取第 1 层（B8 回归）."""
    # 造一个沿 i 变化的场，否则各层标量相同、测不出差别
    for i in range(1, cc.i_total):
        for j in range(1, cc.j_total + 1):
            c = cc.CellList[i][j]
            c.p = 101325.0 * (1.0 + 0.02 * i)
            c.T = c.p / (cc.R * c.rho)
            c.E = c.p / (c.rho * (cc.gamma - 1)) + 0.5 * (c.u ** 2 + c.v ** 2)
            c.H = c.E + c.p / c.rho
            c.c = math.sqrt(cc.gamma * cc.R * c.T)
            c.formvars()
    ss.imagination_mesh_update()
    for im in range(1, cc.IM + 1):
        for j in range(1, cc.j_total + 1):
            g = cc.CellList[cc.i_total + im - 1][j]
            assert g.p == pytest.approx(cc.CellList[im][j].p, rel=1e-15)
            assert g.rho == pytest.approx(cc.CellList[im][j].rho, rel=1e-15)


def test_wall_ghost_no_mass_flux_through_wall(case):
    """镜像 BC 应使壁面上的法向质量通量为 0（无穿透）."""
    set_uniform(rho=1.2, u=50.0, v=10.0, p=101325.0, nut=1e-4)
    ss.imagination_mesh_update()
    ss.calc_convect()
    for j in range(1, cc.j_total + 1):
        f = cc.Facelist_tau[1][j]
        assert f.Flux[1] == pytest.approx(0.0, abs=1e-9), f"mass leaks through wall at j={j}"


# ── 周向周期 ───────────────────────────────────────────────────

def test_periodic_ghosts_alias_correct_cells(case):
    """左 ghost ← 高 j 端、右 ghost ← 低 j 端，索引不能串."""
    for i in range(1, cc.i_total):
        for j in range(1, cc.j_total + 1):
            cc.CellList[i][j].rho = 1.0 + 0.01 * j     # 沿 j 单调的标记场
    ss.imagination_mesh_update()
    for i in range(1, cc.i_total):
        for im in range(1, cc.IM + 1):
            assert cc.CellList[i][cc.j_total + im].rho == \
                   pytest.approx(cc.CellList[i][cc.j_total - im + 1].rho)
            assert cc.CellList[i][cc.j_total + cc.IM + im].rho == \
                   pytest.approx(cc.CellList[i][im].rho)


def test_ghost_rows_do_not_grow(case):
    """反复调用 update 不得改变 CellList 的尺寸（A7 回归：原代码每级都 append）."""
    before = (len(cc.CellList), [len(r) for r in cc.CellList])
    for _ in range(5):
        ss.imagination_mesh_update()
    after = (len(cc.CellList), [len(r) for r in cc.CellList])
    assert before == after, "ghost mesh grew across updates — memory leak / index drift"


def test_ghost_layout_indices(case):
    """虚拟网格的布局约定固定下来，Rust 侧照此实现."""
    n_rows = len(cc.CellList)
    assert n_rows == cc.i_total + 2 * cc.IM          # 0 占位 + 物理 + 壁面 + 远场
    for i in range(1, cc.i_total):
        assert len(cc.CellList[i]) == cc.j_total + 2 * cc.IM + 1
    # 壁面 ghost 与远场 ghost 的 index 标签不得重名（B 组里的索引冲突）
    labels = {cc.CellList[cc.i_total + im - 1][1].index for im in range(1, cc.IM + 1)}
    labels |= {cc.CellList[cc.i_total + cc.IM + im - 1][1].index for im in range(1, cc.IM + 1)}
    assert len(labels) == 2 * cc.IM
