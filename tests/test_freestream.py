"""自由来流保持性 —— 离散格式最强的一类验证.

如果全场（含虚拟单元）取同一均匀状态，那么：

* 对流残差 `Fc` 必须精确为 0 —— 只依赖度量闭合 Σ±n = 0；
* JST 人工粘性 `Fd` 必须精确为 0 —— 一阶/三阶差分作用在相同值上；
* 梯度 `ugrad/vgrad/Tgrad/miublgrad` 必须精确为 0；
* 因此粘性扩散 `Fv` 也为 0。

这些是**机器精度**级的恒等式，任何索引错位、模板写反、法向符号搞错都会立刻
暴露。原始代码的 B4（周期面把单元与自己平均）正是被这类检查抓住的。
"""

from __future__ import annotations

import numpy as np
import pytest

import classconfig as cc
import solvesupple as ss
from conftest import physical_cells, set_uniform

UNIFORM = dict(rho=1.176, u=69.4, v=17.3, p=101325.0, nut=1.5e-4)


@pytest.fixture
def uniform(case):
    set_uniform(**UNIFORM)
    return cc


def _flux_scale():
    """用单面通量的量级做归一化基准."""
    return max(abs(cc.Facelist_tau[i][j].Flux[k])
               for i in range(1, cc.i_total + 1)
               for j in range(1, cc.j_total + 1)
               for k in range(1, 6)) or 1.0


def test_convective_residual_vanishes(uniform):
    """均匀场下平均流对流残差为 0（质量/动量/能量四个分量）."""
    ss.calc_convect()
    scale = _flux_scale()
    worst = max(abs(c.Fc[k]) for c in physical_cells() for k in range(1, 5))
    assert worst < 1e-12 * scale, f"free-stream not preserved: max|Fc| = {worst:.3e}"


def test_turbulence_convection_vanishes(uniform):
    """ν̃ 的对流分量同样为 0."""
    ss.calc_convect()
    scale = _flux_scale()
    worst = max(abs(c.Fc[5]) for c in physical_cells())
    assert worst < 1e-12 * scale


def test_gradients_vanish(uniform):
    """均匀场的 Green-Gauss 梯度精确为 0."""
    ss.calc_convect()
    ss.calc_grad()
    scale = max(abs(c.u) for c in physical_cells()) / min(c.vol for c in physical_cells())
    for c in physical_cells():
        assert abs(c.ugrad[1]) < 1e-11 * scale
        assert abs(c.ugrad[2]) < 1e-11 * scale
        assert abs(c.vgrad[1]) < 1e-11 * scale
        assert abs(c.vgrad[2]) < 1e-11 * scale
        assert abs(c.Tgrad[1]) < 1e-11 * scale
        assert abs(c.Tgrad[2]) < 1e-11 * scale


def test_jst_dissipation_vanishes(uniform):
    """均匀场的 JST 人工粘性为 0（残留仅为 U_ff−3U_f+3U_b−U_bb 的舍入噪声）."""
    ss.min_timestep()
    ss.calc_convect()
    ss.calc_dissipation()
    lam = max(cc.Facelist_tau[i][j].lambda_f
              for i in range(1, cc.i_total + 1) for j in range(1, cc.j_total + 1))
    scale = lam * cc.k4 * max(abs(c.U[k]) for c in physical_cells() for k in range(1, 6))
    worst = max(abs(c.Fd[k]) for c in physical_cells() for k in range(1, 6))
    assert worst < 1e-13 * scale, f"dissipation not vanishing: {worst:.3e} vs scale {scale:.3e}"


def test_viscous_flux_vanishes(uniform):
    """梯度为 0 ⇒ 粘性/湍流扩散通量为 0.

    归一化基准取热扩散通量的物理量级 λ_eff·cp·ΔT/h —— 若扩散项的索引或符号写错，
    残差会直接跳到该量级，而舍入噪声比它低 15 个数量级以上。
    """
    ss.calc_convect()
    ss.calc_grad()
    ss.calc_diffusion()
    T = UNIFORM["p"] / (cc.R * UNIFORM["rho"])
    mu = cc.mu0 * (T / cc.T0) ** 1.5 * (cc.T0 + cc.Ts) / (T + cc.Ts)
    h = np.sqrt(min(c.vol for c in physical_cells()))
    scale = (mu / cc.Pr + mu / cc.Prt) * cc.cp * T / h
    worst = max(abs(c.Fv[k]) for c in physical_cells() for k in range(1, 5))
    assert worst < 1e-13 * scale, f"viscous flux not vanishing: {worst:.3e} vs scale {scale:.3e}"


def test_shock_sensor_vanishes_in_smooth_flow(uniform):
    """压力均匀 ⇒ 激波探测器 ν = |p−2p+p|/(p+2p+p) = 0，二阶耗散被完全关掉."""
    ss.min_timestep()
    ss.calc_convect()
    ss.calc_dissipation()
    assert np.max(np.abs(cc.shockwave_tau)) == 0.0
    assert np.max(np.abs(cc.shockwave_n)) == 0.0
    for i in range(1, cc.i_total + 1):
        for j in range(1, cc.j_total + 1):
            assert cc.Facelist_tau[i][j].epsilon[1] == 0.0
            assert cc.Facelist_tau[i][j].epsilon[2] == pytest.approx(cc.k4)


def test_periodic_seam_is_transparent(case):
    """周期切割线两侧不应有人为的间断.

    构造一个**只沿径向变化**的场（周向严格周期）。那么 j=1 与 j=j_total 的
    单元状态相同，切割线上的面通量应与内部任一条径向面完全一致。
    修复前的 B4 会让 `FaceList_n[i][1]` 把单元 1 与其自身副本平均，这里必然失败。
    """
    for i in range(1, cc.i_total):
        rho = 1.0 + 0.05 * i
        for j in range(1, cc.j_total + 1):
            c = cc.CellList[i][j]
            c.rho, c.u, c.v, c.p = rho, 50.0, 0.0, 101325.0
            c.T = c.p / (cc.R * c.rho)
            c.E = c.p / (c.rho * (cc.gamma - 1)) + 0.5 * (c.u**2 + c.v**2)
            c.H = c.E + c.p / c.rho
            c.c = np.sqrt(cc.gamma * cc.R * c.T)
            c.miubl = 1e-4
            c.formvars()
    ss.imagination_mesh_update()
    ss.calc_convect()

    # 切割线面 (j=1) 的守恒量应等于两侧单元的平均 —— 而两侧单元状态相同
    for i in range(1, cc.i_total):
        seam = cc.FaceList_n[i][1]
        expected = 0.5 * (cc.CellList[i][1].U + cc.CellList[i][cc.j_total].U)
        np.testing.assert_allclose(seam.FU[1:6], expected[1:6], rtol=1e-14, atol=0)


def test_seam_flux_matches_interior(case):
    """周向均匀场下，切割线上的通量应与相邻内部径向面逐位一致."""
    set_uniform(**UNIFORM)
    ss.calc_convect()
    for i in range(1, cc.i_total):
        seam, interior = cc.FaceList_n[i][1], cc.FaceList_n[i][2]
        # 归一化掉法向差异后比较
        for f in (seam, interior):
            assert f.rho == pytest.approx(UNIFORM["rho"], rel=1e-14)
            assert f.u == pytest.approx(UNIFORM["u"], rel=1e-14)
            assert f.v == pytest.approx(UNIFORM["v"], rel=1e-14)
