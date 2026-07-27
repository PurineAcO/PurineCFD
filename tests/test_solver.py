"""时间推进、JST 耗散与收敛性的集成测试."""

from __future__ import annotations

import math

import numpy as np
import pytest

import classconfig as cc
import dissipation as hs
import solvemain as sm
import solvesupple as ss
from conftest import physical_cells, set_uniform, setup_case


# ── Runge-Kutta ────────────────────────────────────────────────

def test_five_rk_stages_are_used():
    """必须跑满 5 级，末级系数为 1（B1 回归：原实现只跑 4 级）."""
    assert cc.RK_STAGES == 5
    assert cc.RK[cc.RK_STAGES] == 1.0
    assert cc.RK[1:6] == (0.25, 1 / 6, 0.375, 0.5, 1.0)


def test_u_former_is_deep_copied(case):
    """U_former 必须是 U 的**拷贝**（C1 回归）."""
    sm.RK(1)
    for c in physical_cells():
        assert c.U_former is not c.U


def test_timestep_advances_once_per_step(case):
    """物理时间每步只累加一次 Δt，不是每个 RK 级都累加（B12 回归）."""
    cc.totaltime = 0.0
    sm.RK(1)
    t1 = cc.totaltime
    sm.RK(2)
    t2 = cc.totaltime
    # 两步的 Δt 量级相当；若按级累加，单步会是 5 倍
    assert 0.2 < t1 / (t2 - t1) < 5.0
    assert t1 > 0


def test_local_timestep_respects_cfl(case):
    """局部时间步应满足 Δt = CFL·V/(|λ|)，且全局取最小值."""
    dt = ss.min_timestep()
    assert dt > 0 and math.isfinite(dt)
    assert dt == pytest.approx(min(c.localdt for c in physical_cells()), rel=1e-15)
    for c in physical_cells():
        assert c.dt == pytest.approx(dt, rel=1e-15)


def test_timestep_scales_with_cfl(case):
    """Δt 应与 CFL 成正比."""
    dt1 = ss.min_timestep()
    old = cc.CFL
    try:
        cc.CFL = old * 0.5
        dt2 = ss.min_timestep()
    finally:
        cc.CFL = old
    assert dt2 == pytest.approx(0.5 * dt1, rel=1e-12)


# ── JST 人工粘性 ───────────────────────────────────────────────

def test_shock_sensor_detects_pressure_jump():
    """压力出现 2Δ 型跳跃时，探测器应给出 O(1) 的值."""
    cc.reset_state()
    cc.shockwave_tau = np.zeros((8, 8))
    cells = []
    for p in (1.0, 1.0, 3.0):
        c = cc.cell_class((1, 1))
        c.p = p
        cells.append(c)
    hs.shockwave_catcher((1, 1), "tau", *cells)
    # |1 − 2·1 + 3| / (1 + 2·1 + 3) = 2/6
    assert cc.shockwave_tau[1][1] == pytest.approx(2.0 / 6.0, rel=1e-14)
    cc.reset_state()


def test_shock_sensor_zero_in_smooth_field():
    cc.reset_state()
    cc.shockwave_tau = np.zeros((8, 8))
    cells = []
    for p in (2.0, 2.0, 2.0):
        c = cc.cell_class((1, 1))
        c.p = p
        cells.append(c)
    hs.shockwave_catcher((1, 1), "tau", *cells)
    assert cc.shockwave_tau[1][1] == 0.0
    cc.reset_state()


def test_epsilon4_switches_off_at_shock():
    """强激波处 ε² 增大、ε⁴ = max(0, k4 − ε²) 被关掉（防止四阶项产生振荡）."""
    cc.reset_state()
    cc.shockwave_n = np.zeros((8, 8))
    cc.shockwave_n[1][1] = 0.9
    f = cc.face_class((1, 1))
    hs.adaptive_dissipation(f, "n")
    assert f.epsilon[1] == pytest.approx(cc.k2 * 0.9)
    assert f.epsilon[2] == 0.0
    cc.reset_state()


def test_dissipation_is_antisymmetric_in_stencil():
    """一阶差分项 d1U = U_f − U_b 反号，交换前后单元应整体反号."""
    def make(U):
        c = cc.cell_class((1, 1))
        c.U = np.array(U, dtype=float)
        return c
    a, b, cc_, d = (make([0, 1, 0, 0, 0, 0]), make([0, 2, 0, 0, 0, 0]),
                    make([0, 4, 0, 0, 0, 0]), make([0, 8, 0, 0, 0, 0]))
    f1 = cc.face_class((1, 1)); f1.lambda_f = 1.0; f1.epsilon = np.array([0.0, 1.0, 0.0])
    f2 = cc.face_class((1, 1)); f2.lambda_f = 1.0; f2.epsilon = np.array([0.0, 1.0, 0.0])
    hs.form_JST_dissipation_term(f1, a, b, cc_, d)      # b|a|(f)|c|d
    hs.form_JST_dissipation_term(f2, cc_, d, a, b)      # d|c|(f)|a|b
    assert f1.Dissipation[1] == pytest.approx(-f2.Dissipation[1], rel=1e-15)


def test_fourth_order_dissipation_annihilates_linear_profile():
    """三阶差分 U_ff − 3U_f + 3U_b − U_bb 对**二次以下**的分布恒为 0."""
    def make(val):
        c = cc.cell_class((1, 1))
        c.U = np.array([0.0, val, 0, 0, 0, 0])
        return c
    # 位置 −2,−1,0,1 上取二次多项式 q(x) = 3x² + 2x + 5
    q = lambda x: 3 * x * x + 2 * x + 5
    f = cc.face_class((1, 1)); f.lambda_f = 1.0; f.epsilon = np.array([0.0, 0.0, 1.0])
    hs.form_JST_dissipation_term(f, make(q(-1)), make(q(-2)), make(q(0)), make(q(1)))
    assert f.Dissipation[1] == pytest.approx(0.0, abs=1e-12)


# ── 收敛与稳健性 ───────────────────────────────────────────────

@pytest.mark.parametrize("steps", [12])
def test_state_stays_physical(case, steps):
    """推进若干步后所有单元仍须满足 ρ>0、p>0、T>0 且全部有限."""
    for s in range(1, steps + 1):
        sm.RK(s)
    for c in physical_cells():
        assert c.rho > 0 and math.isfinite(c.rho)
        assert c.p > 0 and math.isfinite(c.p)
        assert c.T > 0 and math.isfinite(c.T)
        assert math.isfinite(c.u) and math.isfinite(c.v)
        assert math.isfinite(c.miubl)


def test_residual_decreases(case):
    """初始瞬态过后残差应总体下降."""
    res = []
    for s in range(1, 40 + 1):
        sm.RK(s)
        res.append(ss.calc_residual())
    assert res[-1] < res[0], f"residual grew: {res[0]:.3e} → {res[-1]:.3e}"
    assert all(math.isfinite(r) for r in res)


def test_residual_is_zero_for_converged_uniform_euler_state(case):
    """均匀场下平均流的残差分项全为 0 ⇒ U 不会被平均流项推动."""
    set_uniform(rho=cc.rholl, u=cc.ull, v=cc.vll, p=cc.P, nut=cc.miublll)
    ss.min_timestep()
    ss.calc_convect()
    ss.calc_grad()
    ss.calc_diffusion()
    ss.calc_dissipation()
    # 以单面通量量级归一化：四个面的通量相消，残差只应剩机器精度
    scale = max(abs(cc.Facelist_tau[i][j].Flux[k])
                for i in range(1, cc.i_total + 1)
                for j in range(1, cc.j_total + 1) for k in range(1, 6))
    for c in physical_cells():
        for k in range(1, 5):
            total = c.Fc[k] - c.Fv[k] - c.Fd[k]
            assert abs(total) < 1e-14 * scale, \
                f"mean-flow residual {total:.3e} at {c.index}[{k}] (scale {scale:.3e})"


def test_convergence_to_target_residual(tmp_path):
    """完整跑到收敛：应在合理步数内达到 1e-8 且解保持物理."""
    setup_case()
    res = float("inf")
    step = 0
    for step in range(1, 3000 + 1):
        sm.RK(step)
        res = ss.calc_residual()
        if res < 1e-8:
            break
    assert res < 1e-8, f"failed to converge, residual = {res:.3e} after {step} steps"
    assert all(c.rho > 0 and c.p > 0 for c in physical_cells())
    cc.reset_state()


def test_solver_runs_on_finer_mesh(tmp_path):
    """在更细的网格（含近壁加密）上也应稳定推进."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tools.genmesh import gen_o_mesh, write_mesh

    p = tmp_path / "fine.txt"
    write_mesh(p, *gen_o_mesh(12, 32, stretch=1.15))
    setup_case(p)
    for s in range(1, 15 + 1):
        sm.RK(s)
    assert all(c.rho > 0 and c.p > 0 and math.isfinite(c.miubl) for c in physical_cells())
    cc.reset_state()
