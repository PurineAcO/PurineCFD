"""物理/热力学关系与 S-A 模型函数的单元测试."""

from __future__ import annotations

import math

import numpy as np
import pytest

import classconfig as cc
import turbulence as tb
from conftest import physical_cells


def _make_cell(rho=1.2, u=70.0, v=-15.0, p=1.0e5, nut=2e-4):
    c = cc.cell_class((1, 1))
    c.rho, c.u, c.v, c.p, c.miubl = rho, u, v, p, nut
    c.T = p / (cc.R * rho)
    c.E = p / (rho * (cc.gamma - 1)) + 0.5 * (u * u + v * v)
    c.H = c.E + p / rho
    c.c = math.sqrt(cc.gamma * cc.R * c.T)
    c.vol, c.sad = 1e-3, 0.05
    c.formvars()
    return c


# ── 守恒量 ↔ 原始量 ─────────────────────────────────────────────

def test_conservative_primitive_roundtrip():
    """U = U(W) 后再 W = W(U) 必须回到原值（机器精度）."""
    c = _make_cell()
    ref = (c.rho, c.u, c.v, c.p, c.T, c.E, c.H, c.miubl)
    c.rho = c.u = c.v = c.p = c.T = c.E = c.H = c.miubl = float("nan")
    c.form_physic_vars()
    for got, want, name in zip((c.rho, c.u, c.v, c.p, c.T, c.E, c.H, c.miubl),
                               ref, "rho u v p T E H miubl".split()):
        assert got == pytest.approx(want, rel=1e-14), name


def test_negative_density_raises():
    """非物理状态必须抛异常，而不是 print + exit(6) 把进程杀掉."""
    c = _make_cell()
    c.U[1] = -1.0
    with pytest.raises(FloatingPointError):
        c.form_physic_vars()


def test_negative_pressure_raises():
    c = _make_cell()
    c.U[4] = 1.0          # ρE 远小于动能 ⇒ p < 0
    with pytest.raises(FloatingPointError):
        c.form_physic_vars()


def test_enthalpy_definition():
    c = _make_cell()
    assert c.H == pytest.approx(c.E + c.p / c.rho, rel=1e-15)


def test_speed_of_sound_and_mach():
    c = _make_cell()
    c.form_physic_vars()
    assert c.c == pytest.approx(math.sqrt(cc.gamma * c.p / c.rho), rel=1e-14)
    assert c.ma == pytest.approx(math.hypot(c.u, c.v) / c.c, rel=1e-14)


# ── 面通量 ─────────────────────────────────────────────────────

def test_face_flux_matches_analytic_euler_flux():
    """面通量应等于解析的 Euler 通量 F·n."""
    f = cc.face_class((1, 1))
    rho, u, v, p, nut = 1.15, 60.0, -20.0, 9.5e4, 3e-4
    E = p / (rho * (cc.gamma - 1)) + 0.5 * (u * u + v * v)
    f.FU = np.array([0.0, rho, rho * u, rho * v, rho * E, rho * nut])
    f.nx, f.ny = 0.3, -0.7
    f.form_flux()
    vn = u * f.nx + v * f.ny
    assert f.Flux[1] == pytest.approx(rho * vn, rel=1e-14)
    assert f.Flux[2] == pytest.approx(rho * u * vn + p * f.nx, rel=1e-14)
    assert f.Flux[3] == pytest.approx(rho * v * vn + p * f.ny, rel=1e-14)
    assert f.Flux[4] == pytest.approx((rho * E + p) * vn, rel=1e-14)
    assert f.Flux[5] == pytest.approx(rho * nut * vn, rel=1e-14)


def test_flux_is_galilean_linear_in_normal():
    """通量对法向是线性的：F(2n) = 2F(n)。度量闭合能推出自由来流保持性的依据."""
    def flux(scale):
        f = cc.face_class((1, 1))
        f.FU = np.array([0.0, 1.2, 72.0, -18.0, 2.6e5, 3.6e-4])
        f.nx, f.ny = 0.3 * scale, -0.7 * scale
        f.form_flux()
        return f.Flux.copy()
    np.testing.assert_allclose(flux(2.0)[1:], 2.0 * flux(1.0)[1:], rtol=1e-14)


# ── Sutherland ─────────────────────────────────────────────────

def test_sutherland_at_reference_temperature():
    """T = T0 时 Sutherland 公式应精确还原 mu0."""
    mu = cc.mu0 * (cc.T0 / cc.T0) ** 1.5 * (cc.T0 + cc.Ts) / (cc.T0 + cc.Ts)
    assert mu == pytest.approx(cc.mu0, rel=1e-15)


def test_sutherland_monotonic_in_temperature():
    def mu(T):
        return cc.mu0 * (T / cc.T0) ** 1.5 * (cc.T0 + cc.Ts) / (T + cc.Ts)
    vals = [mu(T) for T in (200.0, 300.0, 400.0, 800.0)]
    assert all(b > a for a, b in zip(vals, vals[1:]))


def test_initial_viscosity_uses_sutherland_reference(bare_case):
    """初始化的分子粘度必须以 cc.T0=288.16 K 为参考温度（B7 回归）."""
    c = next(physical_cells())
    expected = cc.mu0 * (c.T / cc.T0) ** 1.5 * (cc.T0 + cc.Ts) / (c.T + cc.Ts)
    assert c.miu == pytest.approx(expected, rel=1e-14)
    assert c.miubl == pytest.approx(0.1 * expected / c.rho, rel=1e-14)


# ── Spalart-Allmaras ───────────────────────────────────────────

def test_fv1_uses_cv1_cubed():
    """fv1 = χ³/(χ³+Cv1³)。修复前分母是 Cv1，这是 B2 的回归测试."""
    c = _make_cell()
    c.ugrad = c.vgrad = c.Tgrad = c.miublgrad = np.zeros(3)
    tb.Spalart_Allmaras(c)
    chi3 = c.chi ** 3
    assert c.fv1 == pytest.approx(chi3 / (chi3 + cc.Cv1 ** 3), rel=1e-14)


@pytest.mark.parametrize("nut,expect", [(1e-12, 0.0), (1e2, 1.0)])
def test_fv1_asymptotes(nut, expect):
    """χ→0 时 fv1→0（层流极限）；χ→∞ 时 fv1→1（μt→ρν̃）."""
    c = _make_cell(nut=nut)
    c.ugrad = c.vgrad = c.Tgrad = c.miublgrad = np.zeros(3)
    tb.Spalart_Allmaras(c)
    assert c.fv1 == pytest.approx(expect, abs=1e-6)


def test_chi_is_nondimensional_viscosity_ratio():
    c = _make_cell()
    c.ugrad = c.vgrad = c.Tgrad = c.miublgrad = np.zeros(3)
    tb.Spalart_Allmaras(c)
    mu = cc.mu0 * (c.T / cc.T0) ** 1.5 * (cc.T0 + cc.Ts) / (c.T + cc.Ts)
    assert c.chi == pytest.approx(c.rho * c.miubl / mu, rel=1e-14)


def test_vorticity_magnitude_is_curl():
    """S 应等于 |∂v/∂x − ∂u/∂y|（B3 回归：修复前小 √2 倍且符号相反）."""
    c = _make_cell()
    c.ugrad = np.array([0.0, 0.0, 3.0])     # ∂u/∂y = 3
    c.vgrad = np.array([0.0, 11.0, 0.0])    # ∂v/∂x = 11
    c.Tgrad = c.miublgrad = np.zeros(3)
    tb.Spalart_Allmaras(c)
    # 用极大的壁面距离压掉 S̃ 中的 ν̃/(κ²d²) 项，使 S̃ ≈ S
    c.sad = 1e8
    tb.form_source_term(c)
    # 反解：P = Cb1(1−ft2)·S̃·ρν̃，此时 S̃ ≈ S = |11−3| = 8
    ft2 = cc.Ct3 * math.exp(-cc.Ct4 * c.chi ** 2)
    S_recovered = c.S[5] / c.vol
    # 只验证生成项方向与量级（破坏项在 d→∞ 时趋于 0）
    assert S_recovered == pytest.approx(cc.Cb1 * (1 - ft2) * 8.0 * c.U[5], rel=1e-6)


def test_source_term_only_touches_turbulence_equation():
    """S-A 源项只能出现在第 5 个方程上，平均流方程不得有源."""
    c = _make_cell()
    c.ugrad = np.array([0.0, 1.0, 3.0])
    c.vgrad = np.array([0.0, 11.0, 2.0])
    c.Tgrad = c.miublgrad = np.array([0.0, 0.5, 0.5])
    tb.Spalart_Allmaras(c)
    tb.form_source_term(c)
    assert np.all(c.S[0:5] == 0.0)
    assert np.isfinite(c.S[5])


def test_source_term_finite_at_zero_vorticity():
    """涡量为 0 时 S̃ 仍需有下限保护，r = ν̃/(S̃κ²d²) 不得除零（C3 回归）."""
    c = _make_cell(nut=0.0)
    c.ugrad = c.vgrad = c.Tgrad = c.miublgrad = np.zeros(3)
    tb.Spalart_Allmaras(c)
    tb.form_source_term(c)
    assert np.isfinite(c.S[5])


def test_fw_bounded():
    """壁面阻尼函数 fw 在 r ∈ [0, rmax] 上应有界."""
    for r in np.linspace(0.0, cc.rmax, 64):
        g = r + cc.Cw2 * (r ** 6 - r)
        fw = g * ((1 + cc.Cw3 ** 6) / (g ** 6 + cc.Cw3 ** 6)) ** (1 / 6)
        assert 0.0 <= fw < 10.0


def test_cw1_closure_relation():
    """Cw1 = Cb1/κ² + (1+Cb2)/σ；配置里的 sigma 是 1/σ，故写作乘法."""
    assert cc.Cw1 == pytest.approx(cc.Cb1 / cc.kappa ** 2 + (1 + cc.Cb2) * cc.sigma, rel=1e-15)
    assert cc.Cw1 == pytest.approx(0.1355 / 0.41 ** 2 + 1.622 * 1.5, rel=1e-12)


def test_prandtl_numbers_are_physical():
    """空气：层流 Pr ≈ 0.71 < 湍流 Prt ≈ 0.9（B11 回归：配置里原本互换）."""
    assert 0.6 < cc.Pr < 0.8
    assert 0.85 < cc.Prt < 1.0
    assert cc.Pr < cc.Prt


def test_diffusion_tensor_symmetry():
    """粘性应力张量必须对称：τxy 出现在 (2,1) 与 (3,0) 两处应相同."""
    c = _make_cell()
    c.ugrad = np.array([0.0, 2.0, 5.0])
    c.vgrad = np.array([0.0, -3.0, 7.0])
    c.Tgrad = c.miublgrad = np.zeros(3)
    tb.Spalart_Allmaras(c)
    assert c.DiffuTurb[2][1] == pytest.approx(c.DiffuTurb[3][0], rel=1e-15)


def test_stress_trace_vanishes_for_divergence_free_field():
    """Stokes 假设：τxx + τyy = (2/3)μ_eff·∇·u，无散场下应为 0."""
    c = _make_cell()
    c.ugrad = np.array([0.0, 2.0, 5.0])     # ∂u/∂x = 2
    c.vgrad = np.array([0.0, -3.0, -2.0])   # ∂v/∂y = −2 ⇒ ∇·u = 0
    c.Tgrad = c.miublgrad = np.zeros(3)
    tb.Spalart_Allmaras(c)
    assert c.DiffuTurb[2][0] + c.DiffuTurb[3][1] == pytest.approx(0.0, abs=1e-18)


def test_stress_trace_tracks_dilatation():
    """可压缩情形：τxx + τyy 必须正比于 (2/3)μ_eff·∇·u."""
    c = _make_cell()
    c.ugrad = np.array([0.0, 2.0, 5.0])
    c.vgrad = np.array([0.0, -3.0, 7.0])    # ∇·u = 2 + 7 = 9
    c.Tgrad = c.miublgrad = np.zeros(3)
    tb.Spalart_Allmaras(c)
    mu = cc.mu0 * (c.T / cc.T0) ** 1.5 * (cc.T0 + cc.Ts) / (c.T + cc.Ts)
    mu_eff = mu + c.U[5] * c.fv1
    trace = c.DiffuTurb[2][0] + c.DiffuTurb[3][1]
    assert trace == pytest.approx(2 / 3 * mu_eff * 9.0, rel=1e-12)
