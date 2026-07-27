//! Spalart-Allmaras 一方程湍流模型的源项。
//!
//! ```text
//! S = P − D + G
//! P = Cb1·(1 − ft2)·S̃·ρν̃                        生成
//! D = (Cw1·fw − Cb1/κ²·ft2)·ρ·(ν̃/d)²             壁面破坏
//! G = (Cb2/σ)·ρ·|∇ν̃|²                            非守恒扩散
//! ```
//!
//! 其中 `S̃ = S + ν̃/(κ²d²)·fv2`,`S = |ω|` 是涡量模。
//!
//! 源项只作用在第 5 个方程上,平均流方程无源。

use rayon::iter::ParallelIterator;

use crate::config::Config;
use crate::field::comp;
use crate::geometry::Geometry;
use crate::state::Cells;

/// `S̃` 的下限。Allmaras (2012) 建议对修正涡量做截断,否则 `fv2 < 0` 时
/// `S̃` 可能过零,`r = ν̃/(S̃κ²d²)` 除零发散。
const S_TILDE_FLOOR: f64 = 1e-10;

/// 壁面阻尼函数 `fw = g·[(1+Cw3⁶)/(g⁶+Cw3⁶)]^(1/6)`,`g = r + Cw2(r⁶−r)`。
///
/// `Cw3⁶` 与 `1+Cw3⁶` 由调用方预先算好(它们只依赖配置);`x^(1/6)` 拆成
/// `∛√x` —— `sqrt` 与 `cbrt` 都比通用的 `powf` 快。
#[inline(always)]
pub fn fw(r: f64, cw2: f64, cw3_6: f64, one_plus_cw3_6: f64) -> f64 {
    let g = r + cw2 * (r.powi(6) - r);
    g * (one_plus_cw3_6 / (g.powi(6) + cw3_6)).sqrt().cbrt()
}

/// 二维涡量模 `S = √(2ΩᵢⱼΩᵢⱼ) = |∂v/∂x − ∂u/∂y|`。
///
/// Python 基线取的是 `½(∂u/∂y − ∂v/∂x)` 再乘 √2,既差 √2 倍又反了号
/// (`BUGS.md` B3)。
#[inline(always)]
pub fn vorticity_magnitude(dvdx: f64, dudy: f64) -> f64 {
    (dvdx - dudy).abs()
}

/// 计算全部物理单元的 S-A 源项(已乘单元体积)。
pub fn compute(cfg: &Config, geom: &Geometry, cells: &mut Cells) {
    let sa = &cfg.spalart_allmaras;
    let d = &cfg.derived;
    let nj = geom.nj as isize;
    // 循环不变量全部提到外面
    let cb1_over_kappa2 = sa.Cb1 * d.inv_kappa2;
    let cb2_sigma = sa.Cb2 * sa.sigma;

    let Cells { src, u, aux, grad, .. } = cells;
    let (u, aux, grad) = (&*u, &*aux, &*grad);
    let (vol, inv_d2) = (&geom.vol, &geom.inv_wall_dist_sq);

    src.par_interior_rows_mut().for_each(|(i, mut row)| {
        for j in 0..nj {
            let uc = u.get(i, j);
            let rho = uc[comp::RHO];
            let rho_nu = uc[comp::RHO_NU];
            let nu_tilde = rho_nu / rho;
            let inv_dist2 = inv_d2.get(i, j);
            let inv_kd2 = inv_dist2 * d.inv_kappa2;

            let a = aux.get(i, j);
            let g = grad.get(i, j);
            let ft2 = sa.Ct3 * (-sa.Ct4 * a.chi * a.chi).exp();
            let fv2 = 1.0 - a.chi / (1.0 + a.chi * a.fv1);

            let s = sa.fv3 * vorticity_magnitude(g.dvdx, g.dudy);
            let s_tilde = (s + nu_tilde * inv_kd2 * fv2).max(S_TILDE_FLOOR);

            // 生成项
            let prod = sa.Cb1 * (1.0 - ft2) * s_tilde * rho_nu;
            // 壁面破坏项
            let r = (nu_tilde / s_tilde * inv_kd2).min(sa.rmax);
            let destr = (d.cw1 * fw(r, sa.Cw2, d.cw3_6, d.one_plus_cw3_6)
                - cb1_over_kappa2 * ft2)
                * rho
                * (nu_tilde * nu_tilde * inv_dist2);
            // 非守恒扩散项
            let diff = cb2_sigma * rho * (g.dnutdx * g.dnutdx + g.dnutdy * g.dnutdy);

            row[j] = (prod - destr + diff) * vol.get(i, j);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::Mesh;
    use crate::state::{Domain, Grad, TurbAux};

    fn setup() -> (Config, Domain) {
        let cfg = Config::from_str(include_str!("../../config.json")).unwrap();
        let mesh = Mesh::parse(include_str!("../../fangdata.txt")).unwrap();
        let geom = Geometry::build(&mesh, cfg.simulation.halo);
        let mut dom = Domain::new(geom, cfg.simulation.halo);
        dom.cells.initialize(&cfg);
        crate::boundary::apply(&cfg, &dom.geom, &mut dom.cells);
        crate::gradient::compute(&dom.geom, &mut dom.cells);
        crate::viscous::compute(&cfg, &dom.geom, &mut dom.cells, &mut dom.faces);
        (cfg, dom)
    }

    #[test]
    fn vorticity_is_the_curl() {
        assert!((vorticity_magnitude(11.0, 3.0) - 8.0).abs() < 1e-15);
        assert!((vorticity_magnitude(3.0, 11.0) - 8.0).abs() < 1e-15);
        assert_eq!(vorticity_magnitude(5.0, 5.0), 0.0);
    }

    const CW3_6: f64 = 64.0; // 2⁶
    const ONE_PLUS: f64 = 65.0;

    #[test]
    fn fw_is_bounded_on_the_admissible_range() {
        for k in 0..=200 {
            let r = 10.0 * k as f64 / 200.0;
            let v = fw(r, 0.3, CW3_6, ONE_PLUS);
            assert!(v.is_finite() && (0.0..10.0).contains(&v), "fw({r}) = {v}");
        }
    }

    #[test]
    fn fw_is_unity_at_r_equals_one() {
        // r = 1 ⇒ g = 1 ⇒ fw = ((1+Cw3⁶)/(1+Cw3⁶))^(1/6) = 1
        assert!((fw(1.0, 0.3, CW3_6, ONE_PLUS) - 1.0).abs() < 1e-14);
    }

    /// `∛√x` 必须与 `x^(1/6)` 在数值上一致(性能改写不得改变结果)。
    #[test]
    fn fw_matches_the_powf_formulation() {
        for k in 1..=200 {
            let r = 10.0 * k as f64 / 200.0;
            let g = r + 0.3 * (r.powi(6) - r);
            let want = g * (ONE_PLUS / (g.powi(6) + CW3_6)).powf(1.0 / 6.0);
            let got = fw(r, 0.3, CW3_6, ONE_PLUS);
            assert!((got - want).abs() <= 1e-14 * want.abs(), "fw({r}): {got} vs {want}");
        }
    }

    #[test]
    fn source_is_finite_everywhere() {
        let (cfg, mut dom) = setup();
        compute(&cfg, &dom.geom, &mut dom.cells);
        for (i, j) in dom.cells.src.interior() {
            assert!(dom.cells.src.get(i, j).is_finite(), "source NaN at ({i},{j})");
        }
    }

    /// 零涡量、零 ν̃ 的极限:`S̃` 的下限截断必须防住除零。
    #[test]
    fn source_is_finite_at_zero_vorticity_and_zero_nut() {
        let (cfg, mut dom) = setup();
        for (i, j) in dom.cells.u.interior().collect::<Vec<_>>() {
            let mut u = dom.cells.u.get(i, j);
            u[comp::RHO_NU] = 0.0;
            dom.cells.u.set(i, j, u);
            dom.cells.grad.set(i, j, Grad::default());
            dom.cells.aux.set(i, j, TurbAux { mu: 1.8e-5, chi: 0.0, fv1: 0.0 });
        }
        compute(&cfg, &dom.geom, &mut dom.cells);
        for (i, j) in dom.cells.src.interior() {
            assert!(dom.cells.src.get(i, j).is_finite());
        }
    }

    /// 源项量级随涡量增大 —— 生成项 ∝ S̃,是模型的基本定标行为。
    ///
    /// 这里不能断言源项**变正**:低 χ 时 `ft2 = Ct3·exp(−Ct4χ²) → 1.2 > 1`,
    /// 于是 `Cb1(1−ft2)S̃ρν̃ < 0`(S-A 原始式中的转捩抑制项)。这是标准行为,
    /// 许多实现干脆采用 SA-noft2 变体把它去掉。
    #[test]
    fn source_magnitude_scales_with_vorticity() {
        let (cfg, mut dom) = setup();
        let mut sample = |w: f64| {
            for (i, j) in dom.cells.grad.interior().collect::<Vec<_>>() {
                let mut g = dom.cells.grad.get(i, j);
                g.dvdx = w;
                g.dudy = 0.0;
                dom.cells.grad.set(i, j, g);
            }
            compute(&cfg, &dom.geom, &mut dom.cells);
            dom.cells.src.get(0, 0)
        };
        let lo = sample(1.0);
        let hi = sample(100.0);
        assert!(
            hi.abs() > 10.0 * lo.abs(),
            "source did not scale with vorticity: {lo:e} -> {hi:e}"
        );
        assert!(lo.signum() == hi.signum(), "production changed sign unexpectedly");
    }

    /// 生成项确实正比于 `(1 − ft2)`:把 χ 调大使 ft2 → 0 后,源项应转为正。
    #[test]
    fn production_is_positive_at_high_eddy_viscosity() {
        let (cfg, mut dom) = setup();
        for (i, j) in dom.cells.u.interior().collect::<Vec<_>>() {
            // χ = 20 ⇒ ft2 = 1.2·e^{−200} ≈ 0
            let mu = dom.cells.aux.get(i, j).mu;
            let mut u = dom.cells.u.get(i, j);
            u[comp::RHO_NU] = 20.0 * mu;
            dom.cells.u.set(i, j, u);
            dom.cells.aux.set(
                i,
                j,
                TurbAux {
                    mu,
                    chi: 20.0,
                    fv1: crate::viscous::fv1(20.0, cfg.derived.cv1_cubed),
                },
            );
            let mut g = dom.cells.grad.get(i, j);
            g.dvdx = 500.0;
            g.dudy = 0.0;
            dom.cells.grad.set(i, j, g);
        }
        compute(&cfg, &dom.geom, &mut dom.cells);
        assert!(dom.cells.src.get(0, 0) > 0.0, "expected net production");
    }
}
