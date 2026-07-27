//! 无粘对流通量。
//!
//! 三步:面上的守恒量(相邻单元一阶中心平均)→ 面上的 Euler 通量 `F·n` →
//! 单元环量。因为虚拟单元已由 [`crate::boundary`] 填好,这里三个循环都是
//! 纯矩形、零特判。
//!
//! 单元环量的符号约定与 [`crate::geometry`] 的法向定义一致(法向指向 i、j 增大
//! 的方向),于是
//!
//! ```text
//! Fc(i,j) = F_τ(i+1) − F_τ(i) + F_n(j+1) − F_n(j)
//! ```
//!
//! 由度量闭合 `Σ±n ≡ 0`,均匀流下 `Fc ≡ 0`(自由来流保持性)。

use rayon::iter::ParallelIterator;

use crate::config::Config;
use crate::field::{comp, Field, Vec5};
use crate::geometry::{FaceGeom, Geometry};
use crate::state::{Cells, Faces};

/// 由面上的守恒量算 Euler 通量 `F·n`(法向已含面积权)。
#[inline(always)]
pub fn euler_flux(fu: Vec5, face: &FaceGeom, gamma: f64) -> Vec5 {
    let rho = fu[comp::RHO];
    let inv_rho = 1.0 / rho;
    let u = fu[comp::MX] * inv_rho;
    let v = fu[comp::MY] * inv_rho;
    let rho_e = fu[comp::RHO_E];
    let p = (gamma - 1.0) * (rho_e - rho * (u * u + v * v) * 0.5);
    let vn = face.nx * u + face.ny * v;
    Vec5::new(
        rho * vn,
        fu[comp::MX] * vn + p * face.nx,
        fu[comp::MY] * vn + p * face.ny,
        (rho_e + p) * vn,
        fu[comp::RHO_NU] * vn,
    )
}

/// 面上的无粘通量,tau 面与 n 面各一遍。
///
/// 面上的守恒量只是中间值,用局部变量算掉即可 —— 不必像 Python 那样再存一个
/// 全场数组(省一次分配和一遍访存)。
pub fn face_fluxes(cfg: &Config, geom: &Geometry, cells: &Cells, faces: &mut Faces) {
    let gamma = cfg.physics.gamma;
    let u = &cells.u;
    let nj = u.nj() as isize;

    // tau 面 (i, j) 分隔单元 (i−1, j) 与 (i, j),i ∈ [0, NI]
    let tau_geom = &geom.tau;
    faces.tau.flux.par_interior_rows_mut().for_each(|(i, mut row)| {
        for j in 0..nj {
            let fu = 0.5 * (u.get(i - 1, j) + u.get(i, j));
            row[j] = euler_flux(fu, tau_geom.at(i, j), gamma);
        }
    });

    // n 面 (i, j) 分隔单元 (i, j−1) 与 (i, j),j ∈ [0, NJ)
    let n_geom = &geom.nrm;
    faces.nrm.flux.par_interior_rows_mut().for_each(|(i, mut row)| {
        for j in 0..nj {
            let fu = 0.5 * (u.get(i, j - 1) + u.get(i, j));
            row[j] = euler_flux(fu, n_geom.at(i, j), gamma);
        }
    });
}

/// 单元上的对流环量。
pub fn assemble(geom: &Geometry, faces: &Faces, out: &mut Field<Vec5>) {
    let (tau, nrm) = (&faces.tau.flux, &faces.nrm.flux);
    let nj = geom.nj as isize;
    out.par_interior_rows_mut().for_each(|(i, mut row)| {
        for j in 0..nj {
            let jp1 = if j + 1 < nj { j + 1 } else { 0 };
            row[j] = tau.get(i + 1, j) - tau.get(i, j) + nrm.get(i, jp1) - nrm.get(i, j);
        }
    });
}

/// 一次完整的对流项计算。
pub fn compute(cfg: &Config, geom: &Geometry, cells: &mut Cells, faces: &mut Faces) {
    face_fluxes(cfg, geom, cells, faces);
    assemble(geom, faces, &mut cells.fc);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::Mesh;
    use crate::state::Domain;

    fn setup() -> (Config, Domain) {
        let cfg = Config::from_str(include_str!("../../config.json")).unwrap();
        let mesh = Mesh::parse(include_str!("../../fangdata.txt")).unwrap();
        let geom = Geometry::build(&mesh, cfg.simulation.halo);
        let mut dom = Domain::new(geom, cfg.simulation.halo);
        dom.cells.initialize(&cfg);
        crate::boundary::apply(&cfg, &dom.geom, &mut dom.cells);
        (cfg, dom)
    }

    #[test]
    fn flux_matches_analytic_euler_flux() {
        let (rho, u, v, p, nut) = (1.15, 60.0, -20.0, 9.5e4, 3e-4);
        let gamma = 1.4;
        let e = p / (rho * (gamma - 1.0)) + 0.5 * (u * u + v * v);
        let fu = Vec5::new(rho, rho * u, rho * v, rho * e, rho * nut);
        let g = FaceGeom { nx: 0.3, ny: -0.7, mx: 0.0, my: 0.0 };
        let f = euler_flux(fu, &g, gamma);
        let vn = u * g.nx + v * g.ny;
        assert!((f[0] - rho * vn).abs() < 1e-12);
        assert!((f[1] - (rho * u * vn + p * g.nx)).abs() < 1e-8);
        assert!((f[2] - (rho * v * vn + p * g.ny)).abs() < 1e-8);
        assert!((f[3] - (rho * e + p) * vn).abs() < 1e-4);
        assert!((f[4] - rho * nut * vn).abs() < 1e-16);
    }

    #[test]
    fn flux_is_linear_in_the_normal() {
        let fu = Vec5::new(1.2, 72.0, -18.0, 2.6e5, 3.6e-4);
        let g1 = FaceGeom { nx: 0.3, ny: -0.7, mx: 0.0, my: 0.0 };
        let g2 = FaceGeom { nx: 0.6, ny: -1.4, mx: 0.0, my: 0.0 };
        let (a, b) = (euler_flux(fu, &g1, 1.4), euler_flux(fu, &g2, 1.4));
        for k in 0..5 {
            assert!((b[k] - 2.0 * a[k]).abs() <= 1e-9 * b[k].abs().max(1.0));
        }
    }

    /// 自由来流保持性:均匀场下平均流的对流残差应为机器精度。
    ///
    /// 必须**绕过**边界条件铺一个含虚拟层的均匀场:固壁镜像会让贴壁处不再均匀
    /// (物理上正确,但那检验的是别的东西)。
    #[test]
    fn free_stream_is_preserved() {
        let (cfg, mut dom) = setup();
        dom.cells.set_uniform(&cfg, 1.176, 69.4, 17.3, 101325.0, 1.5e-4);
        compute(&cfg, &dom.geom, &mut dom.cells, &mut dom.faces);
        let scale = dom
            .faces
            .tau
            .flux
            .interior()
            .map(|(i, j)| dom.faces.tau.flux.get(i, j).amax())
            .fold(0.0f64, f64::max);
        for (i, j) in dom.cells.fc.interior() {
            let fc = dom.cells.fc.get(i, j);
            for k in 0..5 {
                assert!(
                    fc[k].abs() < 1e-12 * scale,
                    "Fc[{k}] = {:e} at ({i},{j}), scale {scale:e}",
                    fc[k]
                );
            }
        }
    }

    /// 镜像壁面 ⇒ 壁面上的法向质量通量为 0。
    #[test]
    fn no_mass_flux_through_the_wall() {
        let (cfg, mut dom) = setup();
        compute(&cfg, &dom.geom, &mut dom.cells, &mut dom.faces);
        for j in 0..dom.cells.nj as isize {
            let m = dom.faces.tau.flux.get(0, j)[comp::RHO];
            assert!(m.abs() < 1e-9, "mass leaks through wall at j={j}: {m:e}");
        }
    }

    /// 内部面通量在单元环量中两两抵消:全场 Σ Fc 只剩边界贡献。
    #[test]
    fn interior_fluxes_telescope() {
        let (cfg, mut dom) = setup();
        // 造一个非均匀场,确保抵消不是因为通量恒等
        for (i, j) in dom.cells.rho.interior().collect::<Vec<_>>() {
            let s = 1.0 + 0.03 * i as f64 + 0.01 * j as f64;
            dom.cells.rho.set(i, j, cfg.derived.rho_inf * s);
            dom.cells.pack(i, j);
        }
        crate::boundary::apply(&cfg, &dom.geom, &mut dom.cells);
        compute(&cfg, &dom.geom, &mut dom.cells, &mut dom.faces);

        let total: f64 = dom
            .cells
            .fc
            .interior()
            .map(|(i, j)| dom.cells.fc.get(i, j)[comp::RHO])
            .sum();
        // 周向面完全抵消(周期),径向只剩壁面与远场
        let ni = dom.cells.ni as isize;
        let boundary: f64 = (0..dom.cells.nj as isize)
            .map(|j| {
                dom.faces.tau.flux.get(ni, j)[comp::RHO] - dom.faces.tau.flux.get(0, j)[comp::RHO]
            })
            .sum();
        assert!(
            (total - boundary).abs() < 1e-9 * boundary.abs().max(1.0),
            "telescoping failed: {total:e} vs {boundary:e}"
        );
    }
}
