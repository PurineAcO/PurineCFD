//! JST 人工粘性(Jameson-Schmidt-Turkel scalar dissipation)。
//!
//! 中心格式本身没有耗散,需要显式加上二阶/四阶人工粘性:
//!
//! ```text
//! D_face = λ_f · ( ε² ·(U₊ − U₋) − ε⁴·(U₊₊ − 3U₊ + 3U₋ − U₋₋) )
//! ```
//!
//! * 四阶项提供背景阻尼,抑制中心格式的奇偶解耦(它对二次以下的分布恒为 0,
//!   因而不影响格式精度);
//! * 二阶项只在激波附近打开,由压力探测器
//!   `ν = |p₋ − 2p₀ + p₊| / (p₋ + 2p₀ + p₊)` 触发;
//! * `ε² = k₂·max(ν)`(面两侧各两个单元),`ε⁴ = max(0, k₄ − ε²)` —— 激波处
//!   四阶项被关掉,避免高阶项在间断附近产生振荡。
//!
//! Python 基线用 `shockwave_tau[k]`(以虚拟单元 `k−2` 为中心)间接表达探测器,
//! 需要一整套偏移换算,`BUGS.md` B6 正是这里索引写重了。这里直接把探测器**定义
//! 在单元上**,四点取最大值写成对称的 `max(ν[i−2..i+1])`,不再有偏移。

use rayon::iter::ParallelIterator;

use crate::config::Config;
use crate::field::{Field, Vec5};
use crate::geometry::Geometry;
use crate::state::{Cells, Eps, Faces};

/// 压力探测器 `ν = |p₋ − 2p₀ + p₊| / (p₋ + 2p₀ + p₊)`。
#[inline(always)]
pub fn pressure_sensor(pm: f64, p0: f64, pp: f64) -> f64 {
    ((pm - 2.0 * p0 + pp) / (pm + 2.0 * p0 + pp)).abs()
}

/// 由四点探测器最大值得到 `(ε², ε⁴)`。
#[inline(always)]
pub fn adaptive_coefficients(nu_max: f64, k2: f64, k4: f64) -> Eps {
    let e2 = k2 * nu_max;
    Eps { e2, e4: (k4 - e2).max(0.0) }
}

/// 面谱半径 λf ——用两侧单元的 `V/Δt_local` 近似,与 CFL 抵消后即为总谱半径。
///
/// `V/Δt` 先在单元上算一遍(`NI·NJ` 次除法),面上只做平均;直接在面循环里除
/// 会做约四倍的除法。
fn spectral_radii(cfg: &Config, geom: &Geometry, cells: &Cells, faces: &mut Faces) {
    let (ni, nj) = (geom.ni as isize, geom.nj as isize);
    let cfl = cfg.simulation.cfl;
    let (vol, dt) = (&geom.vol, &cells.localdt);

    let Faces { tau, nrm, spec_ratio, .. } = faces;
    spec_ratio.par_interior_rows_mut().for_each(|(i, mut row)| {
        for j in 0..nj {
            row[j] = vol.get(i, j) / dt.get(i, j);
        }
    });
    let ratio = &*spec_ratio;

    tau.lambda.par_interior_rows_mut().for_each(|(i, mut row)| {
        // 边界面两侧只有一个物理单元,退化为单侧取值
        let (a, b) = (i.clamp(0, ni - 1), (i - 1).clamp(0, ni - 1));
        for j in 0..nj {
            row[j] = 0.5 * cfl * (ratio.get(a, j) + ratio.get(b, j));
        }
    });
    nrm.lambda.par_interior_rows_mut().for_each(|(i, mut row)| {
        for j in 0..nj {
            let jm = if j > 0 { j - 1 } else { nj - 1 };
            row[j] = 0.5 * cfl * (ratio.get(i, j) + ratio.get(i, jm));
        }
    });
}

/// 单元上的压力探测器,范围覆盖到虚拟层(四点取最大值需要 `[-2, N+1]`)。
fn sensors(geom: &Geometry, cells: &Cells, faces: &mut Faces) {
    let (ni, nj) = (geom.ni as isize, geom.nj as isize);
    let p = &cells.p;
    for i in -2..=ni + 1 {
        for j in 0..nj {
            faces
                .sensor_i
                .set(i, j, pressure_sensor(p.get(i - 1, j), p.get(i, j), p.get(i + 1, j)));
        }
    }
    for i in 0..ni {
        for j in -2..=nj {
            faces
                .sensor_j
                .set(i, j, pressure_sensor(p.get(i, j - 1), p.get(i, j), p.get(i, j + 1)));
        }
    }
}

/// 自适应耗散系数 `ε²`、`ε⁴`。
fn coefficients(cfg: &Config, geom: &Geometry, faces: &mut Faces) {
    let (k2, k4) = (cfg.dissipation.k2, cfg.dissipation.k4);
    let nj = geom.nj as isize;
    let Faces {
        tau,
        nrm,
        sensor_i,
        sensor_j,
        ..
    } = faces;
    let (sensor_i, sensor_j) = (&*sensor_i, &*sensor_j);

    tau.eps.par_interior_rows_mut().for_each(|(i, mut row)| {
        for j in 0..nj {
            // 关于面 i 对称的四点模板:单元 i−2, i−1, i, i+1
            let m = sensor_i
                .get(i - 2, j)
                .max(sensor_i.get(i - 1, j))
                .max(sensor_i.get(i, j))
                .max(sensor_i.get(i + 1, j));
            row[j] = adaptive_coefficients(m, k2, k4);
        }
    });
    nrm.eps.par_interior_rows_mut().for_each(|(i, mut row)| {
        for j in 0..nj {
            let m = sensor_j
                .get(i, j - 2)
                .max(sensor_j.get(i, j - 1))
                .max(sensor_j.get(i, j))
                .max(sensor_j.get(i, j + 1));
            row[j] = adaptive_coefficients(m, k2, k4);
        }
    });
}

/// 面上的 JST 耗散项。
fn face_dissipation(geom: &Geometry, cells: &Cells, faces: &mut Faces) {
    let nj = geom.nj as isize;
    let u = &cells.u;
    let Faces { tau, nrm, .. } = faces;

    {
        let FaceWorkSplit { lambda, eps, out } = FaceWorkSplit {
            lambda: &tau.lambda,
            eps: &tau.eps,
            out: &mut tau.dissipation,
        };
        out.par_interior_rows_mut().for_each(|(i, mut row)| {
            for j in 0..nj {
                let d1 = u.get(i, j) - u.get(i - 1, j);
                let d3 = u.get(i + 1, j) - 3.0 * u.get(i, j) + 3.0 * u.get(i - 1, j)
                    - u.get(i - 2, j);
                let e = eps.get(i, j);
                row[j] = lambda.get(i, j) * (d1 * e.e2 - d3 * e.e4);
            }
        });
    }
    {
        let FaceWorkSplit { lambda, eps, out } = FaceWorkSplit {
            lambda: &nrm.lambda,
            eps: &nrm.eps,
            out: &mut nrm.dissipation,
        };
        out.par_interior_rows_mut().for_each(|(i, mut row)| {
            for j in 0..nj {
                let d1 = u.get(i, j) - u.get(i, j - 1);
                let d3 = u.get(i, j + 1) - 3.0 * u.get(i, j) + 3.0 * u.get(i, j - 1)
                    - u.get(i, j - 2);
                let e = eps.get(i, j);
                row[j] = lambda.get(i, j) * (d1 * e.e2 - d3 * e.e4);
            }
        });
    }
}

/// 把 [`crate::state::FaceWork`] 拆成"读若干 + 写一个"的不相交借用。
struct FaceWorkSplit<'a> {
    lambda: &'a Field<f64>,
    eps: &'a Field<Eps>,
    out: &'a mut Field<Vec5>,
}

/// 单元上的人工粘性环量。
fn assemble(geom: &Geometry, faces: &Faces, out: &mut Field<Vec5>) {
    let nj = geom.nj as isize;
    let (tau, nrm) = (&faces.tau.dissipation, &faces.nrm.dissipation);
    out.par_interior_rows_mut().for_each(|(i, mut row)| {
        for j in 0..nj {
            let jp1 = if j + 1 < nj { j + 1 } else { 0 };
            row[j] = tau.get(i + 1, j) + nrm.get(i, jp1) - tau.get(i, j) - nrm.get(i, j);
        }
    });
}

/// 一次完整的 JST 人工粘性计算。
pub fn compute(cfg: &Config, geom: &Geometry, cells: &mut Cells, faces: &mut Faces) {
    spectral_radii(cfg, geom, cells, faces);
    sensors(geom, cells, faces);
    coefficients(cfg, geom, faces);
    face_dissipation(geom, cells, faces);
    assemble(geom, faces, &mut cells.fd);
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
        crate::timestep::compute(&cfg, &dom.geom, &mut dom.cells);
        (cfg, dom)
    }

    #[test]
    fn sensor_is_zero_in_smooth_flow() {
        assert_eq!(pressure_sensor(2.0, 2.0, 2.0), 0.0);
    }

    #[test]
    fn sensor_detects_a_jump() {
        assert!((pressure_sensor(1.0, 1.0, 3.0) - 2.0 / 6.0).abs() < 1e-15);
    }

    #[test]
    fn sensor_is_bounded_by_one() {
        for &(a, b, c) in &[(1.0, 1e-6, 1.0), (1.0, 1.0, 1e6), (1e-9, 5.0, 1e-9)] {
            let v = pressure_sensor(a, b, c);
            assert!((0.0..=1.0).contains(&v), "sensor {v} out of [0,1]");
        }
    }

    /// 强激波处 ε² 增大、ε⁴ 被关掉。
    #[test]
    fn fourth_order_coefficient_switches_off_at_a_shock() {
        let (k2, k4) = (0.5, 0.0078125);
        let smooth = adaptive_coefficients(0.0, k2, k4);
        assert_eq!(smooth.e2, 0.0);
        assert_eq!(smooth.e4, k4);
        let shock = adaptive_coefficients(0.9, k2, k4);
        assert!((shock.e2 - 0.45).abs() < 1e-18);
        assert_eq!(shock.e4, 0.0);
    }

    #[test]
    fn sensor_fires_across_a_pressure_jump_in_the_field() {
        let (cfg, mut dom) = setup();
        for j in 0..dom.cells.nj as isize {
            dom.cells.p.set(2, j, dom.cells.p.get(2, j) * 4.0);
        }
        sensors(&dom.geom, &dom.cells, &mut dom.faces);
        coefficients(&cfg, &dom.geom, &mut dom.faces);
        let fired = dom
            .faces
            .tau
            .eps
            .interior()
            .any(|(i, j)| dom.faces.tau.eps.get(i, j).e2 > cfg.dissipation.k4);
        assert!(fired, "expected the sensor to fire near the jump");
    }

    #[test]
    fn uniform_flow_gets_no_dissipation() {
        let (cfg, mut dom) = setup();
        dom.cells.set_uniform(&cfg, 1.176, 69.4, 17.3, 101325.0, 1.5e-4);
        compute(&cfg, &dom.geom, &mut dom.cells, &mut dom.faces);
        let lam = dom
            .faces
            .tau
            .lambda
            .interior()
            .map(|(i, j)| dom.faces.tau.lambda.get(i, j))
            .fold(0.0f64, f64::max);
        let umax = dom
            .cells
            .u
            .interior()
            .map(|(i, j)| dom.cells.u.get(i, j).amax())
            .fold(0.0f64, f64::max);
        let scale = lam * cfg.dissipation.k4 * umax;
        for (i, j) in dom.cells.fd.interior() {
            assert!(
                dom.cells.fd.get(i, j).amax() < 1e-13 * scale,
                "Fd nonzero at ({i},{j})"
            );
        }
    }

    /// 光滑流场里二阶(激波)耗散应当关闭,只剩四阶背景耗散。
    #[test]
    fn smooth_flow_leaves_only_background_dissipation() {
        let (cfg, mut dom) = setup();
        compute(&cfg, &dom.geom, &mut dom.cells, &mut dom.faces);
        for (i, j) in dom.faces.tau.eps.interior() {
            let e = dom.faces.tau.eps.get(i, j);
            // 远场边界压力由黎曼求解给出,与 p∞ 只到 ~1e-11 相对精度,
            // 因此探测器有一个极小的本底,而非精确 0
            assert!(e.e2 < 1e-12 * cfg.dissipation.k4, "eps2 = {:e} at ({i},{j})", e.e2);
            assert!((e.e4 - cfg.dissipation.k4).abs() < 1e-12 * cfg.dissipation.k4);
        }
    }

    /// 四阶项对二次以下分布恒为 0 —— 它不破坏格式精度的关键。
    #[test]
    fn fourth_difference_annihilates_quadratics() {
        let q = |x: f64| 3.0 * x * x + 2.0 * x + 5.0;
        let d3 = q(1.0) - 3.0 * q(0.0) + 3.0 * q(-1.0) - q(-2.0);
        assert!(d3.abs() < 1e-12, "d3 = {d3}");
    }

    /// 四阶项对三次分布给出常数(即它确实是三阶差分算子)。
    #[test]
    fn fourth_difference_of_a_cubic_is_constant() {
        let c = |x: f64| 2.0 * x * x * x;
        let d3 = c(1.0) - 3.0 * c(0.0) + 3.0 * c(-1.0) - c(-2.0);
        assert!((d3 - 2.0 * 6.0).abs() < 1e-12, "d3 = {d3}");
    }
}
