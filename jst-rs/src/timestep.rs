//! 当地时间步与全局时间步。
//!
//! 用面平均法向估计单元的两个方向谱半径:
//!
//! ```text
//! Δt_local = CFL·V / ( |u·A + v·B| + |u·C + v·D| + c·(|AB| + |CD|) )
//! ```
//!
//! `(A,B)` 是单元两条 tau 面法向的平均、`(C,D)` 是两条 n 面法向的平均。
//!
//! 全局推进取所有单元的最小值(定常问题的稳态解与 Δt 无关,统一步长可保证
//! 时间精度一致)。求最小值与顺序无关,因此并行归约的结果逐位可复现。
//!
//! 注意 `localdt` 保留**逐单元**的值:[`crate::dissipation`] 用 `V/Δt_local`
//! 近似面谱半径,那里需要的是局部量而非全局最小值。

use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::config::Config;
use crate::geometry::Geometry;
use crate::state::Cells;

/// 计算各单元 `localdt`,返回全局最小值。
pub fn compute(cfg: &Config, geom: &Geometry, cells: &mut Cells) -> f64 {
    let nj = geom.nj as isize;
    let cfl = cfg.simulation.cfl;
    let (tau, nrm, vol) = (&geom.tau, &geom.nrm, &geom.vol);
    let (vx, vy, c) = (&cells.vx, &cells.vy, &cells.c);

    cells.localdt.par_interior_rows_mut().for_each(|(i, mut row)| {
        for j in 0..nj {
            let jp1 = if j + 1 < nj { j + 1 } else { 0 };
            let (f0, f1) = (tau.at(i, j), tau.at(i + 1, j));
            let (g0, g1) = (nrm.at(i, j), nrm.at(i, jp1));
            let a = 0.5 * (f0.nx + f1.nx);
            let b = 0.5 * (f0.ny + f1.ny);
            let cc = 0.5 * (g0.nx + g1.nx);
            let dd = 0.5 * (g0.ny + g1.ny);
            let (u, v) = (vx.get(i, j), vy.get(i, j));
            let conv = (u * a + v * b).abs() + (u * cc + v * dd).abs();
            let acou = c.get(i, j) * ((a * a + b * b).sqrt() + (cc * cc + dd * dd).sqrt());
            row[j] = cfl * vol.get(i, j) / (conv + acou);
        }
    });

    global_min(cells)
}

/// 并行求全局最小 `localdt`。`min` 满足结合律与交换律,结果与线程数无关。
fn global_min(cells: &Cells) -> f64 {
    let nj = cells.nj as isize;
    let dt = &cells.localdt;
    (0..cells.ni as isize)
        .into_par_iter()
        .map(|i| (0..nj).map(|j| dt.get(i, j)).fold(f64::INFINITY, f64::min))
        .reduce(|| f64::INFINITY, f64::min)
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
    fn timestep_is_positive_and_finite() {
        let (cfg, mut dom) = setup();
        let dt = compute(&cfg, &dom.geom, &mut dom.cells);
        assert!(dt > 0.0 && dt.is_finite());
    }

    #[test]
    fn returned_step_is_the_minimum_of_the_local_steps() {
        let (cfg, mut dom) = setup();
        let dt = compute(&cfg, &dom.geom, &mut dom.cells);
        let mut lo = f64::INFINITY;
        for (i, j) in dom.cells.localdt.interior() {
            let l = dom.cells.localdt.get(i, j);
            assert!(l >= dt, "local step below the global minimum at ({i},{j})");
            lo = lo.min(l);
        }
        assert_eq!(lo, dt);
    }

    /// `localdt` 必须保留逐单元的值 —— JST 谱半径依赖它。
    #[test]
    fn local_steps_are_not_flattened_to_the_global_minimum() {
        let (cfg, mut dom) = setup();
        compute(&cfg, &dom.geom, &mut dom.cells);
        let first = dom.cells.localdt.get(0, 0);
        assert!(
            dom.cells
                .localdt
                .interior()
                .any(|(i, j)| (dom.cells.localdt.get(i, j) - first).abs() > 1e-18),
            "all local steps identical — localdt was overwritten"
        );
    }

    #[test]
    fn timestep_scales_linearly_with_cfl() {
        let (mut cfg, mut dom) = setup();
        let dt1 = compute(&cfg, &dom.geom, &mut dom.cells);
        cfg.simulation.cfl *= 0.5;
        let dt2 = compute(&cfg, &dom.geom, &mut dom.cells);
        assert!((dt2 - 0.5 * dt1).abs() < 1e-15 * dt1);
    }

    #[test]
    fn refining_the_mesh_shrinks_the_timestep() {
        // CFL 条件 Δt ~ h/(|u|+c):网格加密一倍,步长应大致减半
        let cfg = Config::from_str(include_str!("../../config.json")).unwrap();
        let dt_of = |nj: usize| {
            let mut txt = format!("{} {}\n", 9, nj);
            for i in 0..9 {
                let rad = 1.0 + 0.5 * i as f64;
                for j in 0..nj {
                    let a = 2.0 * std::f64::consts::PI * j as f64 / nj as f64;
                    txt += &format!("{:.12} {:.12}\n", rad * a.cos(), rad * a.sin());
                }
            }
            let mesh = Mesh::parse(&txt).unwrap();
            let geom = Geometry::build(&mesh, cfg.simulation.halo);
            let mut dom = Domain::new(geom, cfg.simulation.halo);
            dom.cells.initialize(&cfg);
            crate::boundary::apply(&cfg, &dom.geom, &mut dom.cells);
            compute(&cfg, &dom.geom, &mut dom.cells)
        };
        let (a, b) = (dt_of(32), dt_of(64));
        assert!(b < a, "timestep did not shrink: {a:e} → {b:e}");
        assert!(b > 0.25 * a, "timestep shrank far more than expected");
    }

    #[test]
    fn parallel_reduction_is_deterministic() {
        let (cfg, mut dom) = setup();
        let a = compute(&cfg, &dom.geom, &mut dom.cells);
        for _ in 0..8 {
            assert_eq!(compute(&cfg, &dom.geom, &mut dom.cells), a);
        }
    }
}
