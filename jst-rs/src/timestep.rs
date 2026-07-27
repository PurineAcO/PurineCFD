//! 褰撳湴鏃堕棿姝ヤ笌鍏ㄥ眬鏃堕棿姝ャ€?//!
//! 鐢ㄩ潰骞冲潎娉曞悜浼拌鍗曞厓鐨勪袱涓柟鍚戣氨鍗婂緞:
//!
//! ```text
//! 螖t_local = CFL路V / ( |u路A + v路B| + |u路C + v路D| + c路(|AB| + |CD|) )
//! ```
//!
//! `(A,B)` 鏄崟鍏冧袱鏉?tau 闈㈡硶鍚戠殑骞冲潎銆乣(C,D)` 鏄袱鏉?n 闈㈡硶鍚戠殑骞冲潎銆?//!
//! 鍏ㄥ眬鎺ㄨ繘鍙栨墍鏈夊崟鍏冪殑鏈€灏忓€?瀹氬父闂鐨勭ǔ鎬佽В涓?螖t 鏃犲叧,缁熶竴姝ラ暱鍙繚璇?//! 鏃堕棿绮惧害涓€鑷?銆傛眰鏈€灏忓€间笌椤哄簭鏃犲叧,鍥犳骞惰褰掔害鐨勭粨鏋滈€愪綅鍙鐜般€?//!
//! 娉ㄦ剰 `localdt` 淇濈暀**閫愬崟鍏?*鐨勫€?[`crate::dissipation`] 鐢?`V/螖t_local`
//! 杩戜技闈㈣氨鍗婂緞,閭ｉ噷闇€瑕佺殑鏄眬閮ㄩ噺鑰岄潪鍏ㄥ眬鏈€灏忓€笺€?
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::config::Config;
use crate::geometry::Geometry;
use crate::state::Cells;

/// 璁＄畻鍚勫崟鍏?`localdt`,杩斿洖鍏ㄥ眬鏈€灏忓€笺€?pub fn compute(cfg: &Config, geom: &Geometry, cells: &mut Cells) -> f64 {
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

/// 骞惰姹傚叏灞€鏈€灏?`localdt`銆俙min` 婊¤冻缁撳悎寰嬩笌浜ゆ崲寰?缁撴灉涓庣嚎绋嬫暟鏃犲叧銆?fn global_min(cells: &Cells) -> f64 {
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
        let cfg = Config::from_str(include_str!("../config.json")).unwrap();
        let mesh = Mesh::parse(include_str!("../fangdata.txt")).unwrap();
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

    /// `localdt` 蹇呴』淇濈暀閫愬崟鍏冪殑鍊?鈥斺€?JST 璋卞崐寰勪緷璧栧畠銆?    #[test]
    fn local_steps_are_not_flattened_to_the_global_minimum() {
        let (cfg, mut dom) = setup();
        compute(&cfg, &dom.geom, &mut dom.cells);
        let first = dom.cells.localdt.get(0, 0);
        assert!(
            dom.cells
                .localdt
                .interior()
                .any(|(i, j)| (dom.cells.localdt.get(i, j) - first).abs() > 1e-18),
            "all local steps identical 鈥?localdt was overwritten"
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
        // CFL 鏉′欢 螖t ~ h/(|u|+c):缃戞牸鍔犲瘑涓€鍊?姝ラ暱搴斿ぇ鑷村噺鍗?        let cfg = Config::from_str(include_str!("../config.json")).unwrap();
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
        assert!(b < a, "timestep did not shrink: {a:e} 鈫?{b:e}");
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

