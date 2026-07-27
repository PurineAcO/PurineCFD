//! 与 Python 基线的交叉验证。
//!
//! 读的是 `tests/golden/*.json` —— 由 `tools/dump_golden.py` 从 Python 实现导出,
//! Python 侧的 `tests/test_regression.py` 也读同一份文件。两套实现对着同一组
//! 参考数据比对,任何一侧改动格式都会被立刻发现。
//!
//! 比对是**分级**的:几何 → 初始场 → 单个 RK 级的四个残差分项 → N 步后的全场。
//! 一旦对不上,从哪一级开始出现偏差就直接指出了是哪个 kernel 的问题,而不是
//! 只知道"最终结果不同"。
//!
//! 容差按阶段收紧/放松:
//!
//! | 阶段 | 相对容差 | 理由 |
//! |------|----------|------|
//! | 几何 | 1e-14 | 纯确定性算术,只有极少量重结合差异 |
//! | 初始场 | 1e-14 | 同上 |
//! | 残差分项 | 1e-10 | Green-Gauss 里 numpy `dot` 与手写求和的求和次序不同 |
//! | N 步后 | 1e-8 | 时间推进会放大舍入差异 |
//!
//! 真正的算法性差异(索引错位、模板写反、系数写错)会带来 O(1) 的相对偏差,
//! 与上述任何一档容差都相差十个数量级以上。

use std::path::{Path, PathBuf};

use jst::{config::Config, mesh::Mesh, solver::Solver, timestep};
use serde_json::Value;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf()
}

fn golden_files() -> Vec<PathBuf> {
    let dir = repo_root().join("tests/golden");
    let mut v: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    v.sort();
    assert!(!v.is_empty(), "no golden data in {}", dir.display());
    v
}

fn arr(doc: &Value, section: &str, key: &str) -> Vec<f64> {
    doc[section][key]
        .as_array()
        .unwrap_or_else(|| panic!("missing {section}.{key}"))
        .iter()
        .map(|v| v.as_f64().expect("non-numeric golden entry"))
        .collect()
}

/// 按**场**比较而非逐元素比较:容差取 `rtol · ‖want‖∞ + atol`。
///
/// 逐元素相对容差在这里是错的 —— 场里总会有个别单元的值比场的量级低若干个
/// 数量级(例如 AOA=0 时对称面上的 v ≈ 1e-13),对它们要求相对精度等于在比较
/// 两侧各自的舍入噪声,必然失败且毫无信息量。以场的范数归一才是有意义的判据:
/// "误差相对于这个场的量级不超过 rtol"。
#[track_caller]
fn cmp(got: &[f64], want: &[f64], label: &str, rtol: f64, atol: f64) {
    assert_eq!(got.len(), want.len(), "{label}: length {} != {}", got.len(), want.len());
    let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let tol = atol + rtol * scale;
    let mut worst = (0usize, 0.0f64);
    for (k, (&g, &w)) in got.iter().zip(want).enumerate() {
        let err = (g - w).abs();
        if err > worst.1 {
            worst = (k, err);
        }
    }
    let (k, err) = worst;
    assert!(
        err <= tol,
        "{label}: mismatch at flat index {k}: got {:e}, want {:e} \
         (abs err {err:e} > tol {tol:e}; field scale {scale:e}, rtol {rtol:e})",
        got[k],
        want[k],
    );
}

fn resolve_mesh(name: &str) -> PathBuf {
    let root = repo_root();
    for cand in [root.join(name), root.join("meshes").join(name)] {
        if cand.exists() {
            return cand;
        }
    }
    panic!("mesh {name} not found next to the repository root");
}

fn load(path: &Path) -> (Value, Solver) {
    let doc: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let cfg = Config::from_path(repo_root().join("config.json")).unwrap();
    let mesh = Mesh::from_path(resolve_mesh(doc["meta"]["mesh"].as_str().unwrap())).unwrap();
    let solver = Solver::new(cfg, &mesh);
    (doc, solver)
}

/// 把物理单元的某个标量场按行主序展平(与 Python 的导出顺序一致)。
fn flat(f: &jst::Field<f64>) -> Vec<f64> {
    f.to_interior_vec()
}

fn flat_vec5(f: &jst::Field<jst::Vec5>, k: usize) -> Vec<f64> {
    f.interior().map(|(i, j)| f.get(i, j)[k]).collect()
}

#[test]
fn meta_agrees() {
    for path in golden_files() {
        let (doc, s) = load(&path);
        let m = &doc["meta"];
        let tag = path.file_stem().unwrap().to_string_lossy();
        assert_eq!(s.ni() as u64 + 1, m["i_total"].as_u64().unwrap(), "{tag}: i_total");
        assert_eq!(s.nj() as u64, m["j_total"].as_u64().unwrap(), "{tag}: j_total");
        assert_eq!(s.n_cells() as u64, m["n_cells"].as_u64().unwrap(), "{tag}: n_cells");
        for (key, val) in [
            ("gamma", s.cfg.physics.gamma),
            ("R", s.cfg.physics.r_gas),
            ("CFL", s.cfg.simulation.cfl),
            ("Ma", s.cfg.simulation.mach),
            ("k2", s.cfg.dissipation.k2),
            ("k4", s.cfg.dissipation.k4),
        ] {
            assert!(
                (m[key].as_f64().unwrap() - val).abs() < 1e-15,
                "{tag}: config drift on {key}"
            );
        }
    }
}

#[test]
fn geometry_agrees() {
    for path in golden_files() {
        let (doc, s) = load(&path);
        let tag = path.file_stem().unwrap().to_string_lossy();
        let g = &s.dom.geom;

        cmp(&flat(&g.vol), &arr(&doc, "geometry", "cell_vol"), &format!("{tag}/vol"), 1e-14, 0.0);
        cmp(&flat(&g.cx), &arr(&doc, "geometry", "cell_x"), &format!("{tag}/cx"), 1e-13, 1e-15);
        cmp(&flat(&g.cy), &arr(&doc, "geometry", "cell_y"), &format!("{tag}/cy"), 1e-13, 1e-15);
        cmp(
            &flat(&g.wall_dist),
            &arr(&doc, "geometry", "cell_sad"),
            &format!("{tag}/wall_dist"),
            1e-14,
            0.0,
        );

        // 面几何:tau 面 (NI+1)xNJ,n 面 NIxNJ
        let tau: Vec<_> = (0..=g.ni as isize)
            .flat_map(|i| (0..g.nj as isize).map(move |j| (i, j)))
            .collect();
        let nrm: Vec<_> = (0..g.ni as isize)
            .flat_map(|i| (0..g.nj as isize).map(move |j| (i, j)))
            .collect();
        for (key, idx, sel) in [
            ("tau_nx", &tau, 0usize),
            ("tau_ny", &tau, 1),
            ("tau_mx", &tau, 2),
            ("tau_my", &tau, 3),
        ] {
            let got: Vec<f64> = idx
                .iter()
                .map(|&(i, j)| {
                    let f = g.tau.get(i, j);
                    [f.nx, f.ny, f.mx, f.my][sel]
                })
                .collect();
            cmp(&got, &arr(&doc, "geometry", key), &format!("{tag}/{key}"), 1e-13, 1e-15);
        }
        for (key, sel) in [("n_nx", 0usize), ("n_ny", 1), ("n_mx", 2), ("n_my", 3)] {
            let got: Vec<f64> = nrm
                .iter()
                .map(|&(i, j)| {
                    let f = g.nrm.get(i, j);
                    [f.nx, f.ny, f.mx, f.my][sel]
                })
                .collect();
            cmp(&got, &arr(&doc, "geometry", key), &format!("{tag}/{key}"), 1e-13, 1e-15);
        }
    }
}

#[test]
fn initial_state_agrees() {
    for path in golden_files() {
        let (doc, s) = load(&path);
        let tag = path.file_stem().unwrap().to_string_lossy();
        let c = &s.dom.cells;
        for (key, f) in [
            ("rho", &c.rho),
            ("p", &c.p),
            ("T", &c.t),
            ("u", &c.vx),
            ("v", &c.vy),
            ("E", &c.e),
            ("H", &c.h),
            ("c", &c.c),
            ("miubl", &c.nut),
        ] {
            cmp(&flat(f), &arr(&doc, "init", key), &format!("{tag}/init.{key}"), 1e-14, 1e-300);
        }
        for k in 0..5 {
            cmp(
                &flat_vec5(&c.u, k),
                &arr(&doc, "init", &format!("U{}", k + 1)),
                &format!("{tag}/init.U{}", k + 1),
                1e-14,
                1e-300,
            );
        }
    }
}

/// 单个 RK 级的四个残差分项 —— 定位 kernel 级差异的关键用例。
///
/// 先按 golden 记录的 `warmup` 步推进,让流场脱离均匀初始态:否则梯度、粘性项、
/// 源项在两侧都只是零的舍入噪声,比对不出任何信息。
#[test]
fn single_stage_residual_terms_agree() {
    for path in golden_files() {
        let (doc, mut s) = load(&path);
        let tag = path.file_stem().unwrap().to_string_lossy();
        let warmup = doc["meta"]["warmup"].as_u64().unwrap() as usize;

        s.run(Some(warmup), |_, _| {}).unwrap();
        let dt = timestep::compute(&s.cfg, &s.dom.geom, &mut s.dom.cells);
        s.residual_terms();

        let want_dt = doc["terms"]["dt"].as_f64().unwrap();
        assert!(
            (dt - want_dt).abs() < 1e-12 * want_dt,
            "{tag}: dt {dt:e} != {want_dt:e}"
        );

        let c = &s.dom.cells;
        type GradSel = fn(&jst::state::Grad) -> f64;
        type AuxSel = fn(&jst::state::TurbAux) -> f64;
        let grads: [(&str, GradSel); 8] = [
            ("ugrad_x", |g| g.dudx),
            ("ugrad_y", |g| g.dudy),
            ("vgrad_x", |g| g.dvdx),
            ("vgrad_y", |g| g.dvdy),
            ("Tgrad_x", |g| g.dtdx),
            ("Tgrad_y", |g| g.dtdy),
            ("miublgrad_x", |g| g.dnutdx),
            ("miublgrad_y", |g| g.dnutdy),
        ];
        for (key, sel) in grads {
            let got: Vec<f64> = c.grad.interior().map(|(i, j)| sel(&c.grad.get(i, j))).collect();
            cmp(&got, &arr(&doc, "terms", key), &format!("{tag}/terms.{key}"), 1e-10, 1e-280);
        }
        let auxes: [(&str, AuxSel); 2] = [("chi", |a| a.chi), ("fv1", |a| a.fv1)];
        for (key, sel) in auxes {
            let got: Vec<f64> = c.aux.interior().map(|(i, j)| sel(&c.aux.get(i, j))).collect();
            cmp(&got, &arr(&doc, "terms", key), &format!("{tag}/terms.{key}"), 1e-10, 1e-280);
        }
        for k in 0..5 {
            for (term, f) in [("Fc", &c.fc), ("Fv", &c.fv), ("Fd", &c.fd)] {
                let key = format!("{term}{}", k + 1);
                cmp(
                    &flat_vec5(f, k),
                    &arr(&doc, "terms", &key),
                    &format!("{tag}/terms.{key}"),
                    1e-10,
                    1e-280,
                );
            }
        }
        // S-A 源项只在第 5 个方程上
        cmp(&flat(&c.src), &arr(&doc, "terms", "S5"), &format!("{tag}/terms.S5"), 1e-10, 1e-280);
    }
}

#[test]
fn final_state_agrees() {
    for path in golden_files() {
        let (doc, mut s) = load(&path);
        let tag = path.file_stem().unwrap().to_string_lossy();
        let steps = doc["meta"]["steps"].as_u64().unwrap() as usize;

        let report = s.run(Some(steps), |_, _| {}).unwrap();

        let f = &doc["final"];
        let want_res = f["residual"].as_f64().unwrap();
        assert!(
            (report.residual - want_res).abs() < 1e-8 * want_res,
            "{tag}: residual {:e} != {want_res:e}",
            report.residual
        );
        let want_t = f["totaltime"].as_f64().unwrap();
        assert!(
            (report.totaltime - want_t).abs() < 1e-10 * want_t,
            "{tag}: totaltime {:e} != {want_t:e}",
            report.totaltime
        );

        let c = &s.dom.cells;
        for (key, fld) in [
            ("rho", &c.rho),
            ("p", &c.p),
            ("T", &c.t),
            ("u", &c.vx),
            ("v", &c.vy),
            ("miubl", &c.nut),
        ] {
            cmp(&flat(fld), &arr(&doc, "final", key), &format!("{tag}/final.{key}"), 1e-8, 1e-300);
        }
    }
}
