//! 集成层面的性质检验:并行可复现性、守恒性、自由来流保持性、稳健性。
//!
//! 与 `src/**` 里的单元测试不同,这里检验的是**跨模块**才成立的性质 ——
//! 单个 kernel 正确并不蕴含整条链路正确。

use jst::field::comp;
use jst::{config::Config, mesh::Mesh, solver::Solver};

fn config() -> Config {
    Config::from_path(concat!(env!("CARGO_MANIFEST_DIR"), "/../config.json")).unwrap()
}

/// 与 `tools/genmesh.py` 同族的椭圆柱 → 远场圆 O 型网格。
fn synth_mesh(rings: usize, nj: usize, stretch: f64) -> Mesh {
    let (a_wall, b_wall, r_far) = (1.0, 0.5, 5.0);
    let svals: Vec<f64> = if (stretch - 1.0).abs() < 1e-12 {
        (0..rings).map(|i| i as f64 / (rings - 1) as f64).collect()
    } else {
        let w: Vec<f64> = (0..rings - 1).map(|k| stretch.powi(k as i32)).collect();
        let total: f64 = w.iter().sum();
        let mut acc = 0.0;
        std::iter::once(0.0)
            .chain(w.iter().map(|x| {
                acc += x;
                acc / total
            }))
            .collect()
    };
    let mut txt = format!("{rings} {nj}\n");
    for s in svals {
        let (a, b) = (a_wall + s * (r_far - a_wall), b_wall + s * (r_far - b_wall));
        for j in 0..nj {
            let th = 2.0 * std::f64::consts::PI * j as f64 / nj as f64;
            txt += &format!("{:.12} {:.12}\n", a * th.cos(), b * th.sin());
        }
    }
    Mesh::parse(&txt).unwrap()
}

fn solver(rings: usize, nj: usize, stretch: f64) -> Solver {
    Solver::new(config(), &synth_mesh(rings, nj, stretch))
}

/// **并行不得改变结果**。
///
/// 每个并行 kernel 都满足"每个输出元素只写一次、取值只由输入决定",因此结果
/// 与线程数无关;全局归约用的 `min` 也与顺序无关,残差求和则刻意保持串行。
/// 这条性质一旦破坏,重现问题会变得极其困难,值得专门守住。
#[test]
fn results_are_independent_of_thread_count() {
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                let mut s = solver(17, 48, 1.1);
                s.run(Some(25), |_, _| {}).unwrap();
                (
                    s.dom.cells.rho.to_interior_vec(),
                    s.dom.cells.vx.to_interior_vec(),
                    s.dom.cells.nut.to_interior_vec(),
                    s.totaltime,
                )
            })
    };
    let base = run(1);
    for threads in [2, 3, 7, 16] {
        let got = run(threads);
        assert_eq!(base.0, got.0, "density differs at {threads} threads");
        assert_eq!(base.1, got.1, "velocity differs at {threads} threads");
        assert_eq!(base.2, got.2, "nu-tilde differs at {threads} threads");
        assert_eq!(base.3, got.3, "physical time differs at {threads} threads");
    }
}

/// 全局质量守恒:内部面通量两两抵消,净通量只来自壁面与远场。
///
/// 壁面镜像使壁面通量为 0,所以稳态时净流入应趋于 0。这里检验的是**离散**守恒
/// 性 —— 环量装配的符号只要有一处写反就会破坏它。
#[test]
fn discrete_mass_conservation_holds() {
    let mut s = solver(17, 48, 1.1);
    s.run(Some(30), |_, _| {}).unwrap();
    jst::timestep::compute(&s.cfg, &s.dom.geom, &mut s.dom.cells);
    s.residual_terms();

    let (ni, nj) = (s.ni() as isize, s.nj() as isize);
    let cells_total: f64 = (0..ni)
        .flat_map(|i| (0..nj).map(move |j| (i, j)))
        .map(|(i, j)| s.dom.cells.fc.get(i, j)[comp::RHO])
        .sum();
    // 周向面完全抵消(周期);径向只剩壁面与远场
    let boundary: f64 = (0..nj)
        .map(|j| {
            s.dom.faces.tau.flux.get(ni, j)[comp::RHO] - s.dom.faces.tau.flux.get(0, j)[comp::RHO]
        })
        .sum();
    let scale = (0..ni)
        .flat_map(|i| (0..nj).map(move |j| (i, j)))
        .map(|(i, j)| s.dom.cells.fc.get(i, j)[comp::RHO].abs())
        .fold(0.0f64, f64::max);
    assert!(
        (cells_total - boundary).abs() < 1e-10 * scale.max(1.0),
        "mass not conserved: interior sum {cells_total:e} vs boundary {boundary:e}"
    );
}

/// 壁面不得有质量穿透(镜像边界条件的直接后果)。
#[test]
fn no_mass_flux_through_the_wall() {
    let mut s = solver(17, 48, 1.1);
    s.run(Some(30), |_, _| {}).unwrap();
    jst::timestep::compute(&s.cfg, &s.dom.geom, &mut s.dom.cells);
    s.residual_terms();
    let scale = (0..s.nj() as isize)
        .map(|j| s.dom.faces.tau.flux.get(1, j)[comp::RHO].abs())
        .fold(0.0f64, f64::max);
    for j in 0..s.nj() as isize {
        let m = s.dom.faces.tau.flux.get(0, j)[comp::RHO];
        assert!(m.abs() < 1e-10 * scale.max(1.0), "wall leaks at j={j}: {m:e}");
    }
}

/// 自由来流保持性(整条残差链路):均匀场下平均流残差为机器精度。
#[test]
fn free_stream_is_preserved_end_to_end() {
    let mut s = solver(13, 40, 1.0);
    s.dom
        .cells
        .set_uniform(&s.cfg, 1.176, 69.4, 17.3, 101_325.0, 1.5e-4);
    jst::timestep::compute(&s.cfg, &s.dom.geom, &mut s.dom.cells);
    // 注意不能走 residual_terms:它会先应用边界条件,破坏人为铺设的均匀虚拟层
    let jst::Domain { geom, cells, faces } = &mut s.dom;
    jst::convection::compute(&s.cfg, geom, cells, faces);
    jst::gradient::compute(geom, cells);
    jst::viscous::compute(&s.cfg, geom, cells, faces);
    jst::dissipation::compute(&s.cfg, geom, cells, faces);

    let scale = faces
        .tau
        .flux
        .interior()
        .map(|(i, j)| faces.tau.flux.get(i, j).amax())
        .fold(0.0f64, f64::max);
    for (i, j) in cells.fc.interior() {
        let r = cells.fc.get(i, j) - cells.fv.get(i, j) - cells.fd.get(i, j);
        for k in 0..4 {
            assert!(
                r[k].abs() < 1e-13 * scale,
                "free stream not preserved: residual[{k}] = {:e} at ({i},{j})",
                r[k]
            );
        }
    }
}

/// 在若干种网格上都能稳定推进并保持物理性。
#[test]
fn stays_physical_across_mesh_shapes() {
    for &(rings, nj, stretch) in &[(8usize, 16usize, 1.0f64), (13, 40, 1.0), (21, 64, 1.25)] {
        let mut s = solver(rings, nj, stretch);
        s.run(Some(40), |_, _| {})
            .unwrap_or_else(|e| panic!("blew up on {rings}x{nj} stretch {stretch}: {e}"));
        for (i, j) in s.dom.cells.rho.interior() {
            assert!(s.dom.cells.rho.get(i, j) > 0.0, "rho<=0 at ({i},{j})");
            assert!(s.dom.cells.p.get(i, j) > 0.0, "p<=0 at ({i},{j})");
            assert!(s.dom.cells.t.get(i, j).is_finite());
            assert!(s.dom.cells.nut.get(i, j).is_finite());
        }
    }
}

/// 残差应当持续下降直到收敛。
///
/// 判据用"相对初值下降若干个数量级"而非某个绝对阈值:绝对残差的量级取决于
/// 网格与来流条件,而下降**幅度**才是收敛性的本质。
#[test]
fn converges_on_a_moderate_mesh() {
    let mut s = solver(9, 32, 1.0);
    s.cfg.solver.targetres = 1e-9;
    let mut history = Vec::new();
    let report = s.run(Some(6000), |_, r| history.push(r)).unwrap();

    let drop = history[0] / report.residual;
    assert!(
        report.converged,
        "did not converge in {} steps: {:e} (dropped {:.1e}x from {:e})",
        report.steps, report.residual, drop, history[0]
    );
    assert!(
        drop > 1e6,
        "residual only dropped {drop:.2e}x: {:e} -> {:e}",
        history[0],
        report.residual
    );
    // 收敛后的解仍须物理
    for (i, j) in s.dom.cells.rho.interior() {
        assert!(s.dom.cells.rho.get(i, j) > 0.0 && s.dom.cells.p.get(i, j) > 0.0);
    }
}

/// 求解器必须**报错**而不是崩掉或产出 NaN。
#[test]
fn reports_blowup_instead_of_producing_nans() {
    let mut s = solver(13, 40, 1.0);
    s.cfg.simulation.cfl = 200.0; // 远超稳定极限
    match s.run(Some(400), |_, _| {}) {
        Err(e) => {
            let msg = e.to_string();
            assert!(msg.contains("non-physical"), "unexpected error: {msg}");
        }
        Ok(rep) => {
            // 若侥幸没发散,至少解仍须是物理的
            assert!(rep.residual.is_finite());
            for (i, j) in s.dom.cells.rho.interior() {
                assert!(s.dom.cells.rho.get(i, j) > 0.0, "silent NaN at ({i},{j})");
            }
        }
    }
}

/// 网格加密时时间步应随之减小(CFL 条件)。
#[test]
fn timestep_shrinks_with_mesh_refinement() {
    let dt = |rings, nj| {
        let mut s = solver(rings, nj, 1.0);
        jst::timestep::compute(&s.cfg, &s.dom.geom, &mut s.dom.cells)
    };
    let (coarse, fine) = (dt(9, 32), dt(17, 64));
    assert!(fine < coarse, "timestep did not shrink: {coarse:e} -> {fine:e}");
    assert!(fine > 0.2 * coarse, "timestep shrank far more than expected");
}
