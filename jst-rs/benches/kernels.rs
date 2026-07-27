//! 逐 kernel 的微基准 + 整步基准。
//!
//! 每个 kernel 单独计时,可以看出热点分布并防止性能回归 ——
//! 只有整体 wall clock 的话,某个 kernel 变慢一倍可能被别的加速掩盖掉。
//!
//! ```sh
//! cargo bench                      # 全部
//! cargo bench -- convection        # 只跑对流
//! ```

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use jst::{config::Config, mesh::Mesh, solver::Solver, timestep};

/// 与 `tools/genmesh.py` 同一族的椭圆柱 → 远场圆 O 型网格。
fn synth_mesh(ni: usize, nj: usize) -> Mesh {
    let (a_wall, b_wall, r_far) = (1.0, 0.5, 5.0);
    let mut txt = format!("{ni} {nj}\n");
    for i in 0..ni {
        let s = i as f64 / (ni - 1) as f64;
        let a = a_wall + s * (r_far - a_wall);
        let b = b_wall + s * (r_far - b_wall);
        for j in 0..nj {
            let th = 2.0 * std::f64::consts::PI * j as f64 / nj as f64;
            txt += &format!("{:.10} {:.10}\n", a * th.cos(), b * th.sin());
        }
    }
    Mesh::parse(&txt).expect("synthetic mesh")
}

fn make_solver(ni: usize, nj: usize) -> Solver {
    let cfg = Config::from_str(include_str!("../../config.json")).unwrap();
    let mut s = Solver::new(cfg, &synth_mesh(ni, nj));
    // 让各场进入有代表性的状态,避免在"精确均匀"这种特殊输入上测
    for _ in 0..3 {
        s.advance().unwrap();
    }
    timestep::compute(&s.cfg, &s.dom.geom, &mut s.dom.cells);
    s
}

/// 单个 kernel。
fn kernels(c: &mut Criterion) {
    let (ni, nj) = (129, 256);
    let cells = (ni - 1) * nj;
    let mut group = c.benchmark_group("kernel");
    group.throughput(Throughput::Elements(cells as u64));

    macro_rules! bench_kernel {
        ($name:literal, $s:ident, $body:expr) => {
            let mut $s = make_solver(ni, nj);
            group.bench_function($name, |b| b.iter(|| $body));
        };
    }

    bench_kernel!("boundary", s, {
        jst::boundary::apply(&s.cfg, &s.dom.geom, &mut s.dom.cells)
    });
    bench_kernel!("timestep", s, {
        timestep::compute(&s.cfg, &s.dom.geom, &mut s.dom.cells)
    });
    bench_kernel!("convection", s, {
        jst::convection::compute(&s.cfg, &s.dom.geom, &mut s.dom.cells, &mut s.dom.faces)
    });
    bench_kernel!("gradient", s, {
        jst::gradient::compute(&s.dom.geom, &mut s.dom.cells)
    });
    bench_kernel!("viscous", s, {
        jst::viscous::compute(&s.cfg, &s.dom.geom, &mut s.dom.cells, &mut s.dom.faces)
    });
    bench_kernel!("dissipation", s, {
        jst::dissipation::compute(&s.cfg, &s.dom.geom, &mut s.dom.cells, &mut s.dom.faces)
    });
    bench_kernel!("source", s, {
        jst::source::compute(&s.cfg, &s.dom.geom, &mut s.dom.cells)
    });
    group.finish();
}

/// 完整时间步随网格规模的伸缩。
fn step_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("step");
    for &(ni, nj) in &[(17usize, 40usize), (65, 128), (129, 256), (257, 512)] {
        let cells = (ni - 1) * nj;
        group.throughput(Throughput::Elements(cells as u64));
        let mut s = make_solver(ni, nj);
        group.bench_with_input(BenchmarkId::from_parameter(cells), &cells, |b, _| {
            b.iter(|| s.advance().unwrap())
        });
    }
    group.finish();
}

criterion_group!(benches, kernels, step_scaling);
criterion_main!(benches);
