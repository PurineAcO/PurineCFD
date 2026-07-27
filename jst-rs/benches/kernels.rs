//! 閫?kernel 鐨勫井鍩哄噯 + 鏁存鍩哄噯銆?//!
//! 姣忎釜 kernel 鍗曠嫭璁℃椂,鍙互鐪嬪嚭鐑偣鍒嗗竷骞堕槻姝㈡€ц兘鍥炲綊 鈥斺€?//! 鍙湁鏁翠綋 wall clock 鐨勮瘽,鏌愪釜 kernel 鍙樻參涓€鍊嶅彲鑳借鍒殑鍔犻€熸帺鐩栨帀銆?//!
//! ```sh
//! cargo bench                      # 鍏ㄩ儴
//! cargo bench -- convection        # 鍙窇瀵规祦
//! ```

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use jst::{config::Config, mesh::Mesh, solver::Solver, timestep};

/// 涓?`tools/genmesh.py` 鍚屼竴鏃忕殑妞渾鏌?鈫?杩滃満鍦?O 鍨嬬綉鏍笺€?fn synth_mesh(ni: usize, nj: usize) -> Mesh {
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
    let cfg = Config::from_str(include_str!("../config.json")).unwrap();
    let mut s = Solver::new(cfg, &synth_mesh(ni, nj));
    // 璁╁悇鍦鸿繘鍏ユ湁浠ｈ〃鎬х殑鐘舵€?閬垮厤鍦?绮剧‘鍧囧寑"杩欑鐗规畩杈撳叆涓婃祴
    for _ in 0..3 {
        s.advance().unwrap();
    }
    timestep::compute(&s.cfg, &s.dom.geom, &mut s.dom.cells);
    s
}

/// 鍗曚釜 kernel銆?fn kernels(c: &mut Criterion) {
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

/// 瀹屾暣鏃堕棿姝ラ殢缃戞牸瑙勬ā鐨勪几缂┿€?fn step_scaling(c: &mut Criterion) {
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

