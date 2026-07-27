// Probe: Green-Gauss gradient convergence on the ellipse->circle O-mesh family.
use jst::{config::Config, geometry::Geometry, mesh::Mesh, state::Domain};

fn err_at(cfg: &Config, rings: usize, nj: usize) -> (f64, f64) {
    let (ax, ay) = (3.7f64, -1.9f64);
    let mut txt = format!("{rings} {nj}\n");
    for i in 0..rings {
        let s = i as f64 / (rings - 1) as f64;
        let (a, b) = (1.0 + s * 4.0, 0.5 + s * 4.5);
        for j in 0..nj {
            let th = 2.0 * std::f64::consts::PI * j as f64 / nj as f64;
            txt += &format!("{:.12} {:.12}\n", a * th.cos(), b * th.sin());
        }
    }
    let mesh = Mesh::parse(&txt).unwrap();
    let geom = Geometry::build(&mesh, cfg.simulation.halo);
    let mut dom = Domain::new(geom, cfg.simulation.halo);
    dom.cells.initialize(cfg);
    let (ni, njj) = (dom.cells.ni as isize, dom.cells.nj as isize);
    let f = |x: f64, y: f64| ax * x + ay * y;
    for i in 0..ni {
        for j in 0..njj {
            dom.cells
                .vx
                .set(i, j, f(dom.geom.cx.get(i, j), dom.geom.cy.get(i, j)));
        }
    }
    for j in 0..njj {
        let w = dom.geom.tau.get(0, j);
        dom.cells
            .vx
            .set(-1, j, 2.0 * f(w.mx, w.my) - dom.cells.vx.get(0, j));
        let o = dom.geom.tau.get(ni, j);
        dom.cells
            .vx
            .set(ni, j, 2.0 * f(o.mx, o.my) - dom.cells.vx.get(ni - 1, j));
    }
    for i in 0..ni {
        dom.cells.vx.set(i, -1, dom.cells.vx.get(i, njj - 1));
        dom.cells.vx.set(i, njj, dom.cells.vx.get(i, 0));
    }
    jst::gradient::compute(&dom.geom, &mut dom.cells);
    let errs = || {
        dom.cells
            .grad
            .interior()
            .map(|(i, j)| (dom.cells.grad.get(i, j).dudx - ax).abs() / ax.abs())
    };
    let n = (ni * njj) as f64;
    (errs().fold(0.0f64, f64::max), errs().sum::<f64>() / n)
}

fn main() {
    let cfg = Config::from_str(include_str!("../config.json")).unwrap();
    let mut prev: Option<(f64, f64)> = None;
    for &(r, n) in &[(9usize, 32usize), (17, 64), (33, 128), (65, 256), (129, 512)] {
        let (linf, l1) = err_at(&cfg, r, n);
        match prev {
            Some((pi, p1)) => println!(
                "rings={r:4} nj={n:4}  Linf={linf:.4e} (x{:.2})  L1={l1:.4e} (x{:.2})",
                pi / linf,
                p1 / l1
            ),
            None => println!("rings={r:4} nj={n:4}  Linf={linf:.4e}          L1={l1:.4e}"),
        }
        prev = Some((linf, l1));
    }
}

