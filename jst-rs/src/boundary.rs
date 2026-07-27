//! 边界条件 —— **整个求解器里唯一处理虚拟单元的地方**。
//!
//! Python 基线把这三类边界的下标映射散落在每个 kernel 里重复实现,`BUGS.md` 中
//! B4/B5/B6/B8 四个数值错误全部源于此。这里把它们收敛成一次 [`apply`]:
//! 之后所有 kernel 都是不带特判的矩形循环。
//!
//! 三类边界:
//!
//! 1. **固壁**(`i < 0`):镜像法。`i = -1-k` ← 物理单元 `k`,标量同层复制,
//!    速度与 ν̃ 反号 —— 于是壁面上的法向速度与 ν̃ 的插值恰为 0(无滑移、无穿透)。
//! 2. **远场**(`i ≥ NI`):一维黎曼不变量。沿边界法向解出 `R⁺ = vₙ + 2c/(γ−1)`
//!    (内部带出)与 `R⁻ = vₙ − 2c/(γ−1)`(来流带入),再按 `vₙ` 的方向决定切向
//!    速度与 ν̃ 取内部值还是来流值。
//! 3. **周向周期**(`j < 0` 或 `j ≥ NJ`):O 型网格切割线,直接按 `j mod NJ` 取值。
//!
//! 角落 halo(i、j 同时越界)不填 —— 所有 kernel 的模板都是十字形,从不访问角落。

use crate::config::Config;
use crate::geometry::Geometry;
use crate::state::Cells;

/// 把物理单元与远场边界的状态同步到全部虚拟单元上。
pub fn apply(cfg: &Config, geom: &Geometry, cells: &mut Cells) {
    wall_mirror(cells, cfg.simulation.halo);
    far_field_riemann(cfg, geom, cells);
    periodic_seam(cells, cfg.simulation.halo);
}

/// 固壁镜像:`i = -1-k` ← 物理单元 `k`。
fn wall_mirror(cells: &mut Cells, halo: usize) {
    let nj = cells.nj as isize;
    for k in 0..halo as isize {
        let ghost = -1 - k;
        for j in 0..nj {
            // 标量同层复制。
            // BUGFIX(对照 Python B8):基线的标量恒取第 0 层而速度取第 k 层,
            // 镜像不自洽 —— 这里统一取第 k 层。
            cells.rho.set(ghost, j, cells.rho.get(k, j));
            cells.p.set(ghost, j, cells.p.get(k, j));
            cells.t.set(ghost, j, cells.t.get(k, j));
            cells.e.set(ghost, j, cells.e.get(k, j));
            cells.h.set(ghost, j, cells.h.get(k, j));
            cells.c.set(ghost, j, cells.c.get(k, j));
            // 速度与湍流量反号
            cells.vx.set(ghost, j, -cells.vx.get(k, j));
            cells.vy.set(ghost, j, -cells.vy.get(k, j));
            cells.nut.set(ghost, j, -cells.nut.get(k, j));
            cells.pack(ghost, j);
        }
    }
}

/// 周向周期:切割线两侧按 `j mod NJ` 取值。
fn periodic_seam(cells: &mut Cells, halo: usize) {
    let (ni, nj) = (cells.ni as isize, cells.nj as isize);
    for i in 0..ni {
        for k in 1..=halo as isize {
            for (ghost, src) in [(-k, nj - k), (nj - 1 + k, k - 1)] {
                cells.rho.set(i, ghost, cells.rho.get(i, src));
                cells.p.set(i, ghost, cells.p.get(i, src));
                cells.t.set(i, ghost, cells.t.get(i, src));
                cells.e.set(i, ghost, cells.e.get(i, src));
                cells.h.set(i, ghost, cells.h.get(i, src));
                cells.c.set(i, ghost, cells.c.get(i, src));
                cells.vx.set(i, ghost, cells.vx.get(i, src));
                cells.vy.set(i, ghost, cells.vy.get(i, src));
                cells.nut.set(i, ghost, cells.nut.get(i, src));
                cells.pack(i, ghost);
            }
        }
    }
}

/// 远场边界面上的状态。
#[derive(Clone, Copy, Debug, Default)]
pub struct FarState {
    pub rho: f64,
    pub p: f64,
    pub t: f64,
    pub vx: f64,
    pub vy: f64,
    pub e: f64,
    pub nut: f64,
}

/// 远场:一维黎曼不变量,结果写入全部远场虚拟层。
fn far_field_riemann(cfg: &Config, geom: &Geometry, cells: &mut Cells) {
    let (ni, nj) = (cells.ni as isize, cells.nj as isize);
    let halo = cfg.simulation.halo as isize;

    for j in 0..nj {
        let s = riemann_face(cfg, geom, cells, j);
        let c = (cfg.physics.gamma * cfg.physics.r_gas * s.t).sqrt();
        let h = s.e + s.p / s.rho;
        for k in 0..halo {
            let g = ni + k;
            cells.rho.set(g, j, s.rho);
            cells.p.set(g, j, s.p);
            cells.t.set(g, j, s.t);
            cells.e.set(g, j, s.e);
            cells.h.set(g, j, h);
            cells.c.set(g, j, c);
            cells.vx.set(g, j, s.vx);
            cells.vy.set(g, j, s.vy);
            cells.nut.set(g, j, s.nut);
            cells.pack(g, j);
        }
    }
}

/// 解第 `j` 条远场边界面上的黎曼问题。
pub fn riemann_face(cfg: &Config, geom: &Geometry, cells: &Cells, j: isize) -> FarState {
    let ni = cells.ni as isize;
    let face = geom.tau.get(ni, j);
    let len = face.length();
    let (nx, ny) = (face.nx, face.ny);
    let gamma = cfg.physics.gamma;
    let d = &cfg.derived;

    // 法向/切向速度分量。法向指向计算域外侧,故 vn < 0 表示入流。
    let project = |u: f64, v: f64| ((u * nx + v * ny) / len, (u * ny - v * nx) / len);
    let (vn_inf, vt_inf) = project(d.u_inf, d.v_inf);
    let (vn_in, vt_in) = project(cells.vx.get(ni - 1, j), cells.vy.get(ni - 1, j));

    let (c_face, vn_face, vt_face, inflow) = if !d.supersonic {
        let r_in = vn_in + 2.0 * cells.c.get(ni - 1, j) / (gamma - 1.0);
        let r_inf = vn_inf - 2.0 * d.c_inf / (gamma - 1.0);
        let vn = 0.5 * (r_in + r_inf);
        let c = (gamma - 1.0) / 4.0 * (r_in - r_inf);
        let inflow = vn <= 0.0;
        (c, vn, if inflow { vt_inf } else { vt_in }, inflow)
    } else if vn_inf <= 0.0 {
        (d.c_inf, vn_inf, vt_inf, true)
    } else {
        (cells.c.get(ni - 1, j), vn_in, vt_in, false)
    };

    // 由 (c, vn, vt) 重构面上的完整状态,密度走等熵关系
    let t = c_face * c_face / (cfg.physics.r_gas * gamma);
    let rho = d.rho_inf * (t / cfg.simulation.t_inf).powf(1.0 / (gamma - 1.0));
    let p = rho * cfg.physics.r_gas * t;
    let vx = (vt_face * ny + vn_face * nx) / len;
    let vy = (-vt_face * nx + vn_face * ny) / len;
    let e = p / (rho * (gamma - 1.0)) + 0.5 * (vx * vx + vy * vy);

    // 入流:ν̃ 取来流值(置 0 会让湍流被持续冲刷掉,见 BUGS.md B10)
    // 出流:由内部三个单元二次外插
    let nut = if inflow {
        d.nut_inf
    } else {
        extrapolate_nut(geom, cells, j)
    };

    FarState { rho, p, t, vx, vy, e, nut }
}

/// 由最外三层单元向远场面做三点 Lagrange 外插,得到 ν̃。
fn extrapolate_nut(geom: &Geometry, cells: &Cells, j: isize) -> f64 {
    const FLOOR: f64 = 1e-10;
    let ni = cells.ni as isize;
    let face = geom.tau.get(ni, j);

    let mut dist = [0.0f64; 3];
    let mut nut = [0.0f64; 3];
    for (k, slot) in dist.iter_mut().enumerate() {
        let i = ni - 1 - k as isize;
        *slot = (geom.cx.get(i, j) - face.mx).hypot(geom.cy.get(i, j) - face.my);
        nut[k] = cells.nut.get(i, j);
    }

    let (l1, l2, l3) = (dist[0], dist[1], dist[2]);
    let (d12, d13, d23) = (l1 - l2, l1 - l3, l2 - l3);
    // 退化网格(两层单元到面等距)会让分母为 0,退回一阶外插
    if d12.abs() < 1e-14 || d13.abs() < 1e-14 || d23.abs() < 1e-14 {
        return nut[0].max(FLOOR);
    }

    let v = l2 * l3 / (d12 * d13) * nut[0] + l1 * l3 / (-d12 * d23) * nut[1]
        + l1 * l2 / (d23 * -d13) * nut[2];
    if v > FLOOR {
        v
    } else {
        FLOOR
    }
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
        (cfg, dom)
    }

    #[test]
    fn riemann_recovers_freestream_from_freestream_interior() {
        let (cfg, dom) = setup();
        let d = &cfg.derived;
        for j in 0..dom.cells.nj as isize {
            let s = riemann_face(&cfg, &dom.geom, &dom.cells, j);
            assert!((s.t - cfg.simulation.t_inf).abs() < 1e-9, "T at j={j}: {}", s.t);
            assert!((s.rho - d.rho_inf).abs() < 1e-11 * d.rho_inf);
            assert!((s.p - cfg.simulation.p_inf).abs() < 1e-9 * cfg.simulation.p_inf);
            assert!((s.vx - d.u_inf).abs() < 1e-8 * d.c_inf);
            assert!((s.vy - d.v_inf).abs() < 1e-8 * d.c_inf);
        }
    }

    #[test]
    fn inflow_takes_freestream_nut() {
        let (cfg, dom) = setup();
        let mut saw_inflow = false;
        for j in 0..dom.cells.nj as isize {
            let f = dom.geom.tau.get(dom.cells.ni as isize, j);
            let s = riemann_face(&cfg, &dom.geom, &dom.cells, j);
            if s.vx * f.nx + s.vy * f.ny <= 0.0 {
                saw_inflow = true;
                assert!((s.nut - cfg.derived.nut_inf).abs() < 1e-18);
            } else {
                assert!(s.nut > 0.0);
            }
        }
        assert!(saw_inflow, "expected some inflow faces on this case");
    }

    #[test]
    fn wall_ghost_mirrors_velocity_and_copies_scalars_per_layer() {
        let (cfg, mut dom) = setup();
        // 造一个沿 i 变化的场,否则各层标量相同、测不出层错位
        for (i, j) in dom.cells.rho.interior().collect::<Vec<_>>() {
            let p = cfg.simulation.p_inf * (1.0 + 0.02 * (i + 1) as f64);
            dom.cells.p.set(i, j, p);
            dom.cells.vx.set(i, j, 10.0 * (i + 1) as f64);
            dom.cells.pack(i, j);
        }
        apply(&cfg, &dom.geom, &mut dom.cells);
        for k in 0..cfg.simulation.halo as isize {
            for j in 0..dom.cells.nj as isize {
                let g = -1 - k;
                assert_eq!(dom.cells.p.get(g, j), dom.cells.p.get(k, j));
                assert_eq!(dom.cells.vx.get(g, j), -dom.cells.vx.get(k, j));
                assert_eq!(dom.cells.nut.get(g, j), -dom.cells.nut.get(k, j));
            }
        }
    }

    #[test]
    fn periodic_ghosts_wrap_modulo_nj() {
        let (cfg, mut dom) = setup();
        let nj = dom.cells.nj as isize;
        for (i, j) in dom.cells.rho.interior().collect::<Vec<_>>() {
            dom.cells.rho.set(i, j, 1.0 + 0.01 * j as f64);
            dom.cells.pack(i, j);
        }
        apply(&cfg, &dom.geom, &mut dom.cells);
        for i in 0..dom.cells.ni as isize {
            for k in 1..=cfg.simulation.halo as isize {
                assert_eq!(dom.cells.rho.get(i, -k), dom.cells.rho.get(i, nj - k));
                assert_eq!(dom.cells.rho.get(i, nj - 1 + k), dom.cells.rho.get(i, k - 1));
            }
        }
    }

    #[test]
    fn applying_twice_is_idempotent() {
        let (cfg, mut dom) = setup();
        apply(&cfg, &dom.geom, &mut dom.cells);
        let snapshot: Vec<f64> = dom.cells.rho.raw().to_vec();
        apply(&cfg, &dom.geom, &mut dom.cells);
        assert_eq!(snapshot, dom.cells.rho.raw());
    }
}
