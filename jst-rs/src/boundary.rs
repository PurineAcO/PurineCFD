//! 杈圭晫鏉′欢 鈥斺€?**鏁翠釜姹傝В鍣ㄩ噷鍞竴澶勭悊铏氭嫙鍗曞厓鐨勫湴鏂?*銆?//!
//! Python 鍩虹嚎鎶婅繖涓夌被杈圭晫鐨勪笅鏍囨槧灏勬暎钀藉湪姣忎釜 kernel 閲岄噸澶嶅疄鐜?`BUGS.md` 涓?//! B4/B5/B6/B8 鍥涗釜鏁板€奸敊璇叏閮ㄦ簮浜庢銆傝繖閲屾妸瀹冧滑鏀舵暃鎴愪竴娆?[`apply`]:
//! 涔嬪悗鎵€鏈?kernel 閮芥槸涓嶅甫鐗瑰垽鐨勭煩褰㈠惊鐜€?//!
//! 涓夌被杈圭晫:
//!
//! 1. **鍥哄**(`i < 0`):闀滃儚娉曘€俙i = -1-k` 鈫?鐗╃悊鍗曞厓 `k`,鏍囬噺鍚屽眰澶嶅埗,
//!    閫熷害涓?谓虄 鍙嶅彿 鈥斺€?浜庢槸澹侀潰涓婄殑娉曞悜閫熷害涓?谓虄 鐨勬彃鍊兼伆涓?0(鏃犳粦绉汇€佹棤绌块€?銆?//! 2. **杩滃満**(`i 鈮?NI`):涓€缁撮粠鏇间笉鍙橀噺銆傛部杈圭晫娉曞悜瑙ｅ嚭 `R鈦?= v鈧?+ 2c/(纬鈭?)`
//!    (鍐呴儴甯﹀嚭)涓?`R鈦?= v鈧?鈭?2c/(纬鈭?)`(鏉ユ祦甯﹀叆),鍐嶆寜 `v鈧檂 鐨勬柟鍚戝喅瀹氬垏鍚?//!    閫熷害涓?谓虄 鍙栧唴閮ㄥ€艰繕鏄潵娴佸€笺€?//! 3. **鍛ㄥ悜鍛ㄦ湡**(`j < 0` 鎴?`j 鈮?NJ`):O 鍨嬬綉鏍煎垏鍓茬嚎,鐩存帴鎸?`j mod NJ` 鍙栧€笺€?//!
//! 瑙掕惤 halo(i銆乯 鍚屾椂瓒婄晫)涓嶅～ 鈥斺€?鎵€鏈?kernel 鐨勬ā鏉块兘鏄崄瀛楀舰,浠庝笉璁块棶瑙掕惤銆?
use crate::config::Config;
use crate::geometry::Geometry;
use crate::state::Cells;

/// 鎶婄墿鐞嗗崟鍏冧笌杩滃満杈圭晫鐨勭姸鎬佸悓姝ュ埌鍏ㄩ儴铏氭嫙鍗曞厓涓娿€?pub fn apply(cfg: &Config, geom: &Geometry, cells: &mut Cells) {
    wall_mirror(cells, cfg.simulation.halo);
    far_field_riemann(cfg, geom, cells);
    periodic_seam(cells, cfg.simulation.halo);
}

/// 鍥哄闀滃儚:`i = -1-k` 鈫?鐗╃悊鍗曞厓 `k`銆?fn wall_mirror(cells: &mut Cells, halo: usize) {
    let nj = cells.nj as isize;
    for k in 0..halo as isize {
        let ghost = -1 - k;
        for j in 0..nj {
            // 鏍囬噺鍚屽眰澶嶅埗銆?            // BUGFIX(瀵圭収 Python B8):鍩虹嚎鐨勬爣閲忔亽鍙栫 0 灞傝€岄€熷害鍙栫 k 灞?
            // 闀滃儚涓嶈嚜娲?鈥斺€?杩欓噷缁熶竴鍙栫 k 灞傘€?            cells.rho.set(ghost, j, cells.rho.get(k, j));
            cells.p.set(ghost, j, cells.p.get(k, j));
            cells.t.set(ghost, j, cells.t.get(k, j));
            cells.e.set(ghost, j, cells.e.get(k, j));
            cells.h.set(ghost, j, cells.h.get(k, j));
            cells.c.set(ghost, j, cells.c.get(k, j));
            // 閫熷害涓庢箥娴侀噺鍙嶅彿
            cells.vx.set(ghost, j, -cells.vx.get(k, j));
            cells.vy.set(ghost, j, -cells.vy.get(k, j));
            cells.nut.set(ghost, j, -cells.nut.get(k, j));
            cells.pack(ghost, j);
        }
    }
}

/// 鍛ㄥ悜鍛ㄦ湡:鍒囧壊绾夸袱渚ф寜 `j mod NJ` 鍙栧€笺€?fn periodic_seam(cells: &mut Cells, halo: usize) {
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

/// 杩滃満杈圭晫闈笂鐨勭姸鎬併€?#[derive(Clone, Copy, Debug, Default)]
pub struct FarState {
    pub rho: f64,
    pub p: f64,
    pub t: f64,
    pub vx: f64,
    pub vy: f64,
    pub e: f64,
    pub nut: f64,
}

/// 杩滃満:涓€缁撮粠鏇间笉鍙橀噺,缁撴灉鍐欏叆鍏ㄩ儴杩滃満铏氭嫙灞傘€?fn far_field_riemann(cfg: &Config, geom: &Geometry, cells: &mut Cells) {
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

/// 瑙ｇ `j` 鏉¤繙鍦鸿竟鐣岄潰涓婄殑榛庢浖闂銆?pub fn riemann_face(cfg: &Config, geom: &Geometry, cells: &Cells, j: isize) -> FarState {
    let ni = cells.ni as isize;
    let face = geom.tau.get(ni, j);
    let len = face.length();
    let (nx, ny) = (face.nx, face.ny);
    let gamma = cfg.physics.gamma;
    let d = &cfg.derived;

    // 娉曞悜/鍒囧悜閫熷害鍒嗛噺銆傛硶鍚戞寚鍚戣绠楀煙澶栦晶,鏁?vn < 0 琛ㄧず鍏ユ祦銆?    let project = |u: f64, v: f64| ((u * nx + v * ny) / len, (u * ny - v * nx) / len);
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

    // 鐢?(c, vn, vt) 閲嶆瀯闈笂鐨勫畬鏁寸姸鎬?瀵嗗害璧扮瓑鐔靛叧绯?    let t = c_face * c_face / (cfg.physics.r_gas * gamma);
    let rho = d.rho_inf * (t / cfg.simulation.t_inf).powf(1.0 / (gamma - 1.0));
    let p = rho * cfg.physics.r_gas * t;
    let vx = (vt_face * ny + vn_face * nx) / len;
    let vy = (-vt_face * nx + vn_face * ny) / len;
    let e = p / (rho * (gamma - 1.0)) + 0.5 * (vx * vx + vy * vy);

    // 鍏ユ祦:谓虄 鍙栨潵娴佸€?缃?0 浼氳婀嶆祦琚寔缁啿鍒锋帀,瑙?BUGS.md B10)
    // 鍑烘祦:鐢卞唴閮ㄤ笁涓崟鍏冧簩娆″鎻?    let nut = if inflow {
        d.nut_inf
    } else {
        extrapolate_nut(geom, cells, j)
    };

    FarState { rho, p, t, vx, vy, e, nut }
}

/// 鐢辨渶澶栦笁灞傚崟鍏冨悜杩滃満闈㈠仛涓夌偣 Lagrange 澶栨彃,寰楀埌 谓虄銆?fn extrapolate_nut(geom: &Geometry, cells: &Cells, j: isize) -> f64 {
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
    // 閫€鍖栫綉鏍?涓ゅ眰鍗曞厓鍒伴潰绛夎窛)浼氳鍒嗘瘝涓?0,閫€鍥炰竴闃跺鎻?    if d12.abs() < 1e-14 || d13.abs() < 1e-14 || d23.abs() < 1e-14 {
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
        let cfg = Config::from_str(include_str!("../config.json")).unwrap();
        let mesh = Mesh::parse(include_str!("../fangdata.txt")).unwrap();
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
        // 閫犱竴涓部 i 鍙樺寲鐨勫満,鍚﹀垯鍚勫眰鏍囬噺鐩稿悓銆佹祴涓嶅嚭灞傞敊浣?        for (i, j) in dom.cells.rho.interior().collect::<Vec<_>>() {
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

