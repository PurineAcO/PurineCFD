//! Green-Gauss 鍗曞厓姊害銆?//!
//! ```text
//! 鈭囅唡岬⑩奔 = (1/V) 危_faces 蠁_face 路 n_face
//! ```
//!
//! 闈笂鐨?蠁 鍙栫浉閭讳袱鍗曞厓鐨勭畻鏈钩鍧?涓€闃朵腑蹇?銆傚潎鍖€鍦轰笅 `危卤n 鈮?0` 淇濊瘉姊害绮剧‘
//! 涓?0 鈥斺€?鐢辨湰妯″潡鐨勮嚜鐢辨潵娴佺敤渚嬫妸鍏炽€?//!
//! 鍏釜鍒嗛噺鎵撳寘杩?[`Grad`] 涓€璧峰啓:鍥涗釜鍙橀噺鐨勬搴︽€绘槸琚矘鎬ч」涓庢簮椤瑰悓鏃舵秷璐?
//! 鎵撳寘鍚庢湰 kernel 鍙啓涓€涓暟缁?鐪佹帀鍏矾骞惰杩唬鍣ㄧ殑鍚屾鍒囧垎寮€閿€銆?//!
//! 娉ㄦ剰:Python 鍩虹嚎**浠庢湭璋冪敤**姊害璁＄畻(`BUGS.md` A5),瀵艰嚧绮樻€ч」涓庢箥娴佹簮椤?//! 鎭掍负 0,N-S 鏂圭▼闈欓粯閫€鍖栨垚 Euler 鏂圭▼銆?
use rayon::iter::ParallelIterator;

use crate::geometry::Geometry;
use crate::state::{Cells, Grad};

/// 涓€涓爣閲忓満鍦ㄥ洓涓潰涓婄殑 Green-Gauss 璐＄尞銆?macro_rules! gg {
    ($src:expr, $i:expr, $j:expr, $jm:expr, $jp:expr,
     $up:expr, $dn:expr, $lf:expr, $rt:expr, $inv_v:expr) => {{
        let c = $src.get($i, $j);
        let f_up = 0.5 * (c + $src.get($i + 1, $j));
        let f_dn = 0.5 * (c + $src.get($i - 1, $j));
        let f_lf = 0.5 * (c + $src.get($i, $jm));
        let f_rt = 0.5 * (c + $src.get($i, $jp));
        (
            (f_up * $up.nx - f_dn * $dn.nx + f_rt * $rt.nx - f_lf * $lf.nx) * $inv_v,
            (f_up * $up.ny - f_dn * $dn.ny + f_rt * $rt.ny - f_lf * $lf.ny) * $inv_v,
        )
    }};
}

/// 璁＄畻 u銆乿銆乀銆佄教?鍦?*鐗╃悊鍗曞厓**涓婄殑姊害,闅忓悗鎸夎竟鐣屾潯浠跺～鍏呯涓€灞傝櫄鎷熷崟鍏冦€?pub fn compute(geom: &Geometry, cells: &mut Cells) {
    let nj = geom.nj as isize;
    let (inv_vol, tau, nrm) = (&geom.inv_vol, &geom.tau, &geom.nrm);
    // 鎷嗗€?鍙啓 grad,璇诲叾浣欏洓涓暟缁?鈥斺€?鍊熺敤妫€鏌ヨ瘉鏄庢棤鍒悕
    let Cells {
        grad,
        vx,
        vy,
        t,
        nut,
        ..
    } = cells;
    let (vx, vy, t, nut) = (&*vx, &*vy, &*t, &*nut);

    grad.par_interior_rows_mut().for_each(|(i, mut row)| {
        for j in 0..nj {
            let jm = j - 1;
            let jp = j + 1;
            let up = tau.at(i + 1, j);
            let dn = tau.at(i, j);
            let lf = nrm.at(i, j);
            let rt = nrm.at(i, if jp < nj { jp } else { 0 });
            let inv_v = inv_vol.get(i, j);

            let (dudx, dudy) = gg!(vx, i, j, jm, jp, up, dn, lf, rt, inv_v);
            let (dvdx, dvdy) = gg!(vy, i, j, jm, jp, up, dn, lf, rt, inv_v);
            let (dtdx, dtdy) = gg!(t, i, j, jm, jp, up, dn, lf, rt, inv_v);
            let (dnutdx, dnutdy) = gg!(nut, i, j, jm, jp, up, dn, lf, rt, inv_v);
            row[j] = Grad {
                dudx,
                dudy,
                dvdx,
                dvdy,
                dtdx,
                dtdy,
                dnutdx,
                dnutdy,
            };
        }
    });

    fill_ghost_gradients(cells);
}

/// 绗竴灞傝櫄鎷熷崟鍏冪殑姊害 鈥斺€?鍙湁瀹冧滑浼氳 [`crate::viscous`] 鐨勯潰骞冲潎鐢ㄥ埌銆?///
/// * 鍥哄:閫熷害涓?谓虄 鐨勬搴﹀欢鎷撹嚜璐村鍗曞厓;娓╁害姊害缃?0(缁濈儹澹?銆?/// * 杩滃満:鍏ㄩ儴缃?0(绮樻€у奖鍝嶅湪杩滃満鍙拷鐣?銆?/// * 鍛ㄥ悜:鎸夊懆鏈熺洿鎺ュ鍒躲€?fn fill_ghost_gradients(cells: &mut Cells) {
    let (ni, nj) = (cells.ni as isize, cells.nj as isize);
    let g = &mut cells.grad;

    for j in 0..nj {
        let mut wall = g.get(0, j);
        wall.dtdx = 0.0;
        wall.dtdy = 0.0;
        g.set(-1, j, wall);
        g.set(ni, j, Grad::default());
    }
    for i in 0..ni {
        g.set(i, -1, g.get(i, nj - 1));
        g.set(i, nj, g.get(i, 0));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
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

    /// 鍧囧寑鍦虹殑姊害蹇呴』绮剧‘涓?0 鈥斺€?鍙緷璧栧害閲忛棴鍚?鏄渶閿愬埄鐨勭储寮?绗﹀彿妫€鏌ャ€?    #[test]
    fn uniform_field_has_zero_gradient() {
        let (cfg, mut dom) = setup();
        dom.cells.set_uniform(&cfg, 1.176, 69.4, 17.3, 101325.0, 1.5e-4);
        compute(&dom.geom, &mut dom.cells);
        let vol = dom.geom.vol.get(0, 0);
        let scale = dom.cells.vx.get(0, 0).abs() / vol;
        let t_scale = dom.cells.t.get(0, 0).abs() / vol;
        for (i, j) in dom.cells.grad.interior() {
            let g = dom.cells.grad.get(i, j);
            assert!(g.dudx.abs() < 1e-13 * scale, "dudx at ({i},{j}) = {:e}", g.dudx);
            assert!(g.dudy.abs() < 1e-13 * scale);
            assert!(g.dvdx.abs() < 1e-13 * scale);
            assert!(g.dvdy.abs() < 1e-13 * scale);
            assert!(g.dtdx.abs() < 1e-13 * t_scale);
            assert!(g.dtdy.abs() < 1e-13 * t_scale);
        }
    }

    /// 缃戞牸鏀舵暃鎬х爺绌?绾挎€у満涓婄殑姊害璇樊闅忓姞瀵嗚€屽噺灏忋€?    ///
    /// 杩欓噷**涓嶈兘**瑕佹眰"绾挎€у満绮剧‘澶嶇幇" 鈥斺€?闈㈠€煎彇鐨勬槸涓や釜鍗曞厓涓績鐨勭畻鏈钩鍧?
    /// 瀹冪瓑浜庣嚎鎬у嚱鏁板湪涓?*褰㈠績**涓偣涓婄殑鍊?鑰岄潪闈腑鐐逛笂鐨勫€?闈炲潎鍖€缃戞牸涓?    /// 浜岃€呬笉閲嶅悎,鏁呯畝鍗曞钩鍧囩殑 Green-Gauss 鍦ㄦ渶鍧忓崟鍏冧笂鍙湁涓€闃剁簿搴︺€?    /// 瀹炴祴(瑙?`examples/gradient_convergence.rs`):
    ///
    /// ```text
    ///   9x32 鈫?129x512:  L1 姣忔鍔犲瘑 脳3.4 鈫?脳3.9 (鈮堜簩闃?
    ///                    L鈭?姣忔鍔犲瘑 脳1.5 鈫?脳1.8 (鈮堜竴闃?鍙楁渶鎵洸鐨勫闈㈠崟鍏冩敮閰?
    /// ```
    ///
    /// 缂?1/V銆佸樊鍥犲瓙 2銆佹硶鍚戠鍙峰啓鍙嶄箣绫荤殑閿欒閮戒細璁╄宸?*涓嶆敹鏁?*,绔嬪埢鏆撮湶銆?    #[test]
    fn linear_field_gradient_converges_under_refinement() {
        let cfg = Config::from_str(include_str!("../config.json")).unwrap();
        let (ax, ay) = (3.7, -1.9);

        let err_at = |rings: usize, nj: usize| -> (f64, f64) {
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
            dom.cells.initialize(&cfg);

            let (ni, njj) = (dom.cells.ni as isize, dom.cells.nj as isize);
            let f = |x: f64, y: f64| ax * x + ay * y;
            for i in 0..ni {
                for j in 0..njj {
                    dom.cells
                        .vx
                        .set(i, j, f(dom.geom.cx.get(i, j), dom.geom.cy.get(i, j)));
                }
            }
            // 鍏充簬杈圭晫闈腑鐐瑰弽灏?浣胯竟鐣岄潰涓婄殑骞冲潎鍊肩簿纭惤鍦ㄨВ鏋愬€间笂
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

            compute(&dom.geom, &mut dom.cells);
            let errs = || {
                dom.cells
                    .grad
                    .interior()
                    .map(|(i, j)| (dom.cells.grad.get(i, j).dudx - ax).abs() / ax.abs())
            };
            let n = (ni * njj) as f64;
            (errs().fold(0.0f64, f64::max), errs().sum::<f64>() / n)
        };

        let (linf_c, l1_c) = err_at(17, 64);
        let (linf_f, l1_f) = err_at(33, 128);
        assert!(
            l1_f < l1_c / 3.0,
            "L1 gradient error not converging at ~2nd order: {l1_c:e} -> {l1_f:e}"
        );
        assert!(
            linf_f < linf_c / 1.4,
            "Linf gradient error not converging: {linf_c:e} -> {linf_f:e}"
        );
    }

    #[test]
    fn ghost_gradients_are_filled() {
        let (_, mut dom) = setup();
        for (i, j) in dom.cells.vx.interior().collect::<Vec<_>>() {
            dom.cells.vx.set(i, j, 10.0 + i as f64 + 0.5 * j as f64);
        }
        compute(&dom.geom, &mut dom.cells);
        let (ni, nj) = (dom.cells.ni as isize, dom.cells.nj as isize);
        let g = &dom.cells.grad;
        for j in 0..nj {
            assert_eq!(g.get(-1, j).dudx, g.get(0, j).dudx);
            assert_eq!(g.get(-1, j).dtdx, 0.0); // 缁濈儹澹?            assert_eq!(g.get(ni, j).dudx, 0.0); // 杩滃満
        }
        for i in 0..ni {
            assert_eq!(g.get(i, -1).dudx, g.get(i, nj - 1).dudx);
            assert_eq!(g.get(i, nj).dudx, g.get(i, 0).dudx);
        }
    }
}

