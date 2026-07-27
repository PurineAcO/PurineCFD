//! 绮樻€у簲鍔涖€佺儹娴佷笌 Spalart-Allmaras 婀嶆祦鎵╂暎銆?//!
//! 姣忎釜鍗曞厓鍏堢畻鍑烘墿鏁ｅ紶閲忕殑涓ゅ垪 `(D_x, D_y)`:
//!
//! ```text
//! D_x = [0, 蟿xx, 蟿xy, u路蟿xx + v路蟿xy + qx, 蟽鈦宦?渭+蟻谓虄)路鈭偽教?鈭倄]
//! D_y = [0, 蟿xy, 蟿yy, u路蟿xy + v路蟿yy + qy, 蟽鈦宦?渭+蟻谓虄)路鈭偽教?鈭倅]
//! ```
//!
//! 鍏朵腑 蟿 鐢?Stokes 鍋囪,`q = 位_eff路cp路鈭嘥`(绗﹀彿宸插苟鍏ユ€婚€氶噺,鍗充唬琛?//! `+k鈭俆/鈭倄`),`位_eff = 渭/Pr + 渭t/Prt`銆?//!
//! 闈笂鍙栫浉閭讳袱鍗曞厓鐨勭畻鏈钩鍧囧悗鐐逛箻娉曞悜,鍐嶅湪鍗曞厓涓婃眰鐜噺銆?
use rayon::iter::{IndexedParallelIterator, ParallelIterator};

use crate::config::Config;
use crate::field::Vec5;
use crate::geometry::Geometry;
use crate::state::{Cells, DiffTensor, Faces, Grad, TurbAux};

/// S-A 闃诲凹鍑芥暟 `fv1 = 蠂鲁/(蠂鲁 + Cv1鲁)`銆?///
/// 鍒嗘瘝閲岀殑 `Cv1` 蹇呴』鍙栫珛鏂?鈥斺€?Python 鍩虹嚎婕忎簡瀹?`BUGS.md` B2),
/// 浣胯繎澹侀樆灏艰涓ラ噸鍓婂急銆?#[inline(always)]
pub fn fv1(chi: f64, cv1_cubed: f64) -> f64 {
    let c3 = chi * chi * chi;
    c3 / (c3 + cv1_cubed)
}

/// 鍗曠偣鐨勭矘鎬?婀嶆祦鎵╂暎寮犻噺銆?///
/// 鍐欐垚绾爣閲忓嚱鏁?涓嶆帴瑙︿换浣曟暟缁?鏃㈡柟渚垮崟娴?涔熻璋冪敤鏂瑰彲浠ヨ嚜鐢辨媶鍊熸暟缁勩€?#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub fn diffusion_tensor(
    cfg: &Config,
    t: f64,
    rho_nu: f64,
    vx: f64,
    vy: f64,
    g: &Grad,
) -> (TurbAux, DiffTensor) {
    let sa = &cfg.spalart_allmaras;
    let d = &cfg.derived;

    let mu = cfg.mu(t);
    let chi = rho_nu / mu;
    let f1 = fv1(chi, d.cv1_cubed);
    let mu_t = rho_nu * f1;
    let mu_eff = mu + mu_t;
    let lam_eff = mu * d.inv_pr + mu_t * d.inv_prt;

    // Stokes 鍋囪涓嬬殑绮樻€у簲鍔?    let txx = mu_eff * (4.0 / 3.0 * g.dudx - 2.0 / 3.0 * g.dvdy);
    let tyy = mu_eff * (4.0 / 3.0 * g.dvdy - 2.0 / 3.0 * g.dudx);
    let txy = mu_eff * (g.dudy + g.dvdx);

    let qx = lam_eff * d.cp * g.dtdx;
    let qy = lam_eff * d.cp * g.dtdy;
    let nu_diff = sa.sigma * (mu + rho_nu);

    (
        TurbAux { mu, chi, fv1: f1 },
        DiffTensor {
            x: Vec5::new(0.0, txx, txy, vx * txx + vy * txy + qx, nu_diff * g.dnutdx),
            y: Vec5::new(0.0, txy, tyy, vx * txy + vy * tyy + qy, nu_diff * g.dnutdy),
        },
    )
}

/// 鍦ㄧ墿鐞嗗崟鍏?+ 绗竴灞傝櫄鎷熷崟鍏冧笂瑁呴厤鎵╂暎寮犻噺銆?fn cell_tensors(cfg: &Config, cells: &mut Cells) {
    let (ni, nj) = (cells.ni as isize, cells.nj as isize);
    let Cells {
        aux,
        diff,
        t,
        u,
        vx,
        vy,
        grad,
        ..
    } = cells;
    let (t, u, vx, vy, grad) = (&*t, &*u, &*vx, &*vy, &*grad);

    let eval = |i: isize, j: isize| {
        diffusion_tensor(
            cfg,
            t.get(i, j),
            u.get(i, j)[crate::field::comp::RHO_NU],
            vx.get(i, j),
            vy.get(i, j),
            &grad.get(i, j),
        )
    };

    // 鐗╃悊鍗曞厓 鈥斺€?鎸夎骞惰,涓や釜杈撳嚭鏁扮粍鍚屾鍒囧垎
    aux.par_interior_rows_mut()
        .zip(diff.par_interior_rows_mut())
        .for_each(|((i, mut a_row), (_, mut d_row))| {
            for j in 0..nj {
                let (a, d) = eval(i, j);
                a_row[j] = a;
                d_row[j] = d;
            }
        });

    // 绗竴灞傝櫄鎷熷崟鍏?鏁伴噺 O(NI+NJ),涓茶鍗冲彲)
    for j in 0..nj {
        for i in [-1, ni] {
            let (a, d) = eval(i, j);
            aux.set(i, j, a);
            diff.set(i, j, d);
        }
    }
    for i in 0..ni {
        for j in [-1, nj] {
            let (a, d) = eval(i, j);
            aux.set(i, j, a);
            diff.set(i, j, d);
        }
    }
}

/// 闈笂鐨勬墿鏁ｉ€氶噺 `陆(D_a + D_b)路n`銆?fn face_diffusion(geom: &Geometry, cells: &Cells, faces: &mut Faces) {
    let nj = geom.nj as isize;
    let d = &cells.diff;

    let tau_geom = &geom.tau;
    faces.tau.diff.par_interior_rows_mut().for_each(|(i, mut row)| {
        for j in 0..nj {
            let g = tau_geom.at(i, j);
            let (a, b) = (d.get(i - 1, j), d.get(i, j));
            row[j] = (a.x + b.x) * (0.5 * g.nx) + (a.y + b.y) * (0.5 * g.ny);
        }
    });

    let n_geom = &geom.nrm;
    faces.nrm.diff.par_interior_rows_mut().for_each(|(i, mut row)| {
        for j in 0..nj {
            let g = n_geom.at(i, j);
            let (a, b) = (d.get(i, j - 1), d.get(i, j));
            row[j] = (a.x + b.x) * (0.5 * g.nx) + (a.y + b.y) * (0.5 * g.ny);
        }
    });
}

/// 涓€娆″畬鏁寸殑绮樻€?婀嶆祦鎵╂暎椤硅绠椼€?pub fn compute(cfg: &Config, geom: &Geometry, cells: &mut Cells, faces: &mut Faces) {
    cell_tensors(cfg, cells);
    face_diffusion(geom, cells, faces);

    let nj = geom.nj as isize;
    let (tau, nrm) = (&faces.tau.diff, &faces.nrm.diff);
    cells.fv.par_interior_rows_mut().for_each(|(i, mut row)| {
        for j in 0..nj {
            let jp1 = if j + 1 < nj { j + 1 } else { 0 };
            row[j] = tau.get(i + 1, j) - tau.get(i, j) + nrm.get(i, jp1) - nrm.get(i, j);
        }
    });
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

    fn sample(cfg: &Config, dom: &Domain, g: Grad) -> (TurbAux, DiffTensor) {
        diffusion_tensor(
            cfg,
            dom.cells.t.get(0, 0),
            dom.cells.u.get(0, 0)[crate::field::comp::RHO_NU],
            dom.cells.vx.get(0, 0),
            dom.cells.vy.get(0, 0),
            &g,
        )
    }

    #[test]
    fn fv1_asymptotes() {
        let cv1c = 7.1f64.powi(3);
        assert!(fv1(1e-8, cv1c).abs() < 1e-20);
        assert!((fv1(1e6, cv1c) - 1.0).abs() < 1e-12);
        // 涓棿鍊煎繀椤荤敤 Cv1鲁 鑰岄潪 Cv1
        assert!((fv1(3.0, cv1c) - 27.0 / (27.0 + 357.911)).abs() < 1e-9);
    }

    #[test]
    fn stress_is_symmetric_and_traceless_for_solenoidal_field() {
        let (cfg, dom) = setup();
        let g = Grad {
            dudx: 2.0,
            dvdy: -2.0, // 鈭嚶穟 = 0
            dudy: 5.0,
            dvdx: -3.0,
            ..Default::default()
        };
        let (_, d) = sample(&cfg, &dom, g);
        assert!((d.x[2] - d.y[1]).abs() < 1e-20, "tau_xy asymmetric");
        assert!((d.x[1] + d.y[2]).abs() < 1e-20, "trace nonzero");
    }

    #[test]
    fn trace_tracks_dilatation() {
        let (cfg, dom) = setup();
        let g = Grad { dudx: 2.0, dvdy: 7.0, ..Default::default() }; // 鈭嚶穟 = 9
        let (a, d) = sample(&cfg, &dom, g);
        let mu_eff = a.mu + dom.cells.u.get(0, 0)[4] * a.fv1;
        assert!((d.x[1] + d.y[2] - 2.0 / 3.0 * mu_eff * 9.0).abs() < 1e-18);
    }

    #[test]
    fn continuity_equation_has_no_diffusion() {
        let (cfg, dom) = setup();
        let g = Grad { dudx: 2.0, dtdx: 11.0, ..Default::default() };
        let (_, d) = sample(&cfg, &dom, g);
        assert_eq!(d.x[0], 0.0);
        assert_eq!(d.y[0], 0.0);
    }

    #[test]
    fn chi_is_the_viscosity_ratio() {
        let (cfg, dom) = setup();
        let (a, _) = sample(&cfg, &dom, Grad::default());
        let want = dom.cells.rho.get(0, 0) * dom.cells.nut.get(0, 0) / a.mu;
        assert!((a.chi - want).abs() < 1e-12 * want);
    }

    #[test]
    fn uniform_field_has_no_viscous_flux() {
        let (cfg, mut dom) = setup();
        dom.cells.set_uniform(&cfg, 1.176, 69.4, 17.3, 101325.0, 1.5e-4);
        crate::gradient::compute(&dom.geom, &mut dom.cells);
        compute(&cfg, &dom.geom, &mut dom.cells, &mut dom.faces);
        let t = cfg.simulation.t_inf;
        let mu = cfg.mu(t);
        let h = dom.geom.vol.get(0, 0).sqrt();
        let scale =
            (mu / cfg.spalart_allmaras.Pr + mu / cfg.spalart_allmaras.Prt) * cfg.derived.cp * t / h;
        for (i, j) in dom.cells.fv.interior() {
            let f = dom.cells.fv.get(i, j);
            for k in 0..4 {
                assert!(f[k].abs() < 1e-12 * scale, "Fv[{k}] = {:e} at ({i},{j})", f[k]);
            }
        }
    }

    /// 鎵╂暎寮犻噺蹇呴』鍦ㄧ涓€灞傝櫄鎷熷崟鍏冧笂涔熺畻鍑烘潵 鈥斺€?杈圭晫闈㈢殑骞冲潎瑕佺敤鍒般€?    #[test]
    fn ghost_layer_tensors_are_populated() {
        let (cfg, mut dom) = setup();
        for (i, j) in dom.cells.grad.interior().collect::<Vec<_>>() {
            dom.cells.grad.set(i, j, Grad { dudy: 100.0, ..Default::default() });
        }
        crate::gradient::compute(&dom.geom, &mut dom.cells);
        cell_tensors(&cfg, &mut dom.cells);
        let (ni, nj) = (dom.cells.ni as isize, dom.cells.nj as isize);
        for j in 0..nj {
            assert!(dom.cells.aux.get(-1, j).mu > 0.0, "wall ghost aux unset at j={j}");
            assert!(dom.cells.aux.get(ni, j).mu > 0.0, "far ghost aux unset at j={j}");
        }
        for i in 0..ni {
            assert!(dom.cells.aux.get(i, -1).mu > 0.0);
            assert!(dom.cells.aux.get(i, nj).mu > 0.0);
        }
    }
}

