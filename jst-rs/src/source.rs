//! Spalart-Allmaras 涓€鏂圭▼婀嶆祦妯″瀷鐨勬簮椤广€?//!
//! ```text
//! S = P 鈭?D + G
//! P = Cb1路(1 鈭?ft2)路S虄路蟻谓虄                        鐢熸垚
//! D = (Cw1路fw 鈭?Cb1/魏虏路ft2)路蟻路(谓虄/d)虏             澹侀潰鐮村潖
//! G = (Cb2/蟽)路蟻路|鈭囄教億虏                            闈炲畧鎭掓墿鏁?//! ```
//!
//! 鍏朵腑 `S虄 = S + 谓虄/(魏虏d虏)路fv2`,`S = |蠅|` 鏄丁閲忔ā銆?//!
//! 婧愰」鍙綔鐢ㄥ湪绗?5 涓柟绋嬩笂,骞冲潎娴佹柟绋嬫棤婧愩€?
use rayon::iter::ParallelIterator;

use crate::config::Config;
use crate::field::comp;
use crate::geometry::Geometry;
use crate::state::Cells;

/// `S虄` 鐨勪笅闄愩€侫llmaras (2012) 寤鸿瀵逛慨姝ｆ丁閲忓仛鎴柇,鍚﹀垯 `fv2 < 0` 鏃?/// `S虄` 鍙兘杩囬浂,`r = 谓虄/(S虄魏虏d虏)` 闄ら浂鍙戞暎銆?const S_TILDE_FLOOR: f64 = 1e-10;

/// 澹侀潰闃诲凹鍑芥暟 `fw = g路[(1+Cw3鈦?/(g鈦?Cw3鈦?]^(1/6)`,`g = r + Cw2(r鈦垛垝r)`銆?///
/// `Cw3鈦禶 涓?`1+Cw3鈦禶 鐢辫皟鐢ㄦ柟棰勫厛绠楀ソ(瀹冧滑鍙緷璧栭厤缃?;`x^(1/6)` 鎷嗘垚
/// `鈭涒垰x` 鈥斺€?`sqrt` 涓?`cbrt` 閮芥瘮閫氱敤鐨?`powf` 蹇€?#[inline(always)]
pub fn fw(r: f64, cw2: f64, cw3_6: f64, one_plus_cw3_6: f64) -> f64 {
    let g = r + cw2 * (r.powi(6) - r);
    g * (one_plus_cw3_6 / (g.powi(6) + cw3_6)).sqrt().cbrt()
}

/// 浜岀淮娑￠噺妯?`S = 鈭?2惟岬⑩奔惟岬⑩奔) = |鈭倂/鈭倄 鈭?鈭倁/鈭倅|`銆?///
/// Python 鍩虹嚎鍙栫殑鏄?`陆(鈭倁/鈭倅 鈭?鈭倂/鈭倄)` 鍐嶄箻 鈭?,鏃㈠樊 鈭? 鍊嶅張鍙嶄簡鍙?/// (`BUGS.md` B3)銆?#[inline(always)]
pub fn vorticity_magnitude(dvdx: f64, dudy: f64) -> f64 {
    (dvdx - dudy).abs()
}

/// 璁＄畻鍏ㄩ儴鐗╃悊鍗曞厓鐨?S-A 婧愰」(宸蹭箻鍗曞厓浣撶Н)銆?pub fn compute(cfg: &Config, geom: &Geometry, cells: &mut Cells) {
    let sa = &cfg.spalart_allmaras;
    let d = &cfg.derived;
    let nj = geom.nj as isize;
    // 寰幆涓嶅彉閲忓叏閮ㄦ彁鍒板闈?    let cb1_over_kappa2 = sa.Cb1 * d.inv_kappa2;
    let cb2_sigma = sa.Cb2 * sa.sigma;

    let Cells { src, u, aux, grad, .. } = cells;
    let (u, aux, grad) = (&*u, &*aux, &*grad);
    let (vol, inv_d2) = (&geom.vol, &geom.inv_wall_dist_sq);

    src.par_interior_rows_mut().for_each(|(i, mut row)| {
        for j in 0..nj {
            let uc = u.get(i, j);
            let rho = uc[comp::RHO];
            let rho_nu = uc[comp::RHO_NU];
            let nu_tilde = rho_nu / rho;
            let inv_dist2 = inv_d2.get(i, j);
            let inv_kd2 = inv_dist2 * d.inv_kappa2;

            let a = aux.get(i, j);
            let g = grad.get(i, j);
            let ft2 = sa.Ct3 * (-sa.Ct4 * a.chi * a.chi).exp();
            let fv2 = 1.0 - a.chi / (1.0 + a.chi * a.fv1);

            let s = sa.fv3 * vorticity_magnitude(g.dvdx, g.dudy);
            let s_tilde = (s + nu_tilde * inv_kd2 * fv2).max(S_TILDE_FLOOR);

            // 鐢熸垚椤?            let prod = sa.Cb1 * (1.0 - ft2) * s_tilde * rho_nu;
            // 澹侀潰鐮村潖椤?            let r = (nu_tilde / s_tilde * inv_kd2).min(sa.rmax);
            let destr = (d.cw1 * fw(r, sa.Cw2, d.cw3_6, d.one_plus_cw3_6)
                - cb1_over_kappa2 * ft2)
                * rho
                * (nu_tilde * nu_tilde * inv_dist2);
            // 闈炲畧鎭掓墿鏁ｉ」
            let diff = cb2_sigma * rho * (g.dnutdx * g.dnutdx + g.dnutdy * g.dnutdy);

            row[j] = (prod - destr + diff) * vol.get(i, j);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::Mesh;
    use crate::state::{Domain, Grad, TurbAux};

    fn setup() -> (Config, Domain) {
        let cfg = Config::from_str(include_str!("../config.json")).unwrap();
        let mesh = Mesh::parse(include_str!("../fangdata.txt")).unwrap();
        let geom = Geometry::build(&mesh, cfg.simulation.halo);
        let mut dom = Domain::new(geom, cfg.simulation.halo);
        dom.cells.initialize(&cfg);
        crate::boundary::apply(&cfg, &dom.geom, &mut dom.cells);
        crate::gradient::compute(&dom.geom, &mut dom.cells);
        crate::viscous::compute(&cfg, &dom.geom, &mut dom.cells, &mut dom.faces);
        (cfg, dom)
    }

    #[test]
    fn vorticity_is_the_curl() {
        assert!((vorticity_magnitude(11.0, 3.0) - 8.0).abs() < 1e-15);
        assert!((vorticity_magnitude(3.0, 11.0) - 8.0).abs() < 1e-15);
        assert_eq!(vorticity_magnitude(5.0, 5.0), 0.0);
    }

    const CW3_6: f64 = 64.0; // 2鈦?    const ONE_PLUS: f64 = 65.0;

    #[test]
    fn fw_is_bounded_on_the_admissible_range() {
        for k in 0..=200 {
            let r = 10.0 * k as f64 / 200.0;
            let v = fw(r, 0.3, CW3_6, ONE_PLUS);
            assert!(v.is_finite() && (0.0..10.0).contains(&v), "fw({r}) = {v}");
        }
    }

    #[test]
    fn fw_is_unity_at_r_equals_one() {
        // r = 1 鈬?g = 1 鈬?fw = ((1+Cw3鈦?/(1+Cw3鈦?)^(1/6) = 1
        assert!((fw(1.0, 0.3, CW3_6, ONE_PLUS) - 1.0).abs() < 1e-14);
    }

    /// `鈭涒垰x` 蹇呴』涓?`x^(1/6)` 鍦ㄦ暟鍊间笂涓€鑷?鎬ц兘鏀瑰啓涓嶅緱鏀瑰彉缁撴灉)銆?    #[test]
    fn fw_matches_the_powf_formulation() {
        for k in 1..=200 {
            let r = 10.0 * k as f64 / 200.0;
            let g = r + 0.3 * (r.powi(6) - r);
            let want = g * (ONE_PLUS / (g.powi(6) + CW3_6)).powf(1.0 / 6.0);
            let got = fw(r, 0.3, CW3_6, ONE_PLUS);
            assert!((got - want).abs() <= 1e-14 * want.abs(), "fw({r}): {got} vs {want}");
        }
    }

    #[test]
    fn source_is_finite_everywhere() {
        let (cfg, mut dom) = setup();
        compute(&cfg, &dom.geom, &mut dom.cells);
        for (i, j) in dom.cells.src.interior() {
            assert!(dom.cells.src.get(i, j).is_finite(), "source NaN at ({i},{j})");
        }
    }

    /// 闆舵丁閲忋€侀浂 谓虄 鐨勬瀬闄?`S虄` 鐨勪笅闄愭埅鏂繀椤婚槻浣忛櫎闆躲€?    #[test]
    fn source_is_finite_at_zero_vorticity_and_zero_nut() {
        let (cfg, mut dom) = setup();
        for (i, j) in dom.cells.u.interior().collect::<Vec<_>>() {
            let mut u = dom.cells.u.get(i, j);
            u[comp::RHO_NU] = 0.0;
            dom.cells.u.set(i, j, u);
            dom.cells.grad.set(i, j, Grad::default());
            dom.cells.aux.set(i, j, TurbAux { mu: 1.8e-5, chi: 0.0, fv1: 0.0 });
        }
        compute(&cfg, &dom.geom, &mut dom.cells);
        for (i, j) in dom.cells.src.interior() {
            assert!(dom.cells.src.get(i, j).is_finite());
        }
    }

    /// 婧愰」閲忕骇闅忔丁閲忓澶?鈥斺€?鐢熸垚椤?鈭?S虄,鏄ā鍨嬬殑鍩烘湰瀹氭爣琛屼负銆?    ///
    /// 杩欓噷涓嶈兘鏂█婧愰」**鍙樻**:浣?蠂 鏃?`ft2 = Ct3路exp(鈭扖t4蠂虏) 鈫?1.2 > 1`,
    /// 浜庢槸 `Cb1(1鈭抐t2)S虄蟻谓虄 < 0`(S-A 鍘熷寮忎腑鐨勮浆鎹╂姂鍒堕」)銆傝繖鏄爣鍑嗚涓?
    /// 璁稿瀹炵幇骞茶剢閲囩敤 SA-noft2 鍙樹綋鎶婂畠鍘绘帀銆?    #[test]
    fn source_magnitude_scales_with_vorticity() {
        let (cfg, mut dom) = setup();
        let mut sample = |w: f64| {
            for (i, j) in dom.cells.grad.interior().collect::<Vec<_>>() {
                let mut g = dom.cells.grad.get(i, j);
                g.dvdx = w;
                g.dudy = 0.0;
                dom.cells.grad.set(i, j, g);
            }
            compute(&cfg, &dom.geom, &mut dom.cells);
            dom.cells.src.get(0, 0)
        };
        let lo = sample(1.0);
        let hi = sample(100.0);
        assert!(
            hi.abs() > 10.0 * lo.abs(),
            "source did not scale with vorticity: {lo:e} -> {hi:e}"
        );
        assert!(lo.signum() == hi.signum(), "production changed sign unexpectedly");
    }

    /// 鐢熸垚椤圭‘瀹炴姣斾簬 `(1 鈭?ft2)`:鎶?蠂 璋冨ぇ浣?ft2 鈫?0 鍚?婧愰」搴旇浆涓烘銆?    #[test]
    fn production_is_positive_at_high_eddy_viscosity() {
        let (cfg, mut dom) = setup();
        for (i, j) in dom.cells.u.interior().collect::<Vec<_>>() {
            // 蠂 = 20 鈬?ft2 = 1.2路e^{鈭?00} 鈮?0
            let mu = dom.cells.aux.get(i, j).mu;
            let mut u = dom.cells.u.get(i, j);
            u[comp::RHO_NU] = 20.0 * mu;
            dom.cells.u.set(i, j, u);
            dom.cells.aux.set(
                i,
                j,
                TurbAux {
                    mu,
                    chi: 20.0,
                    fv1: crate::viscous::fv1(20.0, cfg.derived.cv1_cubed),
                },
            );
            let mut g = dom.cells.grad.get(i, j);
            g.dvdx = 500.0;
            g.dudy = 0.0;
            dom.cells.grad.set(i, j, g);
        }
        compute(&cfg, &dom.geom, &mut dom.cells);
        assert!(dom.cells.src.get(0, 0) > 0.0, "expected net production");
    }
}

