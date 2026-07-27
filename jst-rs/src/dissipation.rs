//! JST 浜哄伐绮樻€?Jameson-Schmidt-Turkel scalar dissipation)銆?//!
//! 涓績鏍煎紡鏈韩娌℃湁鑰楁暎,闇€瑕佹樉寮忓姞涓婁簩闃?鍥涢樁浜哄伐绮樻€?
//!
//! ```text
//! D_face = 位_f 路 ( 蔚虏 路(U鈧?鈭?U鈧? 鈭?蔚鈦绰?U鈧娾倞 鈭?3U鈧?+ 3U鈧?鈭?U鈧嬧倠) )
//! ```
//!
//! * 鍥涢樁椤规彁渚涜儗鏅樆灏?鎶戝埗涓績鏍煎紡鐨勫鍋惰В鑰?瀹冨浜屾浠ヤ笅鐨勫垎甯冩亽涓?0,
//!   鍥犺€屼笉褰卞搷鏍煎紡绮惧害);
//! * 浜岄樁椤瑰彧鍦ㄦ縺娉㈤檮杩戞墦寮€,鐢卞帇鍔涙帰娴嬪櫒
//!   `谓 = |p鈧?鈭?2p鈧€ + p鈧妡 / (p鈧?+ 2p鈧€ + p鈧?` 瑙﹀彂;
//! * `蔚虏 = k鈧偮穖ax(谓)`(闈袱渚у悇涓や釜鍗曞厓),`蔚鈦?= max(0, k鈧?鈭?蔚虏)` 鈥斺€?婵€娉㈠
//!   鍥涢樁椤硅鍏虫帀,閬垮厤楂橀樁椤瑰湪闂存柇闄勮繎浜х敓鎸崱銆?//!
//! Python 鍩虹嚎鐢?`shockwave_tau[k]`(浠ヨ櫄鎷熷崟鍏?`k鈭?` 涓轰腑蹇?闂存帴琛ㄨ揪鎺㈡祴鍣?
//! 闇€瑕佷竴鏁村鍋忕Щ鎹㈢畻,`BUGS.md` B6 姝ｆ槸杩欓噷绱㈠紩鍐欓噸浜嗐€傝繖閲岀洿鎺ユ妸鎺㈡祴鍣?*瀹氫箟
//! 鍦ㄥ崟鍏冧笂**,鍥涚偣鍙栨渶澶у€煎啓鎴愬绉扮殑 `max(谓[i鈭?..i+1])`,涓嶅啀鏈夊亸绉汇€?
use rayon::iter::ParallelIterator;

use crate::config::Config;
use crate::field::{Field, Vec5};
use crate::geometry::Geometry;
use crate::state::{Cells, Eps, Faces};

/// 鍘嬪姏鎺㈡祴鍣?`谓 = |p鈧?鈭?2p鈧€ + p鈧妡 / (p鈧?+ 2p鈧€ + p鈧?`銆?#[inline(always)]
pub fn pressure_sensor(pm: f64, p0: f64, pp: f64) -> f64 {
    ((pm - 2.0 * p0 + pp) / (pm + 2.0 * p0 + pp)).abs()
}

/// 鐢卞洓鐐规帰娴嬪櫒鏈€澶у€煎緱鍒?`(蔚虏, 蔚鈦?`銆?#[inline(always)]
pub fn adaptive_coefficients(nu_max: f64, k2: f64, k4: f64) -> Eps {
    let e2 = k2 * nu_max;
    Eps { e2, e4: (k4 - e2).max(0.0) }
}

/// 闈㈣氨鍗婂緞 位f 鈥斺€旂敤涓や晶鍗曞厓鐨?`V/螖t_local` 杩戜技,涓?CFL 鎶垫秷鍚庡嵆涓烘€昏氨鍗婂緞銆?///
/// `V/螖t` 鍏堝湪鍗曞厓涓婄畻涓€閬?`NI路NJ` 娆￠櫎娉?,闈笂鍙仛骞冲潎;鐩存帴鍦ㄩ潰寰幆閲岄櫎
/// 浼氬仛绾﹀洓鍊嶇殑闄ゆ硶銆?fn spectral_radii(cfg: &Config, geom: &Geometry, cells: &Cells, faces: &mut Faces) {
    let (ni, nj) = (geom.ni as isize, geom.nj as isize);
    let cfl = cfg.simulation.cfl;
    let (vol, dt) = (&geom.vol, &cells.localdt);

    let Faces { tau, nrm, spec_ratio, .. } = faces;
    spec_ratio.par_interior_rows_mut().for_each(|(i, mut row)| {
        for j in 0..nj {
            row[j] = vol.get(i, j) / dt.get(i, j);
        }
    });
    let ratio = &*spec_ratio;

    tau.lambda.par_interior_rows_mut().for_each(|(i, mut row)| {
        // 杈圭晫闈袱渚у彧鏈変竴涓墿鐞嗗崟鍏?閫€鍖栦负鍗曚晶鍙栧€?        let (a, b) = (i.clamp(0, ni - 1), (i - 1).clamp(0, ni - 1));
        for j in 0..nj {
            row[j] = 0.5 * cfl * (ratio.get(a, j) + ratio.get(b, j));
        }
    });
    nrm.lambda.par_interior_rows_mut().for_each(|(i, mut row)| {
        for j in 0..nj {
            let jm = if j > 0 { j - 1 } else { nj - 1 };
            row[j] = 0.5 * cfl * (ratio.get(i, j) + ratio.get(i, jm));
        }
    });
}

/// 鍗曞厓涓婄殑鍘嬪姏鎺㈡祴鍣?鑼冨洿瑕嗙洊鍒拌櫄鎷熷眰(鍥涚偣鍙栨渶澶у€奸渶瑕?`[-2, N+1]`)銆?fn sensors(geom: &Geometry, cells: &Cells, faces: &mut Faces) {
    let (ni, nj) = (geom.ni as isize, geom.nj as isize);
    let p = &cells.p;
    for i in -2..=ni + 1 {
        for j in 0..nj {
            faces
                .sensor_i
                .set(i, j, pressure_sensor(p.get(i - 1, j), p.get(i, j), p.get(i + 1, j)));
        }
    }
    for i in 0..ni {
        for j in -2..=nj {
            faces
                .sensor_j
                .set(i, j, pressure_sensor(p.get(i, j - 1), p.get(i, j), p.get(i, j + 1)));
        }
    }
}

/// 鑷€傚簲鑰楁暎绯绘暟 `蔚虏`銆乣蔚鈦碻銆?fn coefficients(cfg: &Config, geom: &Geometry, faces: &mut Faces) {
    let (k2, k4) = (cfg.dissipation.k2, cfg.dissipation.k4);
    let nj = geom.nj as isize;
    let Faces {
        tau,
        nrm,
        sensor_i,
        sensor_j,
        ..
    } = faces;
    let (sensor_i, sensor_j) = (&*sensor_i, &*sensor_j);

    tau.eps.par_interior_rows_mut().for_each(|(i, mut row)| {
        for j in 0..nj {
            // 鍏充簬闈?i 瀵圭О鐨勫洓鐐规ā鏉?鍗曞厓 i鈭?, i鈭?, i, i+1
            let m = sensor_i
                .get(i - 2, j)
                .max(sensor_i.get(i - 1, j))
                .max(sensor_i.get(i, j))
                .max(sensor_i.get(i + 1, j));
            row[j] = adaptive_coefficients(m, k2, k4);
        }
    });
    nrm.eps.par_interior_rows_mut().for_each(|(i, mut row)| {
        for j in 0..nj {
            let m = sensor_j
                .get(i, j - 2)
                .max(sensor_j.get(i, j - 1))
                .max(sensor_j.get(i, j))
                .max(sensor_j.get(i, j + 1));
            row[j] = adaptive_coefficients(m, k2, k4);
        }
    });
}

/// 闈笂鐨?JST 鑰楁暎椤广€?fn face_dissipation(geom: &Geometry, cells: &Cells, faces: &mut Faces) {
    let nj = geom.nj as isize;
    let u = &cells.u;
    let Faces { tau, nrm, .. } = faces;

    {
        let FaceWorkSplit { lambda, eps, out } = FaceWorkSplit {
            lambda: &tau.lambda,
            eps: &tau.eps,
            out: &mut tau.dissipation,
        };
        out.par_interior_rows_mut().for_each(|(i, mut row)| {
            for j in 0..nj {
                let d1 = u.get(i, j) - u.get(i - 1, j);
                let d3 = u.get(i + 1, j) - 3.0 * u.get(i, j) + 3.0 * u.get(i - 1, j)
                    - u.get(i - 2, j);
                let e = eps.get(i, j);
                row[j] = lambda.get(i, j) * (d1 * e.e2 - d3 * e.e4);
            }
        });
    }
    {
        let FaceWorkSplit { lambda, eps, out } = FaceWorkSplit {
            lambda: &nrm.lambda,
            eps: &nrm.eps,
            out: &mut nrm.dissipation,
        };
        out.par_interior_rows_mut().for_each(|(i, mut row)| {
            for j in 0..nj {
                let d1 = u.get(i, j) - u.get(i, j - 1);
                let d3 = u.get(i, j + 1) - 3.0 * u.get(i, j) + 3.0 * u.get(i, j - 1)
                    - u.get(i, j - 2);
                let e = eps.get(i, j);
                row[j] = lambda.get(i, j) * (d1 * e.e2 - d3 * e.e4);
            }
        });
    }
}

/// 鎶?[`crate::state::FaceWork`] 鎷嗘垚"璇昏嫢骞?+ 鍐欎竴涓?鐨勪笉鐩镐氦鍊熺敤銆?struct FaceWorkSplit<'a> {
    lambda: &'a Field<f64>,
    eps: &'a Field<Eps>,
    out: &'a mut Field<Vec5>,
}

/// 鍗曞厓涓婄殑浜哄伐绮樻€х幆閲忋€?fn assemble(geom: &Geometry, faces: &Faces, out: &mut Field<Vec5>) {
    let nj = geom.nj as isize;
    let (tau, nrm) = (&faces.tau.dissipation, &faces.nrm.dissipation);
    out.par_interior_rows_mut().for_each(|(i, mut row)| {
        for j in 0..nj {
            let jp1 = if j + 1 < nj { j + 1 } else { 0 };
            row[j] = tau.get(i + 1, j) + nrm.get(i, jp1) - tau.get(i, j) - nrm.get(i, j);
        }
    });
}

/// 涓€娆″畬鏁寸殑 JST 浜哄伐绮樻€ц绠椼€?pub fn compute(cfg: &Config, geom: &Geometry, cells: &mut Cells, faces: &mut Faces) {
    spectral_radii(cfg, geom, cells, faces);
    sensors(geom, cells, faces);
    coefficients(cfg, geom, faces);
    face_dissipation(geom, cells, faces);
    assemble(geom, faces, &mut cells.fd);
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
        crate::timestep::compute(&cfg, &dom.geom, &mut dom.cells);
        (cfg, dom)
    }

    #[test]
    fn sensor_is_zero_in_smooth_flow() {
        assert_eq!(pressure_sensor(2.0, 2.0, 2.0), 0.0);
    }

    #[test]
    fn sensor_detects_a_jump() {
        assert!((pressure_sensor(1.0, 1.0, 3.0) - 2.0 / 6.0).abs() < 1e-15);
    }

    #[test]
    fn sensor_is_bounded_by_one() {
        for &(a, b, c) in &[(1.0, 1e-6, 1.0), (1.0, 1.0, 1e6), (1e-9, 5.0, 1e-9)] {
            let v = pressure_sensor(a, b, c);
            assert!((0.0..=1.0).contains(&v), "sensor {v} out of [0,1]");
        }
    }

    /// 寮烘縺娉㈠ 蔚虏 澧炲ぇ銆佄碘伌 琚叧鎺夈€?    #[test]
    fn fourth_order_coefficient_switches_off_at_a_shock() {
        let (k2, k4) = (0.5, 0.0078125);
        let smooth = adaptive_coefficients(0.0, k2, k4);
        assert_eq!(smooth.e2, 0.0);
        assert_eq!(smooth.e4, k4);
        let shock = adaptive_coefficients(0.9, k2, k4);
        assert!((shock.e2 - 0.45).abs() < 1e-18);
        assert_eq!(shock.e4, 0.0);
    }

    #[test]
    fn sensor_fires_across_a_pressure_jump_in_the_field() {
        let (cfg, mut dom) = setup();
        for j in 0..dom.cells.nj as isize {
            dom.cells.p.set(2, j, dom.cells.p.get(2, j) * 4.0);
        }
        sensors(&dom.geom, &dom.cells, &mut dom.faces);
        coefficients(&cfg, &dom.geom, &mut dom.faces);
        let fired = dom
            .faces
            .tau
            .eps
            .interior()
            .any(|(i, j)| dom.faces.tau.eps.get(i, j).e2 > cfg.dissipation.k4);
        assert!(fired, "expected the sensor to fire near the jump");
    }

    #[test]
    fn uniform_flow_gets_no_dissipation() {
        let (cfg, mut dom) = setup();
        dom.cells.set_uniform(&cfg, 1.176, 69.4, 17.3, 101325.0, 1.5e-4);
        compute(&cfg, &dom.geom, &mut dom.cells, &mut dom.faces);
        let lam = dom
            .faces
            .tau
            .lambda
            .interior()
            .map(|(i, j)| dom.faces.tau.lambda.get(i, j))
            .fold(0.0f64, f64::max);
        let umax = dom
            .cells
            .u
            .interior()
            .map(|(i, j)| dom.cells.u.get(i, j).amax())
            .fold(0.0f64, f64::max);
        let scale = lam * cfg.dissipation.k4 * umax;
        for (i, j) in dom.cells.fd.interior() {
            assert!(
                dom.cells.fd.get(i, j).amax() < 1e-13 * scale,
                "Fd nonzero at ({i},{j})"
            );
        }
    }

    /// 鍏夋粦娴佸満閲屼簩闃?婵€娉?鑰楁暎搴斿綋鍏抽棴,鍙墿鍥涢樁鑳屾櫙鑰楁暎銆?    #[test]
    fn smooth_flow_leaves_only_background_dissipation() {
        let (cfg, mut dom) = setup();
        compute(&cfg, &dom.geom, &mut dom.cells, &mut dom.faces);
        for (i, j) in dom.faces.tau.eps.interior() {
            let e = dom.faces.tau.eps.get(i, j);
            // 杩滃満杈圭晫鍘嬪姏鐢遍粠鏇兼眰瑙ｇ粰鍑?涓?p鈭?鍙埌 ~1e-11 鐩稿绮惧害,
            // 鍥犳鎺㈡祴鍣ㄦ湁涓€涓瀬灏忕殑鏈簳,鑰岄潪绮剧‘ 0
            assert!(e.e2 < 1e-12 * cfg.dissipation.k4, "eps2 = {:e} at ({i},{j})", e.e2);
            assert!((e.e4 - cfg.dissipation.k4).abs() < 1e-12 * cfg.dissipation.k4);
        }
    }

    /// 鍥涢樁椤瑰浜屾浠ヤ笅鍒嗗竷鎭掍负 0 鈥斺€?瀹冧笉鐮村潖鏍煎紡绮惧害鐨勫叧閿€?    #[test]
    fn fourth_difference_annihilates_quadratics() {
        let q = |x: f64| 3.0 * x * x + 2.0 * x + 5.0;
        let d3 = q(1.0) - 3.0 * q(0.0) + 3.0 * q(-1.0) - q(-2.0);
        assert!(d3.abs() < 1e-12, "d3 = {d3}");
    }

    /// 鍥涢樁椤瑰涓夋鍒嗗竷缁欏嚭甯告暟(鍗冲畠纭疄鏄笁闃跺樊鍒嗙畻瀛?銆?    #[test]
    fn fourth_difference_of_a_cubic_is_constant() {
        let c = |x: f64| 2.0 * x * x * x;
        let d3 = c(1.0) - 3.0 * c(0.0) + 3.0 * c(-1.0) - c(-2.0);
        assert!((d3 - 2.0 * 6.0).abs() < 1e-12, "d3 = {d3}");
    }
}

