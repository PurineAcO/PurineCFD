//! 鏃犵矘瀵规祦閫氶噺銆?//!
//! 涓夋:闈笂鐨勫畧鎭掗噺(鐩搁偦鍗曞厓涓€闃朵腑蹇冨钩鍧?鈫?闈笂鐨?Euler 閫氶噺 `F路n` 鈫?//! 鍗曞厓鐜噺銆傚洜涓鸿櫄鎷熷崟鍏冨凡鐢?[`crate::boundary`] 濉ソ,杩欓噷涓変釜寰幆閮芥槸
//! 绾煩褰€侀浂鐗瑰垽銆?//!
//! 鍗曞厓鐜噺鐨勭鍙风害瀹氫笌 [`crate::geometry`] 鐨勬硶鍚戝畾涔変竴鑷?娉曞悜鎸囧悜 i銆乯 澧炲ぇ
//! 鐨勬柟鍚?,浜庢槸
//!
//! ```text
//! Fc(i,j) = F_蟿(i+1) 鈭?F_蟿(i) + F_n(j+1) 鈭?F_n(j)
//! ```
//!
//! 鐢卞害閲忛棴鍚?`危卤n 鈮?0`,鍧囧寑娴佷笅 `Fc 鈮?0`(鑷敱鏉ユ祦淇濇寔鎬?銆?
use rayon::iter::ParallelIterator;

use crate::config::Config;
use crate::field::{comp, Field, Vec5};
use crate::geometry::{FaceGeom, Geometry};
use crate::state::{Cells, Faces};

/// 鐢遍潰涓婄殑瀹堟亽閲忕畻 Euler 閫氶噺 `F路n`(娉曞悜宸插惈闈㈢Н鏉?銆?#[inline(always)]
pub fn euler_flux(fu: Vec5, face: &FaceGeom, gamma: f64) -> Vec5 {
    let rho = fu[comp::RHO];
    let inv_rho = 1.0 / rho;
    let u = fu[comp::MX] * inv_rho;
    let v = fu[comp::MY] * inv_rho;
    let rho_e = fu[comp::RHO_E];
    let p = (gamma - 1.0) * (rho_e - rho * (u * u + v * v) * 0.5);
    let vn = face.nx * u + face.ny * v;
    Vec5::new(
        rho * vn,
        fu[comp::MX] * vn + p * face.nx,
        fu[comp::MY] * vn + p * face.ny,
        (rho_e + p) * vn,
        fu[comp::RHO_NU] * vn,
    )
}

/// 闈笂鐨勬棤绮橀€氶噺,tau 闈笌 n 闈㈠悇涓€閬嶃€?///
/// 闈笂鐨勫畧鎭掗噺鍙槸涓棿鍊?鐢ㄥ眬閮ㄥ彉閲忕畻鎺夊嵆鍙?鈥斺€?涓嶅繀鍍?Python 閭ｆ牱鍐嶅瓨涓€涓?/// 鍏ㄥ満鏁扮粍(鐪佷竴娆″垎閰嶅拰涓€閬嶈瀛?銆?pub fn face_fluxes(cfg: &Config, geom: &Geometry, cells: &Cells, faces: &mut Faces) {
    let gamma = cfg.physics.gamma;
    let u = &cells.u;
    let nj = u.nj() as isize;

    // tau 闈?(i, j) 鍒嗛殧鍗曞厓 (i鈭?, j) 涓?(i, j),i 鈭?[0, NI]
    let tau_geom = &geom.tau;
    faces.tau.flux.par_interior_rows_mut().for_each(|(i, mut row)| {
        for j in 0..nj {
            let fu = 0.5 * (u.get(i - 1, j) + u.get(i, j));
            row[j] = euler_flux(fu, tau_geom.at(i, j), gamma);
        }
    });

    // n 闈?(i, j) 鍒嗛殧鍗曞厓 (i, j鈭?) 涓?(i, j),j 鈭?[0, NJ)
    let n_geom = &geom.nrm;
    faces.nrm.flux.par_interior_rows_mut().for_each(|(i, mut row)| {
        for j in 0..nj {
            let fu = 0.5 * (u.get(i, j - 1) + u.get(i, j));
            row[j] = euler_flux(fu, n_geom.at(i, j), gamma);
        }
    });
}

/// 鍗曞厓涓婄殑瀵规祦鐜噺銆?pub fn assemble(geom: &Geometry, faces: &Faces, out: &mut Field<Vec5>) {
    let (tau, nrm) = (&faces.tau.flux, &faces.nrm.flux);
    let nj = geom.nj as isize;
    out.par_interior_rows_mut().for_each(|(i, mut row)| {
        for j in 0..nj {
            let jp1 = if j + 1 < nj { j + 1 } else { 0 };
            row[j] = tau.get(i + 1, j) - tau.get(i, j) + nrm.get(i, jp1) - nrm.get(i, j);
        }
    });
}

/// 涓€娆″畬鏁寸殑瀵规祦椤硅绠椼€?pub fn compute(cfg: &Config, geom: &Geometry, cells: &mut Cells, faces: &mut Faces) {
    face_fluxes(cfg, geom, cells, faces);
    assemble(geom, faces, &mut cells.fc);
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

    #[test]
    fn flux_matches_analytic_euler_flux() {
        let (rho, u, v, p, nut) = (1.15, 60.0, -20.0, 9.5e4, 3e-4);
        let gamma = 1.4;
        let e = p / (rho * (gamma - 1.0)) + 0.5 * (u * u + v * v);
        let fu = Vec5::new(rho, rho * u, rho * v, rho * e, rho * nut);
        let g = FaceGeom { nx: 0.3, ny: -0.7, mx: 0.0, my: 0.0 };
        let f = euler_flux(fu, &g, gamma);
        let vn = u * g.nx + v * g.ny;
        assert!((f[0] - rho * vn).abs() < 1e-12);
        assert!((f[1] - (rho * u * vn + p * g.nx)).abs() < 1e-8);
        assert!((f[2] - (rho * v * vn + p * g.ny)).abs() < 1e-8);
        assert!((f[3] - (rho * e + p) * vn).abs() < 1e-4);
        assert!((f[4] - rho * nut * vn).abs() < 1e-16);
    }

    #[test]
    fn flux_is_linear_in_the_normal() {
        let fu = Vec5::new(1.2, 72.0, -18.0, 2.6e5, 3.6e-4);
        let g1 = FaceGeom { nx: 0.3, ny: -0.7, mx: 0.0, my: 0.0 };
        let g2 = FaceGeom { nx: 0.6, ny: -1.4, mx: 0.0, my: 0.0 };
        let (a, b) = (euler_flux(fu, &g1, 1.4), euler_flux(fu, &g2, 1.4));
        for k in 0..5 {
            assert!((b[k] - 2.0 * a[k]).abs() <= 1e-9 * b[k].abs().max(1.0));
        }
    }

    /// 鑷敱鏉ユ祦淇濇寔鎬?鍧囧寑鍦轰笅骞冲潎娴佺殑瀵规祦娈嬪樊搴斾负鏈哄櫒绮惧害銆?    ///
    /// 蹇呴』**缁曡繃**杈圭晫鏉′欢閾轰竴涓惈铏氭嫙灞傜殑鍧囧寑鍦?鍥哄闀滃儚浼氳璐村澶勪笉鍐嶅潎鍖€
    /// (鐗╃悊涓婃纭?浣嗛偅妫€楠岀殑鏄埆鐨勪笢瑗?銆?    #[test]
    fn free_stream_is_preserved() {
        let (cfg, mut dom) = setup();
        dom.cells.set_uniform(&cfg, 1.176, 69.4, 17.3, 101325.0, 1.5e-4);
        compute(&cfg, &dom.geom, &mut dom.cells, &mut dom.faces);
        let scale = dom
            .faces
            .tau
            .flux
            .interior()
            .map(|(i, j)| dom.faces.tau.flux.get(i, j).amax())
            .fold(0.0f64, f64::max);
        for (i, j) in dom.cells.fc.interior() {
            let fc = dom.cells.fc.get(i, j);
            for k in 0..5 {
                assert!(
                    fc[k].abs() < 1e-12 * scale,
                    "Fc[{k}] = {:e} at ({i},{j}), scale {scale:e}",
                    fc[k]
                );
            }
        }
    }

    /// 闀滃儚澹侀潰 鈬?澹侀潰涓婄殑娉曞悜璐ㄩ噺閫氶噺涓?0銆?    #[test]
    fn no_mass_flux_through_the_wall() {
        let (cfg, mut dom) = setup();
        compute(&cfg, &dom.geom, &mut dom.cells, &mut dom.faces);
        for j in 0..dom.cells.nj as isize {
            let m = dom.faces.tau.flux.get(0, j)[comp::RHO];
            assert!(m.abs() < 1e-9, "mass leaks through wall at j={j}: {m:e}");
        }
    }

    /// 鍐呴儴闈㈤€氶噺鍦ㄥ崟鍏冪幆閲忎腑涓や袱鎶垫秷:鍏ㄥ満 危 Fc 鍙墿杈圭晫璐＄尞銆?    #[test]
    fn interior_fluxes_telescope() {
        let (cfg, mut dom) = setup();
        // 閫犱竴涓潪鍧囧寑鍦?纭繚鎶垫秷涓嶆槸鍥犱负閫氶噺鎭掔瓑
        for (i, j) in dom.cells.rho.interior().collect::<Vec<_>>() {
            let s = 1.0 + 0.03 * i as f64 + 0.01 * j as f64;
            dom.cells.rho.set(i, j, cfg.derived.rho_inf * s);
            dom.cells.pack(i, j);
        }
        crate::boundary::apply(&cfg, &dom.geom, &mut dom.cells);
        compute(&cfg, &dom.geom, &mut dom.cells, &mut dom.faces);

        let total: f64 = dom
            .cells
            .fc
            .interior()
            .map(|(i, j)| dom.cells.fc.get(i, j)[comp::RHO])
            .sum();
        // 鍛ㄥ悜闈㈠畬鍏ㄦ姷娑?鍛ㄦ湡),寰勫悜鍙墿澹侀潰涓庤繙鍦?        let ni = dom.cells.ni as isize;
        let boundary: f64 = (0..dom.cells.nj as isize)
            .map(|j| {
                dom.faces.tau.flux.get(ni, j)[comp::RHO] - dom.faces.tau.flux.get(0, j)[comp::RHO]
            })
            .sum();
        assert!(
            (total - boundary).abs() < 1e-9 * boundary.abs().max(1.0),
            "telescoping failed: {total:e} vs {boundary:e}"
        );
    }
}

