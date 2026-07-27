//! 鏃堕棿鎺ㄨ繘椹卞姩:鏄惧紡浜旂骇 Runge-Kutta銆?//!
//! ```text
//! U鈦解伆鈦?= U鈦?//! U鈦结祻鈦?= U鈦?鈭?伪鈧柭肺攖路R(U鈦结祻鈦宦光伨)/V ,  k = 1鈥?
//! U鈦库伜鹿 = U鈦解伒鈦?//! ```
//!
//! `伪 = (1/4, 1/6, 3/8, 1/2, 1)` 鈥斺€?**鏈骇蹇呴』涓?1**,鍚﹀垯鏍煎紡涓庢椂闂村鏁颁笉鐩稿
//! (Python 鍩虹嚎鍙窇浜嗗墠 4 绾?瑙?`BUGS.md` B1)銆傛畫宸?//! `R = Fc 鈭?Fv 鈭?Fd 鈭?S` 鍦ㄦ瘡涓€绾ч噸鏂拌绠?螖t 鍒欏湪鏁翠釜鏃堕棿姝ュ唴鍐荤粨銆?
use crate::config::{Config, RK_ALPHA};
use crate::field::comp;
use crate::geometry::Geometry;
use crate::mesh::Mesh;
use crate::state::{Cells, Domain, F64Field, NonPhysical};
use crate::{boundary, convection, dissipation, gradient, source, timestep, viscous};

/// 鍒ゅ畾"瑙ｅ凡鍙戞暎"鐨勪笅闄愩€備綆浜庢鍊艰鏄庢牸寮忓凡缁忓け绋?缁х画鎺ㄨ繘鍙細浜у嚭 NaN銆?const MIN_DENSITY: f64 = 1e-15;
const MIN_PRESSURE: f64 = 1e-15;

/// 姹傝В鍣?鎷ユ湁缃戞牸銆佸嚑浣曚笌鍏ㄩ儴鍦洪噺銆?pub struct Solver {
    pub cfg: Config,
    pub dom: Domain,
    /// 绱鐗╃悊鏃堕棿
    pub totaltime: f64,
    /// 宸插畬鎴愮殑鏃堕棿姝ユ暟
    pub step: usize,
    /// 涓婁竴鏃堕棿姝ョ殑瀵嗗害,鐢ㄤ簬娈嬪樊
    rho_prev: F64Field,
}

/// 涓€娆?`run` 鐨勭粨鏋溿€?#[derive(Debug, Clone, Copy)]
pub struct RunReport {
    pub steps: usize,
    pub residual: f64,
    pub totaltime: f64,
    pub converged: bool,
}

impl Solver {
    /// 鐢辩綉鏍间笌閰嶇疆瑁呴厤姹傝В鍣?骞跺畬鎴愬垵濮嬪寲(鍑犱綍 鈫?鍒濆満 鈫?杈圭晫)銆?    pub fn new(cfg: Config, mesh: &Mesh) -> Self {
        let halo = cfg.simulation.halo;
        let geom = Geometry::build(mesh, halo);
        let (ni, nj) = (geom.ni, geom.nj);
        let mut dom = Domain::new(geom, halo);
        dom.cells.initialize(&cfg);
        let mut s = Self {
            cfg,
            dom,
            totaltime: 0.0,
            step: 0,
            rho_prev: F64Field::new(ni, nj, halo),
        };
        boundary::apply(&s.cfg, &s.dom.geom, &mut s.dom.cells);
        s
    }

    pub fn from_paths(config: &str, mesh: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let cfg = Config::from_path(config)?;
        let mesh = Mesh::from_path(mesh)?;
        Ok(Self::new(cfg, &mesh))
    }

    #[inline]
    pub fn ni(&self) -> usize {
        self.dom.cells.ni
    }
    #[inline]
    pub fn nj(&self) -> usize {
        self.dom.cells.nj
    }
    #[inline]
    pub fn n_cells(&self) -> usize {
        self.ni() * self.nj()
    }

    /// 璁＄畻涓€绾х殑娈嬪樊鍒嗛」 `Fc`銆乣Fv`銆乣Fd`銆乣S`(涓嶆帹杩?`U`)銆?    ///
    /// 鍗曠嫭鏆撮湶鍑烘潵鏄负浜?golden 姣斿涓庡熀鍑嗘祴璇曞彲浠ュ彧娴嬫煇涓€娈点€?    pub fn residual_terms(&mut self) {
        let Self { cfg, dom, .. } = self;
        boundary::apply(cfg, &dom.geom, &mut dom.cells);
        convection::compute(cfg, &dom.geom, &mut dom.cells, &mut dom.faces);
        gradient::compute(&dom.geom, &mut dom.cells);
        viscous::compute(cfg, &dom.geom, &mut dom.cells, &mut dom.faces);
        dissipation::compute(cfg, &dom.geom, &mut dom.cells, &mut dom.faces);
        source::compute(cfg, &dom.geom, &mut dom.cells);
    }

    /// 鎺ㄨ繘涓€涓椂闂存銆?    pub fn advance(&mut self) -> Result<(), NonPhysical> {
        // 姝ュ垵蹇収:U_former 渚涘悇绾т娇鐢?rho_prev 渚涙畫宸娇鐢ㄣ€?        // 涓や釜鏁扮粍鐨勭墿鐞嗗尯甯冨眬瀹屽叏涓€鑷?鏁村潡鎷疯礉鍗冲彲(鍚?halo,鏃犳墍璋?銆?        self.dom
            .cells
            .u_former
            .raw_mut()
            .copy_from_slice(self.dom.cells.u.raw());
        self.rho_prev
            .raw_mut()
            .copy_from_slice(self.dom.cells.rho.raw());

        // 螖t 鍦ㄦ暣涓?RK 姝ュ唴鍐荤粨,骞朵笖姣忔鍙疮鍔犱竴娆＄墿鐞嗘椂闂?        let dt = timestep::compute(&self.cfg, &self.dom.geom, &mut self.dom.cells);
        self.totaltime += dt;

        for &alpha in RK_ALPHA.iter() {
            self.residual_terms();
            self.update(alpha * dt);
            self.unpack()?;
        }
        self.step += 1;
        Ok(())
    }

    /// `U = U_former 鈭?a路(Fc 鈭?Fv 鈭?Fd 鈭?S)/V`銆?    fn update(&mut self, a: f64) {
        use rayon::iter::ParallelIterator;
        let nj = self.dom.cells.nj as isize;
        let Cells {
            u,
            u_former,
            fc,
            fv,
            fd,
            src,
            ..
        } = &mut self.dom.cells;
        let vol = &self.dom.geom.inv_vol;
        let (u_former, fc, fv, fd, src) = (&*u_former, &*fc, &*fv, &*fd, &*src);

        u.par_interior_rows_mut().for_each(|(i, mut row)| {
            for j in 0..nj {
                let mut r = fc.get(i, j) - fv.get(i, j) - fd.get(i, j);
                r[comp::RHO_NU] -= src.get(i, j);
                row[j] = u_former.get(i, j) - r * (a * vol.get(i, j));
            }
        });
    }

    /// 鐢卞畧鎭掗噺杩樺師鍘熷鍙橀噺銆傞潪鐗╃悊鐘舵€?蟻鈮? 鎴?p鈮?)杩斿洖閿欒鑰屼笉鏄粓姝㈣繘绋嬨€?    ///
    /// 涔濅釜杈撳嚭鏁扮粍瑕佸悓鏃跺啓,涓庡叾鎶?rayon 鐨勪節璺?`zip` 寮鸿鎷煎嚭鏉?涓嶅涓茶 鈥斺€?    /// 杩欎竴姝ュ彧鏄€愮偣浠ｆ暟,璁垮瓨鍙楅檺,骞惰鏀剁泭鎶典笉杩囧悓姝ュ紑閿€銆?    fn unpack(&mut self) -> Result<(), NonPhysical> {
        let (gamma, r_gas) = (self.cfg.physics.gamma, self.cfg.physics.r_gas);
        let c = &mut self.dom.cells;
        let nj = c.nj as isize;

        for i in 0..c.ni as isize {
            for j in 0..nj {
                let u = c.u.get(i, j);
                let rho = u[comp::RHO];
                // 鐢?`< =` 鍔犳樉寮?NaN 妫€鏌?鍙戞暎鏃跺畧鎭掗噺鍙兘宸茬粡鏄?NaN,
                // 鑰?NaN 涓庝换浣曢槇鍊兼瘮杈冮兘涓哄亣,鍗曢潬 `rho <= eps` 浼氭紡鎺夈€?                if rho.is_nan() || rho <= MIN_DENSITY {
                    return Err(NonPhysical { i, j, rho, p: f64::NAN });
                }
                let inv = 1.0 / rho;
                let (vx, vy) = (u[comp::MX] * inv, u[comp::MY] * inv);
                let e = u[comp::RHO_E] * inv;
                let p = (gamma - 1.0) * (u[comp::RHO_E] - rho * (vx * vx + vy * vy) * 0.5);
                if p.is_nan() || p <= MIN_PRESSURE {
                    return Err(NonPhysical { i, j, rho, p });
                }
                let t = p / (r_gas * rho);
                c.rho.set(i, j, rho);
                c.vx.set(i, j, vx);
                c.vy.set(i, j, vy);
                c.nut.set(i, j, u[comp::RHO_NU] * inv);
                c.e.set(i, j, e);
                c.p.set(i, j, p);
                c.h.set(i, j, e + p * inv);
                c.t.set(i, j, t);
                c.c.set(i, j, (r_gas * gamma * t).sqrt());
            }
        }
        Ok(())
    }

    /// 鍩轰簬瀵嗗害鍙樺寲鐨?L2 娈嬪樊 `鈭?危(蟻 鈭?蟻_prev)虏 / N)`銆?    ///
    /// 涓茶姹傚拰浠ヤ繚璇侀€愪綅鍙鐜?浠ｄ环 <1% 鐨勬鑰楁椂)銆?    pub fn residual(&self) -> f64 {
        let (ni, nj) = (self.ni() as isize, self.nj() as isize);
        let mut acc = 0.0;
        for i in 0..ni {
            for j in 0..nj {
                let d = self.dom.cells.rho.get(i, j) - self.rho_prev.get(i, j);
                acc += d * d;
            }
        }
        (acc / (ni as f64) / (nj as f64)).sqrt()
    }

    /// 杩唬鍒版敹鏁涙垨杈惧埌姝ユ暟涓婇檺銆俙on_step` 姣忔鍥炶皟涓€娆?`(step, residual)`銆?    pub fn run<F>(&mut self, max_steps: Option<usize>, mut on_step: F) -> Result<RunReport, NonPhysical>
    where
        F: FnMut(usize, f64),
    {
        let limit = max_steps.unwrap_or(self.cfg.solver.iteration);
        let target = self.cfg.solver.targetres;
        let mut residual = f64::NAN;
        let mut done = 0;
        for k in 1..=limit {
            self.advance()?;
            residual = self.residual();
            done = k;
            on_step(k, residual);
            if residual < target {
                break;
            }
        }
        Ok(RunReport {
            steps: done,
            residual,
            totaltime: self.totaltime,
            converged: residual < target,
        })
    }

    /// 瀵煎嚭鐗╃悊鍗曞厓鐨勬祦鍦轰负 CSV(鍒椾笌 Python 鍩虹嚎鐨?`result.csv` 涓€鑷?銆?    pub fn write_csv(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        let f = std::fs::File::create(path)?;
        let mut w = std::io::BufWriter::new(f);
        writeln!(w, "i,j,x,y,rho,p,T,u,v,miubl")?;
        let c = &self.dom.cells;
        let g = &self.dom.geom;
        for i in 0..self.ni() as isize {
            for j in 0..self.nj() as isize {
                writeln!(
                    w,
                    "{},{},{:.8e},{:.8e},{:.8e},{:.8e},{:.8e},{:.8e},{:.8e},{:.8e}",
                    i + 1,
                    j + 1,
                    g.cx.get(i, j),
                    g.cy.get(i, j),
                    c.rho.get(i, j),
                    c.p.get(i, j),
                    c.t.get(i, j),
                    c.vx.get(i, j),
                    c.vy.get(i, j),
                    c.nut.get(i, j)
                )?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solver() -> Solver {
        let cfg = Config::from_str(include_str!("../config.json")).unwrap();
        let mesh = Mesh::parse(include_str!("../fangdata.txt")).unwrap();
        Solver::new(cfg, &mesh)
    }

    #[test]
    fn advances_without_blowing_up() {
        let mut s = solver();
        for _ in 0..20 {
            s.advance().unwrap();
        }
        let c = &s.dom.cells;
        for (i, j) in c.rho.interior() {
            assert!(c.rho.get(i, j) > 0.0, "rho <= 0 at ({i},{j})");
            assert!(c.p.get(i, j) > 0.0);
            assert!(c.t.get(i, j).is_finite() && c.nut.get(i, j).is_finite());
        }
    }

    #[test]
    fn physical_time_accumulates_once_per_step() {
        let mut s = solver();
        s.advance().unwrap();
        let t1 = s.totaltime;
        s.advance().unwrap();
        let dt2 = s.totaltime - t1;
        // 鑻ユ寜绾х疮鍔?鍗曟浼氭槸 5 鍊?        assert!((0.2..5.0).contains(&(t1 / dt2)), "t1={t1:e} dt2={dt2:e}");
    }

    #[test]
    fn residual_decreases() {
        let mut s = solver();
        let mut first = f64::NAN;
        let mut last = f64::NAN;
        for k in 0..60 {
            s.advance().unwrap();
            let r = s.residual();
            if k == 0 {
                first = r;
            }
            last = r;
        }
        assert!(last < first, "residual grew: {first:e} 鈫?{last:e}");
    }

    #[test]
    fn converges_to_target() {
        let mut s = solver();
        s.cfg.solver.targetres = 1e-8;
        let rep = s.run(Some(3000), |_, _| {}).unwrap();
        assert!(rep.converged, "did not converge: {:e}", rep.residual);
    }

    #[test]
    fn run_is_deterministic_across_repeated_invocations() {
        let snapshot = |steps| {
            let mut s = solver();
            s.run(Some(steps), |_, _| {}).unwrap();
            s.dom.cells.rho.to_interior_vec()
        };
        assert_eq!(snapshot(15), snapshot(15));
    }

    /// 鍧囧寑鏉ユ祦涓嬪钩鍧囨祦娈嬪樊搴斾负鏈哄櫒绮惧害(鑷敱鏉ユ祦淇濇寔鎬?鍚叏閮ㄥ洓涓垎椤?銆?    ///
    /// 杩欓噷缁曞紑 `boundary::apply` 鐩存帴閾烘弧鍚櫄鎷熷眰鐨勫潎鍖€鍦?鍥哄闀滃儚浼氱牬鍧?    /// 璐村澶勭殑鍧囧寑鎬?閭ｆ槸鐗╃悊涓婃纭殑,浣嗕笉鏄湰鐢ㄤ緥妫€楠岀殑鍐呭)銆?    #[test]
    fn free_stream_mean_flow_residual_vanishes() {
        let mut s = solver();
        s.dom.cells.set_uniform(&s.cfg, 1.176, 69.4, 17.3, 101325.0, 1.5e-4);
        timestep::compute(&s.cfg, &s.dom.geom, &mut s.dom.cells);
        let Solver { cfg, dom, .. } = &mut s;
        convection::compute(cfg, &dom.geom, &mut dom.cells, &mut dom.faces);
        gradient::compute(&dom.geom, &mut dom.cells);
        viscous::compute(cfg, &dom.geom, &mut dom.cells, &mut dom.faces);
        dissipation::compute(cfg, &dom.geom, &mut dom.cells, &mut dom.faces);
        let scale = s
            .dom
            .faces
            .tau
            .flux
            .interior()
            .map(|(i, j)| s.dom.faces.tau.flux.get(i, j).amax())
            .fold(0.0f64, f64::max);
        let c = &s.dom.cells;
        for (i, j) in c.fc.interior() {
            let r = c.fc.get(i, j) - c.fv.get(i, j) - c.fd.get(i, j);
            for k in 0..4 {
                assert!(r[k].abs() < 1e-13 * scale, "residual[{k}]={:e} at ({i},{j})", r[k]);
            }
        }
    }

    #[test]
    fn non_physical_state_is_reported_not_fatal() {
        let mut s = solver();
        let mut u = s.dom.cells.u.get(0, 0);
        u[comp::RHO] = -1.0;
        s.dom.cells.u.set(0, 0, u);
        assert!(s.unpack().is_err());
    }
}

