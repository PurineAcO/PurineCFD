//! 时间推进驱动:显式五级 Runge-Kutta。
//!
//! ```text
//! U⁽⁰⁾ = Uⁿ
//! U⁽ᵏ⁾ = Uⁿ − αₖ·Δt·R(U⁽ᵏ⁻¹⁾)/V ,  k = 1…5
//! Uⁿ⁺¹ = U⁽⁵⁾
//! ```
//!
//! `α = (1/4, 1/6, 3/8, 1/2, 1)` —— **末级必须为 1**,否则格式与时间导数不相容
//! (Python 基线只跑了前 4 级,见 `BUGS.md` B1)。残差
//! `R = Fc − Fv − Fd − S` 在每一级重新计算,Δt 则在整个时间步内冻结。

use crate::config::{Config, RK_ALPHA};
use crate::field::comp;
use crate::geometry::Geometry;
use crate::mesh::Mesh;
use crate::state::{Cells, Domain, F64Field, NonPhysical};
use crate::{boundary, convection, dissipation, gradient, source, timestep, viscous};

/// 判定"解已发散"的下限。低于此值说明格式已经失稳,继续推进只会产出 NaN。
const MIN_DENSITY: f64 = 1e-15;
const MIN_PRESSURE: f64 = 1e-15;

/// 求解器:拥有网格、几何与全部场量。
pub struct Solver {
    pub cfg: Config,
    pub dom: Domain,
    /// 累计物理时间
    pub totaltime: f64,
    /// 已完成的时间步数
    pub step: usize,
    /// 上一时间步的密度,用于残差
    rho_prev: F64Field,
}

/// 一次 `run` 的结果。
#[derive(Debug, Clone, Copy)]
pub struct RunReport {
    pub steps: usize,
    pub residual: f64,
    pub totaltime: f64,
    pub converged: bool,
}

impl Solver {
    /// 由网格与配置装配求解器,并完成初始化(几何 → 初场 → 边界)。
    pub fn new(cfg: Config, mesh: &Mesh) -> Self {
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

    /// 计算一级的残差分项 `Fc`、`Fv`、`Fd`、`S`(不推进 `U`)。
    ///
    /// 单独暴露出来是为了 golden 比对与基准测试可以只测某一段。
    pub fn residual_terms(&mut self) {
        let Self { cfg, dom, .. } = self;
        boundary::apply(cfg, &dom.geom, &mut dom.cells);
        convection::compute(cfg, &dom.geom, &mut dom.cells, &mut dom.faces);
        gradient::compute(&dom.geom, &mut dom.cells);
        viscous::compute(cfg, &dom.geom, &mut dom.cells, &mut dom.faces);
        dissipation::compute(cfg, &dom.geom, &mut dom.cells, &mut dom.faces);
        source::compute(cfg, &dom.geom, &mut dom.cells);
    }

    /// 推进一个时间步。
    pub fn advance(&mut self) -> Result<(), NonPhysical> {
        // 步初快照:U_former 供各级使用,rho_prev 供残差使用。
        // 两个数组的物理区布局完全一致,整块拷贝即可(含 halo,无所谓)。
        self.dom
            .cells
            .u_former
            .raw_mut()
            .copy_from_slice(self.dom.cells.u.raw());
        self.rho_prev
            .raw_mut()
            .copy_from_slice(self.dom.cells.rho.raw());

        // Δt 在整个 RK 步内冻结,并且每步只累加一次物理时间
        let dt = timestep::compute(&self.cfg, &self.dom.geom, &mut self.dom.cells);
        self.totaltime += dt;

        for &alpha in RK_ALPHA.iter() {
            self.residual_terms();
            self.update(alpha * dt);
            self.unpack()?;
        }
        self.step += 1;
        Ok(())
    }

    /// `U = U_former − a·(Fc − Fv − Fd − S)/V`。
    fn update(&mut self, a: f64) {
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

    /// 由守恒量还原原始变量。非物理状态(ρ≤0 或 p≤0)返回错误而不是终止进程。
    ///
    /// 九个输出数组要同时写,与其把 rayon 的九路 `zip` 强行拼出来,不如串行 ——
    /// 这一步只是逐点代数,访存受限,并行收益抵不过同步开销。
    fn unpack(&mut self) -> Result<(), NonPhysical> {
        let (gamma, r_gas) = (self.cfg.physics.gamma, self.cfg.physics.r_gas);
        let c = &mut self.dom.cells;
        let nj = c.nj as isize;

        for i in 0..c.ni as isize {
            for j in 0..nj {
                let u = c.u.get(i, j);
                let rho = u[comp::RHO];
                // 用 `< =` 加显式 NaN 检查:发散时守恒量可能已经是 NaN,
                // 而 NaN 与任何阈值比较都为假,单靠 `rho <= eps` 会漏掉。
                if rho.is_nan() || rho <= MIN_DENSITY {
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

    /// 基于密度变化的 L2 残差 `√(Σ(ρ − ρ_prev)² / N)`。
    ///
    /// 串行求和以保证逐位可复现(代价 <1% 的步耗时)。
    pub fn residual(&self) -> f64 {
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

    /// 迭代到收敛或达到步数上限。`on_step` 每步回调一次 `(step, residual)`。
    pub fn run<F>(&mut self, max_steps: Option<usize>, mut on_step: F) -> Result<RunReport, NonPhysical>
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

    /// 导出物理单元的流场为 CSV(列与 Python 基线的 `result.csv` 一致)。
    pub fn write_csv(&self, path: &std::path::Path) -> std::io::Result<()> {
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
        let cfg = Config::from_str(include_str!("../../config.json")).unwrap();
        let mesh = Mesh::parse(include_str!("../../fangdata.txt")).unwrap();
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
        // 若按级累加,单步会是 5 倍
        assert!((0.2..5.0).contains(&(t1 / dt2)), "t1={t1:e} dt2={dt2:e}");
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
        assert!(last < first, "residual grew: {first:e} → {last:e}");
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

    /// 均匀来流下平均流残差应为机器精度(自由来流保持性,含全部四个分项)。
    ///
    /// 这里绕开 `boundary::apply` 直接铺满含虚拟层的均匀场:固壁镜像会破坏
    /// 贴壁处的均匀性(那是物理上正确的,但不是本用例检验的内容)。
    #[test]
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
