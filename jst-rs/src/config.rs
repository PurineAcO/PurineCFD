//! 求解器配置 —— 直接读 Python 基线用的同一份 `config.json`。
//!
//! 所有派生量(cv、cp、来流状态、Cw1、ν̃∞ …)在 [`Config::finish`] 里一次算好,
//! 之后就是只读的。这样两侧实现共享同一套输入,交叉验证才有意义。

use std::path::Path;

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Physics {
    pub gamma: f64,
    #[serde(rename = "R")]
    pub r_gas: f64,
    /// Sutherland 参考温度
    #[serde(rename = "T0")]
    pub t_ref: f64,
    /// Sutherland 常数温度
    #[serde(rename = "Ts")]
    pub t_suth: f64,
    pub mu0: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Simulation {
    /// 来流攻角(度)
    #[serde(rename = "AOA")]
    pub aoa: f64,
    #[serde(rename = "Ma")]
    pub mach: f64,
    #[serde(rename = "CFL")]
    pub cfl: f64,
    /// 虚拟层数
    #[serde(rename = "IM")]
    pub halo: usize,
    /// 来流静温
    #[serde(rename = "T")]
    pub t_inf: f64,
    /// 来流静压
    #[serde(rename = "P")]
    pub p_inf: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SolverCfg {
    #[serde(default = "default_iteration")]
    pub iteration: usize,
    #[serde(default = "default_targetres")]
    pub targetres: f64,
}

fn default_iteration() -> usize {
    10_000
}
fn default_targetres() -> f64 {
    1e-10
}

impl Default for SolverCfg {
    fn default() -> Self {
        Self {
            iteration: default_iteration(),
            targetres: default_targetres(),
        }
    }
}

/// Spalart-Allmaras 一方程湍流模型的常数。
///
/// 注意 `sigma` 沿用 Python 基线的约定,存的是 **1/σ**(=1.5,即 σ_SA = 2/3)。
#[derive(Debug, Clone, Deserialize)]
#[allow(non_snake_case)]
pub struct SpalartAllmaras {
    pub Cv1: f64,
    /// 层流普朗特数(空气 ≈ 0.71)
    pub Pr: f64,
    /// 湍流普朗特数(≈ 0.9)
    pub Prt: f64,
    /// 1/σ
    pub sigma: f64,
    pub Cb1: f64,
    pub Cb2: f64,
    pub Ct3: f64,
    pub Ct4: f64,
    pub Cw2: f64,
    pub Cw3: f64,
    pub fv3: f64,
    pub kappa: f64,
    pub rmax: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Dissipation {
    /// 二阶(激波)耗散系数
    pub k2: f64,
    /// 四阶(背景)耗散系数
    pub k4: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub physics: Physics,
    pub simulation: Simulation,
    #[serde(default)]
    pub solver: SolverCfg,
    pub spalart_allmaras: SpalartAllmaras,
    pub dissipation: Dissipation,

    #[serde(skip)]
    pub derived: Derived,
}

/// 由配置一次性推出的常量。
#[derive(Debug, Clone, Default)]
pub struct Derived {
    pub cv: f64,
    pub cp: f64,
    /// Cw1 = Cb1/κ² + (1+Cb2)/σ
    pub cw1: f64,
    pub cv1_cubed: f64,
    /// 来流声速
    pub c_inf: f64,
    /// 来流 x/y 方向速度
    pub u_inf: f64,
    pub v_inf: f64,
    pub rho_inf: f64,
    /// 来流分子粘度(Sutherland)
    pub mu_inf: f64,
    /// 来流湍流工作变量 ν̃∞ = 0.1·ν∞
    pub nut_inf: f64,
    /// 来流是否超声速
    pub supersonic: bool,
    /// `mu0·(T0+Ts)`,Sutherland 公式里的常数因子
    pub suth_num: f64,
    /// `1/T0`,免去逐单元的除法
    pub inv_t_ref: f64,
    /// `1/Pr`、`1/Prt`
    pub inv_pr: f64,
    pub inv_prt: f64,
    /// `1/κ²`
    pub inv_kappa2: f64,
    /// `Cw3⁶` 与 `1 + Cw3⁶`,fw 里的常数
    pub cw3_6: f64,
    pub one_plus_cw3_6: f64,
}

/// Sutherland 分子粘度。
///
/// `x^1.5` 写成 `x·√x`:两者数学上等价,但 `sqrt` 是一条硬件指令,而 `powf`
/// 要走 libm 的通用幂函数(约慢一个数量级)。这个函数每个单元每级都要调一次,
/// 是热点之一。
#[inline(always)]
pub fn sutherland(mu0: f64, t: f64, t_ref: f64, t_suth: f64) -> f64 {
    let r = t / t_ref;
    mu0 * (r * r.sqrt()) * (t_ref + t_suth) / (t + t_suth)
}

impl Config {
    pub fn from_path(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let text = std::fs::read_to_string(path.as_ref())?;
        Self::from_str(&text)
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_str(text: &str) -> std::io::Result<Self> {
        let mut cfg: Config = serde_json::from_str(text)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        cfg.finish();
        Ok(cfg)
    }

    /// 计算全部派生量。修改了任何输入之后都要重新调用。
    pub fn finish(&mut self) {
        let p = &self.physics;
        let s = &self.simulation;
        let sa = &self.spalart_allmaras;

        let cv = p.r_gas / (p.gamma - 1.0);
        let c_inf = (p.gamma * p.r_gas * s.t_inf).sqrt();
        let aoa = s.aoa.to_radians();
        let rho_inf = s.p_inf / (p.r_gas * s.t_inf);
        let mu_inf = sutherland(p.mu0, s.t_inf, p.t_ref, p.t_suth);
        let u_inf = c_inf * s.mach * aoa.cos();
        let v_inf = c_inf * s.mach * aoa.sin();

        self.derived = Derived {
            cv,
            cp: p.gamma * cv,
            cw1: sa.Cb1 / (sa.kappa * sa.kappa) + (1.0 + sa.Cb2) * sa.sigma,
            cv1_cubed: sa.Cv1.powi(3),
            c_inf,
            u_inf,
            v_inf,
            rho_inf,
            mu_inf,
            nut_inf: 0.1 * mu_inf / rho_inf,
            supersonic: u_inf.hypot(v_inf) >= c_inf,
            suth_num: p.mu0 * (p.t_ref + p.t_suth),
            inv_t_ref: 1.0 / p.t_ref,
            inv_pr: 1.0 / sa.Pr,
            inv_prt: 1.0 / sa.Prt,
            inv_kappa2: 1.0 / (sa.kappa * sa.kappa),
            cw3_6: sa.Cw3.powi(6),
            one_plus_cw3_6: 1.0 + sa.Cw3.powi(6),
        };
    }

    /// Sutherland 粘度(用本配置的参考值)。热点函数,见 [`sutherland`] 的说明。
    #[inline(always)]
    pub fn mu(&self, t: f64) -> f64 {
        let r = t * self.derived.inv_t_ref;
        r * r.sqrt() * self.derived.suth_num / (t + self.physics.t_suth)
    }
}

/// 显式 Runge-Kutta 的级系数。末级必须为 1,否则格式不相容。
pub const RK_ALPHA: [f64; 5] = [0.25, 1.0 / 6.0, 0.375, 0.5, 1.0];

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = include_str!("../../config.json");

    #[test]
    fn parses_repository_config() {
        let cfg = Config::from_str(SAMPLE).unwrap();
        assert_eq!(cfg.physics.gamma, 1.4);
        assert_eq!(cfg.simulation.halo, 3);
        assert_eq!(cfg.solver.iteration, 10_000);
    }

    #[test]
    fn solver_section_is_optional() {
        let text = SAMPLE.replace(
            "\"solver\": {\n        \"iteration\": 10000,\n        \"targetres\": 1e-10\n    },",
            "",
        );
        let cfg = Config::from_str(&text).unwrap();
        assert_eq!(cfg.solver.iteration, default_iteration());
    }

    #[test]
    fn derived_freestream_matches_isentropic_relations() {
        let cfg = Config::from_str(SAMPLE).unwrap();
        let d = &cfg.derived;
        assert!((d.c_inf - (1.4f64 * 287.06 * 300.0).sqrt()).abs() < 1e-12);
        assert!((d.rho_inf - 101325.0 / (287.06 * 300.0)).abs() < 1e-12);
        assert!((d.u_inf - d.c_inf * 0.2).abs() < 1e-12);
        assert!(d.v_inf.abs() < 1e-12);
        assert!(!d.supersonic);
    }

    #[test]
    fn cw1_closure() {
        let cfg = Config::from_str(SAMPLE).unwrap();
        let sa = &cfg.spalart_allmaras;
        let want = sa.Cb1 / sa.kappa.powi(2) + (1.0 + sa.Cb2) * sa.sigma;
        assert!((cfg.derived.cw1 - want).abs() < 1e-15);
    }

    #[test]
    fn prandtl_numbers_are_physical() {
        let cfg = Config::from_str(SAMPLE).unwrap();
        assert!(cfg.spalart_allmaras.Pr < cfg.spalart_allmaras.Prt);
    }

    #[test]
    fn rk_last_stage_is_unity() {
        assert_eq!(RK_ALPHA[RK_ALPHA.len() - 1], 1.0);
    }

    #[test]
    fn sutherland_recovers_mu0_at_reference() {
        let cfg = Config::from_str(SAMPLE).unwrap();
        assert!((cfg.mu(cfg.physics.t_ref) - cfg.physics.mu0).abs() < 1e-20);
    }
}
