//! 姹傝В鍣ㄩ厤缃?鈥斺€?鐩存帴璇?Python 鍩虹嚎鐢ㄧ殑鍚屼竴浠?`config.json`銆?//!
//! 鎵€鏈夋淳鐢熼噺(cv銆乧p銆佹潵娴佺姸鎬併€丆w1銆佄教冣垶 鈥?鍦?[`Config::finish`] 閲屼竴娆＄畻濂?
//! 涔嬪悗灏辨槸鍙鐨勩€傝繖鏍蜂袱渚у疄鐜板叡浜悓涓€濂楄緭鍏?浜ゅ弶楠岃瘉鎵嶆湁鎰忎箟銆?
use std::path::Path;

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Physics {
    pub gamma: f64,
    #[serde(rename = "R")]
    pub r_gas: f64,
    /// Sutherland 鍙傝€冩俯搴?    #[serde(rename = "T0")]
    pub t_ref: f64,
    /// Sutherland 甯告暟娓╁害
    #[serde(rename = "Ts")]
    pub t_suth: f64,
    pub mu0: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Simulation {
    /// 鏉ユ祦鏀昏(搴?
    #[serde(rename = "AOA")]
    pub aoa: f64,
    #[serde(rename = "Ma")]
    pub mach: f64,
    #[serde(rename = "CFL")]
    pub cfl: f64,
    /// 铏氭嫙灞傛暟
    #[serde(rename = "IM")]
    pub halo: usize,
    /// 鏉ユ祦闈欐俯
    #[serde(rename = "T")]
    pub t_inf: f64,
    /// 鏉ユ祦闈欏帇
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

/// Spalart-Allmaras 涓€鏂圭▼婀嶆祦妯″瀷鐨勫父鏁般€?///
/// 娉ㄦ剰 `sigma` 娌跨敤 Python 鍩虹嚎鐨勭害瀹?瀛樼殑鏄?**1/蟽**(=1.5,鍗?蟽_SA = 2/3)銆?#[derive(Debug, Clone, Deserialize)]
#[allow(non_snake_case)]
pub struct SpalartAllmaras {
    pub Cv1: f64,
    /// 灞傛祦鏅湕鐗规暟(绌烘皵 鈮?0.71)
    pub Pr: f64,
    /// 婀嶆祦鏅湕鐗规暟(鈮?0.9)
    pub Prt: f64,
    /// 1/蟽
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
    /// 浜岄樁(婵€娉?鑰楁暎绯绘暟
    pub k2: f64,
    /// 鍥涢樁(鑳屾櫙)鑰楁暎绯绘暟
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

/// 鐢遍厤缃竴娆℃€ф帹鍑虹殑甯搁噺銆?#[derive(Debug, Clone, Default)]
pub struct Derived {
    pub cv: f64,
    pub cp: f64,
    /// Cw1 = Cb1/魏虏 + (1+Cb2)/蟽
    pub cw1: f64,
    pub cv1_cubed: f64,
    /// 鏉ユ祦澹伴€?    pub c_inf: f64,
    /// 鏉ユ祦 x/y 鏂瑰悜閫熷害
    pub u_inf: f64,
    pub v_inf: f64,
    pub rho_inf: f64,
    /// 鏉ユ祦鍒嗗瓙绮樺害(Sutherland)
    pub mu_inf: f64,
    /// 鏉ユ祦婀嶆祦宸ヤ綔鍙橀噺 谓虄鈭?= 0.1路谓鈭?    pub nut_inf: f64,
    /// 鏉ユ祦鏄惁瓒呭０閫?    pub supersonic: bool,
    /// `mu0路(T0+Ts)`,Sutherland 鍏紡閲岀殑甯告暟鍥犲瓙
    pub suth_num: f64,
    /// `1/T0`,鍏嶅幓閫愬崟鍏冪殑闄ゆ硶
    pub inv_t_ref: f64,
    /// `1/Pr`銆乣1/Prt`
    pub inv_pr: f64,
    pub inv_prt: f64,
    /// `1/魏虏`
    pub inv_kappa2: f64,
    /// `Cw3鈦禶 涓?`1 + Cw3鈦禶,fw 閲岀殑甯告暟
    pub cw3_6: f64,
    pub one_plus_cw3_6: f64,
}

/// Sutherland 鍒嗗瓙绮樺害銆?///
/// `x^1.5` 鍐欐垚 `x路鈭歺`:涓よ€呮暟瀛︿笂绛変环,浣?`sqrt` 鏄竴鏉＄‖浠舵寚浠?鑰?`powf`
/// 瑕佽蛋 libm 鐨勯€氱敤骞傚嚱鏁?绾︽參涓€涓暟閲忕骇)銆傝繖涓嚱鏁版瘡涓崟鍏冩瘡绾ч兘瑕佽皟涓€娆?
/// 鏄儹鐐逛箣涓€銆?#[inline(always)]
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

    /// 璁＄畻鍏ㄩ儴娲剧敓閲忋€備慨鏀逛簡浠讳綍杈撳叆涔嬪悗閮借閲嶆柊璋冪敤銆?    pub fn finish(&mut self) {
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

    /// Sutherland 绮樺害(鐢ㄦ湰閰嶇疆鐨勫弬鑰冨€?銆傜儹鐐瑰嚱鏁?瑙?[`sutherland`] 鐨勮鏄庛€?    #[inline(always)]
    pub fn mu(&self, t: f64) -> f64 {
        let r = t * self.derived.inv_t_ref;
        r * r.sqrt() * self.derived.suth_num / (t + self.physics.t_suth)
    }
}

/// 鏄惧紡 Runge-Kutta 鐨勭骇绯绘暟銆傛湯绾у繀椤讳负 1,鍚﹀垯鏍煎紡涓嶇浉瀹广€?pub const RK_ALPHA: [f64; 5] = [0.25, 1.0 / 6.0, 0.375, 0.5, 1.0];

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = include_str!("../config.json");

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

