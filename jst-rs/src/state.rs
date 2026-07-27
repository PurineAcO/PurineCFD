//! 流场状态的存储布局。
//!
//! # 分组 SoA
//!
//! 不是"每个标量一个数组"(纯 SoA),也不是 Python 那样"每个单元一个对象"(AoS),
//! 而是**按使用方式分组**:总是被一起读写的量打包成一个小结构体,各组之间仍然
//! 是独立的连续数组。
//!
//! * [`Grad`] —— 八个梯度分量。粘性项与源项一次要用全部八个;打包后
//!   `gradient::compute` 只写一个数组,不必 `zip` 八个并行迭代器
//!   (rayon 的多路 `zip` 要求各路同步切分,开销远高于收益)。
//! * [`TurbAux`] —— `μ, χ, fv1`,S-A 的三个中间量,总是同时产生、同时消费。
//! * [`DiffTensor`] —— 扩散张量的两列,面上做平均时成对使用。
//! * [`Vec5`] —— 五个守恒分量,所有通量运算都作用在整组上。
//!
//! 这样每个 kernel 都是"读若干数组、写**一个**数组",借用检查直接证明无别名,
//! rayon 并行既不需要 `unsafe` 也不需要克隆中间结果。

use crate::config::Config;
use crate::field::{Field, Vec5};
use crate::geometry::Geometry;

pub type F64Field = Field<f64>;
pub type Vec5Field = Field<Vec5>;

/// 单元上的 Green-Gauss 梯度。
#[derive(Clone, Copy, Debug, Default)]
pub struct Grad {
    pub dudx: f64,
    pub dudy: f64,
    pub dvdx: f64,
    pub dvdy: f64,
    pub dtdx: f64,
    pub dtdy: f64,
    pub dnutdx: f64,
    pub dnutdy: f64,
}

/// S-A 模型的逐单元中间量。
#[derive(Clone, Copy, Debug, Default)]
pub struct TurbAux {
    /// 分子粘度 μ(Sutherland)
    pub mu: f64,
    /// χ = ρν̃/μ
    pub chi: f64,
    /// 阻尼函数 fv1
    pub fv1: f64,
}

/// 粘性/湍流扩散张量的两列。
#[derive(Clone, Copy, Debug, Default)]
pub struct DiffTensor {
    pub x: Vec5,
    pub y: Vec5,
}

/// 面上的自适应耗散系数。
#[derive(Clone, Copy, Debug, Default)]
pub struct Eps {
    /// 二阶(激波)系数 ε²
    pub e2: f64,
    /// 四阶(背景)系数 ε⁴
    pub e4: f64,
}

/// 单元中心量。
#[derive(Clone, Debug)]
pub struct Cells {
    pub ni: usize,
    pub nj: usize,

    // ── 守恒量 ──────────────────────────────────────────────
    /// `[ρ, ρu, ρv, ρE, ρν̃]`
    pub u: Vec5Field,
    /// 时间步开始时的守恒量(RK 各级都基于它更新)
    pub u_former: Vec5Field,

    // ── 原始变量 ────────────────────────────────────────────
    pub rho: F64Field,
    pub p: F64Field,
    pub t: F64Field,
    pub vx: F64Field,
    pub vy: F64Field,
    /// 单位质量总能
    pub e: F64Field,
    /// 单位质量总焓
    pub h: F64Field,
    /// 声速
    pub c: F64Field,
    /// 湍流工作变量 ν̃
    pub nut: F64Field,

    // ── 导出量 ──────────────────────────────────────────────
    pub grad: Field<Grad>,
    pub aux: Field<TurbAux>,
    pub diff: Field<DiffTensor>,

    // ── 残差分项 ────────────────────────────────────────────
    /// 对流通量
    pub fc: Vec5Field,
    /// 粘性/湍流扩散通量
    pub fv: Vec5Field,
    /// JST 人工粘性
    pub fd: Vec5Field,
    /// S-A 源项(只有第 5 分量非零,故存标量)
    pub src: F64Field,

    /// 当地时间步(逐单元,JST 谱半径需要)
    pub localdt: F64Field,
}

impl Cells {
    pub fn new(ni: usize, nj: usize, halo: usize) -> Self {
        let f = || Field::<f64>::new(ni, nj, halo);
        let v = || Field::<Vec5>::new(ni, nj, halo);
        Self {
            ni,
            nj,
            u: v(),
            u_former: v(),
            rho: f(),
            p: f(),
            t: f(),
            vx: f(),
            vy: f(),
            e: f(),
            h: f(),
            c: f(),
            nut: f(),
            grad: Field::new(ni, nj, halo),
            aux: Field::new(ni, nj, halo),
            diff: Field::new(ni, nj, halo),
            fc: v(),
            fv: v(),
            fd: v(),
            src: f(),
            localdt: f(),
        }
    }

    /// 由原始变量装配守恒量 `U`。
    #[inline]
    pub fn pack(&mut self, i: isize, j: isize) {
        let rho = self.rho.get(i, j);
        self.u.set(
            i,
            j,
            Vec5::new(
                rho,
                rho * self.vx.get(i, j),
                rho * self.vy.get(i, j),
                rho * self.e.get(i, j),
                rho * self.nut.get(i, j),
            ),
        );
    }

    /// 逐单元写入一组一致的原始变量并装配守恒量。
    #[inline]
    fn write_state(&mut self, i: isize, j: isize, s: &PrimState) {
        self.rho.set(i, j, s.rho);
        self.p.set(i, j, s.p);
        self.t.set(i, j, s.t);
        self.vx.set(i, j, s.vx);
        self.vy.set(i, j, s.vy);
        self.e.set(i, j, s.e);
        self.h.set(i, j, s.h);
        self.c.set(i, j, s.c);
        self.nut.set(i, j, s.nut);
        self.pack(i, j);
    }

    /// 用来流条件初始化全部**物理**单元。
    pub fn initialize(&mut self, cfg: &Config) {
        let s = PrimState::from_primitives(
            cfg,
            cfg.simulation.p_inf / (cfg.physics.r_gas * cfg.simulation.t_inf),
            cfg.derived.u_inf,
            cfg.derived.v_inf,
            cfg.simulation.p_inf,
            cfg.derived.nut_inf,
        );
        let mu = cfg.mu(cfg.simulation.t_inf);
        for i in 0..self.ni as isize {
            for j in 0..self.nj as isize {
                self.write_state(i, j, &s);
                self.aux.set(i, j, TurbAux { mu, chi: 0.0, fv1: 0.0 });
            }
        }
    }

    /// 把**全部**单元(含虚拟层)置为同一均匀状态。
    ///
    /// 这是自由来流保持性验证的前提:格式若离散一致,均匀场下对流残差、人工
    /// 粘性与梯度都应精确为 0。注意它会覆盖虚拟层,因此**不能**先调用
    /// [`crate::boundary::apply`] —— 固壁镜像会让贴壁处不再均匀(那是物理上正确
    /// 的,但不是这里要检验的性质)。
    pub fn set_uniform(&mut self, cfg: &Config, rho: f64, vx: f64, vy: f64, p: f64, nut: f64) {
        let s = PrimState::from_primitives(cfg, rho, vx, vy, p, nut);
        for (i, j) in self.rho.all_indices().collect::<Vec<_>>() {
            self.write_state(i, j, &s);
        }
    }
}

/// 一组自洽的原始变量。
#[derive(Clone, Copy, Debug, Default)]
pub struct PrimState {
    pub rho: f64,
    pub p: f64,
    pub t: f64,
    pub vx: f64,
    pub vy: f64,
    pub e: f64,
    pub h: f64,
    pub c: f64,
    pub nut: f64,
}

impl PrimState {
    /// 由 (ρ, u, v, p, ν̃) 补全 T、E、H、c。
    pub fn from_primitives(cfg: &Config, rho: f64, vx: f64, vy: f64, p: f64, nut: f64) -> Self {
        let gamma = cfg.physics.gamma;
        let t = p / (cfg.physics.r_gas * rho);
        let e = p / (rho * (gamma - 1.0)) + 0.5 * (vx * vx + vy * vy);
        Self {
            rho,
            p,
            t,
            vx,
            vy,
            e,
            h: e + p / rho,
            c: (gamma * cfg.physics.r_gas * t).sqrt(),
            nut,
        }
    }
}

/// 非物理状态(密度或压力非正)。
#[derive(Debug, Clone, Copy)]
pub struct NonPhysical {
    pub i: isize,
    pub j: isize,
    pub rho: f64,
    pub p: f64,
}

impl std::fmt::Display for NonPhysical {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "non-physical state at cell ({}, {}): rho = {:e}, p = {:e}",
            self.i, self.j, self.rho, self.p
        )
    }
}

impl std::error::Error for NonPhysical {}

/// 面上的工作量。
///
/// 注意这里**不存**面上的守恒量 —— 它只是计算无粘通量的中间值,直接在
/// [`crate::convection`] 的循环里用局部变量算掉,省一个数组和一遍访存。
#[derive(Clone, Debug)]
pub struct FaceWork {
    /// 无粘通量 F·n
    pub flux: Vec5Field,
    /// 粘性/湍流扩散通量
    pub diff: Vec5Field,
    /// JST 人工粘性
    pub dissipation: Vec5Field,
    /// 面谱半径 λf
    pub lambda: F64Field,
    /// 自适应耗散系数
    pub eps: Field<Eps>,
}

impl FaceWork {
    pub fn new(ni: usize, nj: usize) -> Self {
        Self {
            flux: Field::new(ni, nj, 0),
            diff: Field::new(ni, nj, 0),
            dissipation: Field::new(ni, nj, 0),
            lambda: Field::new(ni, nj, 0),
            eps: Field::new(ni, nj, 0),
        }
    }
}

/// tau 面 + n 面的工作量,外加 JST 激波探测器。
#[derive(Clone, Debug)]
pub struct Faces {
    /// 周向面,`(NI+1) x NJ`
    pub tau: FaceWork,
    /// 径向面,`NI x NJ`
    pub nrm: FaceWork,
    /// 以**单元**为中心的压力探测器,i 方向
    pub sensor_i: F64Field,
    /// 同上,j 方向
    pub sensor_j: F64Field,
    /// 逐单元的 `V/Δt_local`,面谱半径取它的两侧平均(避免在面循环里重复做除法)
    pub spec_ratio: F64Field,
}

impl Faces {
    pub fn new(ni: usize, nj: usize, halo: usize) -> Self {
        Self {
            tau: FaceWork::new(ni + 1, nj),
            nrm: FaceWork::new(ni, nj),
            // 探测器以单元为中心,ε² 的四点取值需要 [-2, N+1],halo=3 足够
            sensor_i: Field::new(ni, nj, halo),
            sensor_j: Field::new(ni, nj, halo),
            spec_ratio: Field::new(ni, nj, halo),
        }
    }
}

/// 便于整体传递的求解域。
#[derive(Clone, Debug)]
pub struct Domain {
    pub geom: Geometry,
    pub cells: Cells,
    pub faces: Faces,
}

impl Domain {
    pub fn new(geom: Geometry, halo: usize) -> Self {
        let (ni, nj) = (geom.ni, geom.nj);
        Self {
            geom,
            cells: Cells::new(ni, nj, halo),
            faces: Faces::new(ni, nj, halo),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::Mesh;

    fn setup() -> (Config, Domain) {
        let cfg = Config::from_str(include_str!("../../config.json")).unwrap();
        let mesh = Mesh::parse(include_str!("../../fangdata.txt")).unwrap();
        let geom = Geometry::build(&mesh, cfg.simulation.halo);
        let mut dom = Domain::new(geom, cfg.simulation.halo);
        dom.cells.initialize(&cfg);
        (cfg, dom)
    }

    #[test]
    fn initialization_is_uniform_and_consistent() {
        let (cfg, dom) = setup();
        let c = &dom.cells;
        for (i, j) in c.rho.interior() {
            assert!((c.rho.get(i, j) - cfg.derived.rho_inf).abs() < 1e-12);
            assert!((c.p.get(i, j) - cfg.simulation.p_inf).abs() < 1e-9);
            assert!((c.t.get(i, j) - cfg.simulation.t_inf).abs() < 1e-12);
            let eos = c.rho.get(i, j) * cfg.physics.r_gas * c.t.get(i, j);
            assert!((c.p.get(i, j) - eos).abs() < 1e-6 * eos);
        }
    }

    #[test]
    fn pack_matches_definition() {
        let (_, dom) = setup();
        let c = &dom.cells;
        for (i, j) in c.u.interior() {
            let u = c.u.get(i, j);
            let rho = c.rho.get(i, j);
            assert!((u[0] - rho).abs() < 1e-15);
            assert!((u[1] - rho * c.vx.get(i, j)).abs() < 1e-12);
            assert!((u[3] - rho * c.e.get(i, j)).abs() < 1e-9);
            assert!((u[4] - rho * c.nut.get(i, j)).abs() < 1e-20);
        }
    }

    #[test]
    fn initial_nut_is_ten_percent_of_kinematic_viscosity() {
        let (cfg, dom) = setup();
        let want = 0.1 * cfg.mu(cfg.simulation.t_inf) / cfg.derived.rho_inf;
        assert!((dom.cells.nut.get(0, 0) - want).abs() < 1e-18);
        assert!((cfg.derived.nut_inf - want).abs() < 1e-20);
    }

    #[test]
    fn prim_state_is_thermodynamically_consistent() {
        let cfg = Config::from_str(include_str!("../../config.json")).unwrap();
        let s = PrimState::from_primitives(&cfg, 1.2, 70.0, -15.0, 1.0e5, 2e-4);
        assert!((s.p - s.rho * cfg.physics.r_gas * s.t).abs() < 1e-9);
        assert!((s.h - (s.e + s.p / s.rho)).abs() < 1e-9);
        assert!((s.c - (cfg.physics.gamma * s.p / s.rho).sqrt()).abs() < 1e-9);
    }

    #[test]
    fn set_uniform_covers_the_halo() {
        let (cfg, mut dom) = setup();
        dom.cells.set_uniform(&cfg, 1.5, 10.0, 20.0, 9e4, 1e-4);
        for (i, j) in dom.cells.rho.all_indices() {
            assert_eq!(dom.cells.rho.get(i, j), 1.5);
            assert_eq!(dom.cells.vy.get(i, j), 20.0);
        }
    }

    #[test]
    fn halo_is_addressable_for_all_cell_fields() {
        let (_, mut dom) = setup();
        let h = 3;
        let (ni, nj) = (dom.cells.ni as isize, dom.cells.nj as isize);
        dom.cells.rho.set(-h, -h, 1.0);
        dom.cells.rho.set(ni + h - 1, nj + h - 1, 2.0);
        assert_eq!(dom.cells.rho.get(-h, -h), 1.0);
        assert_eq!(dom.cells.rho.get(ni + h - 1, nj + h - 1), 2.0);
    }
}
