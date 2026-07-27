//! # JST + Spalart-Allmaras 二维有限体积求解器
//!
//! O 型结构网格上的可压 Navier-Stokes 求解器:空间用格心有限体积中心格式 +
//! JST 标量人工粘性,湍流用 Spalart-Allmaras 一方程模型,时间用显式五级
//! Runge-Kutta 推进到定常。
//!
//! 本 crate 是同目录 Python 实现的重写。数值格式逐项对齐(`tests/golden.rs`
//! 直接读 Python 导出的参考数据做交叉验证),但架构做了两处根本性调整:
//!
//! ## 1. Halo 化的下标空间
//!
//! Python 把虚拟单元追加在数组尾部,于是每个 kernel 都要自己写一遍
//! "壁面取这个、远场取那个、切割线取另一个"的映射。这类手写映射贡献了
//! `BUGS.md` 里的四个数值错误。这里把单元下标扩展到 `[-H, N+H)`,虚拟单元住在
//! 负下标上,边界条件收敛成唯一的 [`boundary::apply`],其余 kernel 全是无特判的
//! 矩形循环 —— 一整类 bug 在结构上被消除。
//!
//! ## 2. SoA + 按行并行
//!
//! 每个物理量一个连续 `Vec`(见 [`field::Field`])。顺序扫描只加载真正用到的量,
//! 编译器可自动向量化;而"输出一个数组、输入若干别的数组"这一模式让借用检查
//! 直接证明无别名,rayon 并行不需要任何 `unsafe`。所有并行 kernel 都是
//! 「每个输出元素只写一次、值只由输入决定」,因此结果与线程数无关,逐位可复现。
//!
//! ## 模块分层
//!
//! ```text
//! mesh ─► geometry ─┐
//! config ───────────┼─► state ─► solver ─► (bin/jst)
//!                   │      ▲
//!         boundary ─┘      │  kernels:
//!                          ├── timestep     当地/全局时间步
//!                          ├── convection   无粘对流通量
//!                          ├── gradient     Green-Gauss 梯度
//!                          ├── viscous      粘性应力 + 湍流扩散
//!                          ├── dissipation  JST 人工粘性
//!                          └── source       S-A 源项
//! ```
//!
//! 每个 kernel 都是接收 `&Geometry`、`&Cells`、`&mut 输出` 的自由函数,可以单独
//! 测试与基准,替换其中任何一个(比如把中心格式换成 Roe 迎风)不牵动其余部分。
//!
//! ## 用法
//!
//! ```no_run
//! use jst::{Config, Mesh, Solver};
//!
//! let cfg = Config::from_path("config.json").unwrap();
//! let mesh = Mesh::from_path("fangdata.txt").unwrap();
//! let mut solver = Solver::new(cfg, &mesh);
//! let report = solver.run(Some(1000), |step, res| {
//!     if step % 100 == 0 {
//!         println!("step {step}: residual {res:.3e}");
//!     }
//! }).unwrap();
//! println!("converged: {}", report.converged);
//! ```

pub mod boundary;
pub mod config;
pub mod convection;
pub mod dissipation;
pub mod field;
pub mod geometry;
pub mod gradient;
pub mod mesh;
pub mod solver;
pub mod source;
pub mod state;
pub mod timestep;
pub mod viscous;

pub use config::Config;
pub use field::{Field, Vec5};
pub use geometry::Geometry;
pub use mesh::{Mesh, MeshError};
pub use solver::{RunReport, Solver};
pub use state::{Cells, DiffTensor, Domain, Eps, Faces, Grad, NonPhysical, PrimState, TurbAux};
