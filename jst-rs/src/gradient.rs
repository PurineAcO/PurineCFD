//! Green-Gauss 单元梯度。
//!
//! ```text
//! ∇φ|ᵢⱼ = (1/V) Σ_faces φ_face · n_face
//! ```
//!
//! 面上的 φ 取相邻两单元的算术平均(一阶中心)。均匀场下 `Σ±n ≡ 0` 保证梯度精确
//! 为 0 —— 由本模块的自由来流用例把关。
//!
//! 八个分量打包进 [`Grad`] 一起写:四个变量的梯度总是被粘性项与源项同时消费,
//! 打包后本 kernel 只写一个数组,省掉八路并行迭代器的同步切分开销。
//!
//! 注意:Python 基线**从未调用**梯度计算(`BUGS.md` A5),导致粘性项与湍流源项
//! 恒为 0,N-S 方程静默退化成 Euler 方程。

use rayon::iter::ParallelIterator;

use crate::geometry::Geometry;
use crate::state::{Cells, Grad};

/// 一个标量场在四个面上的 Green-Gauss 贡献。
macro_rules! gg {
    ($src:expr, $i:expr, $j:expr, $jm:expr, $jp:expr,
     $up:expr, $dn:expr, $lf:expr, $rt:expr, $inv_v:expr) => {{
        let c = $src.get($i, $j);
        let f_up = 0.5 * (c + $src.get($i + 1, $j));
        let f_dn = 0.5 * (c + $src.get($i - 1, $j));
        let f_lf = 0.5 * (c + $src.get($i, $jm));
        let f_rt = 0.5 * (c + $src.get($i, $jp));
        (
            (f_up * $up.nx - f_dn * $dn.nx + f_rt * $rt.nx - f_lf * $lf.nx) * $inv_v,
            (f_up * $up.ny - f_dn * $dn.ny + f_rt * $rt.ny - f_lf * $lf.ny) * $inv_v,
        )
    }};
}

/// 计算 u、v、T、ν̃ 在**物理单元**上的梯度,随后按边界条件填充第一层虚拟单元。
pub fn compute(geom: &Geometry, cells: &mut Cells) {
    let nj = geom.nj as isize;
    let (inv_vol, tau, nrm) = (&geom.inv_vol, &geom.tau, &geom.nrm);
    // 拆借:只写 grad,读其余四个数组 —— 借用检查证明无别名
    let Cells {
        grad,
        vx,
        vy,
        t,
        nut,
        ..
    } = cells;
    let (vx, vy, t, nut) = (&*vx, &*vy, &*t, &*nut);

    grad.par_interior_rows_mut().for_each(|(i, mut row)| {
        for j in 0..nj {
            let jm = j - 1;
            let jp = j + 1;
            let up = tau.at(i + 1, j);
            let dn = tau.at(i, j);
            let lf = nrm.at(i, j);
            let rt = nrm.at(i, if jp < nj { jp } else { 0 });
            let inv_v = inv_vol.get(i, j);

            let (dudx, dudy) = gg!(vx, i, j, jm, jp, up, dn, lf, rt, inv_v);
            let (dvdx, dvdy) = gg!(vy, i, j, jm, jp, up, dn, lf, rt, inv_v);
            let (dtdx, dtdy) = gg!(t, i, j, jm, jp, up, dn, lf, rt, inv_v);
            let (dnutdx, dnutdy) = gg!(nut, i, j, jm, jp, up, dn, lf, rt, inv_v);
            row[j] = Grad {
                dudx,
                dudy,
                dvdx,
                dvdy,
                dtdx,
                dtdy,
                dnutdx,
                dnutdy,
            };
        }
    });

    fill_ghost_gradients(cells);
}

/// 第一层虚拟单元的梯度 —— 只有它们会被 [`crate::viscous`] 的面平均用到。
///
/// * 固壁:速度与 ν̃ 的梯度延拓自贴壁单元;温度梯度置 0(绝热壁)。
/// * 远场:全部置 0(粘性影响在远场可忽略)。
/// * 周向:按周期直接复制。
fn fill_ghost_gradients(cells: &mut Cells) {
    let (ni, nj) = (cells.ni as isize, cells.nj as isize);
    let g = &mut cells.grad;

    for j in 0..nj {
        let mut wall = g.get(0, j);
        wall.dtdx = 0.0;
        wall.dtdy = 0.0;
        g.set(-1, j, wall);
        g.set(ni, j, Grad::default());
    }
    for i in 0..ni {
        g.set(i, -1, g.get(i, nj - 1));
        g.set(i, nj, g.get(i, 0));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::mesh::Mesh;
    use crate::state::Domain;

    fn setup() -> (Config, Domain) {
        let cfg = Config::from_str(include_str!("../../config.json")).unwrap();
        let mesh = Mesh::parse(include_str!("../../fangdata.txt")).unwrap();
        let geom = Geometry::build(&mesh, cfg.simulation.halo);
        let mut dom = Domain::new(geom, cfg.simulation.halo);
        dom.cells.initialize(&cfg);
        crate::boundary::apply(&cfg, &dom.geom, &mut dom.cells);
        (cfg, dom)
    }

    /// 均匀场的梯度必须精确为 0 —— 只依赖度量闭合,是最锐利的索引/符号检查。
    #[test]
    fn uniform_field_has_zero_gradient() {
        let (cfg, mut dom) = setup();
        dom.cells.set_uniform(&cfg, 1.176, 69.4, 17.3, 101325.0, 1.5e-4);
        compute(&dom.geom, &mut dom.cells);
        let vol = dom.geom.vol.get(0, 0);
        let scale = dom.cells.vx.get(0, 0).abs() / vol;
        let t_scale = dom.cells.t.get(0, 0).abs() / vol;
        for (i, j) in dom.cells.grad.interior() {
            let g = dom.cells.grad.get(i, j);
            assert!(g.dudx.abs() < 1e-13 * scale, "dudx at ({i},{j}) = {:e}", g.dudx);
            assert!(g.dudy.abs() < 1e-13 * scale);
            assert!(g.dvdx.abs() < 1e-13 * scale);
            assert!(g.dvdy.abs() < 1e-13 * scale);
            assert!(g.dtdx.abs() < 1e-13 * t_scale);
            assert!(g.dtdy.abs() < 1e-13 * t_scale);
        }
    }

    /// 网格收敛性研究:线性场上的梯度误差随加密而减小。
    ///
    /// 这里**不能**要求"线性场精确复现" —— 面值取的是两个单元中心的算术平均,
    /// 它等于线性函数在两**形心**中点上的值,而非面中点上的值;非均匀网格上
    /// 二者不重合,故简单平均的 Green-Gauss 在最坏单元上只有一阶精度。
    /// 实测(见 `examples/gradient_convergence.rs`):
    ///
    /// ```text
    ///   9x32 → 129x512:  L1 每次加密 ×3.4 → ×3.9 (≈二阶)
    ///                    L∞ 每次加密 ×1.5 → ×1.8 (≈一阶,受最扭曲的壁面单元支配)
    /// ```
    ///
    /// 缺 1/V、差因子 2、法向符号写反之类的错误都会让误差**不收敛**,立刻暴露。
    #[test]
    fn linear_field_gradient_converges_under_refinement() {
        let cfg = Config::from_str(include_str!("../../config.json")).unwrap();
        let (ax, ay) = (3.7, -1.9);

        let err_at = |rings: usize, nj: usize| -> (f64, f64) {
            let mut txt = format!("{rings} {nj}\n");
            for i in 0..rings {
                let s = i as f64 / (rings - 1) as f64;
                let (a, b) = (1.0 + s * 4.0, 0.5 + s * 4.5);
                for j in 0..nj {
                    let th = 2.0 * std::f64::consts::PI * j as f64 / nj as f64;
                    txt += &format!("{:.12} {:.12}\n", a * th.cos(), b * th.sin());
                }
            }
            let mesh = Mesh::parse(&txt).unwrap();
            let geom = Geometry::build(&mesh, cfg.simulation.halo);
            let mut dom = Domain::new(geom, cfg.simulation.halo);
            dom.cells.initialize(&cfg);

            let (ni, njj) = (dom.cells.ni as isize, dom.cells.nj as isize);
            let f = |x: f64, y: f64| ax * x + ay * y;
            for i in 0..ni {
                for j in 0..njj {
                    dom.cells
                        .vx
                        .set(i, j, f(dom.geom.cx.get(i, j), dom.geom.cy.get(i, j)));
                }
            }
            // 关于边界面中点反射,使边界面上的平均值精确落在解析值上
            for j in 0..njj {
                let w = dom.geom.tau.get(0, j);
                dom.cells
                    .vx
                    .set(-1, j, 2.0 * f(w.mx, w.my) - dom.cells.vx.get(0, j));
                let o = dom.geom.tau.get(ni, j);
                dom.cells
                    .vx
                    .set(ni, j, 2.0 * f(o.mx, o.my) - dom.cells.vx.get(ni - 1, j));
            }
            for i in 0..ni {
                dom.cells.vx.set(i, -1, dom.cells.vx.get(i, njj - 1));
                dom.cells.vx.set(i, njj, dom.cells.vx.get(i, 0));
            }

            compute(&dom.geom, &mut dom.cells);
            let errs = || {
                dom.cells
                    .grad
                    .interior()
                    .map(|(i, j)| (dom.cells.grad.get(i, j).dudx - ax).abs() / ax.abs())
            };
            let n = (ni * njj) as f64;
            (errs().fold(0.0f64, f64::max), errs().sum::<f64>() / n)
        };

        let (linf_c, l1_c) = err_at(17, 64);
        let (linf_f, l1_f) = err_at(33, 128);
        assert!(
            l1_f < l1_c / 3.0,
            "L1 gradient error not converging at ~2nd order: {l1_c:e} -> {l1_f:e}"
        );
        assert!(
            linf_f < linf_c / 1.4,
            "Linf gradient error not converging: {linf_c:e} -> {linf_f:e}"
        );
    }

    #[test]
    fn ghost_gradients_are_filled() {
        let (_, mut dom) = setup();
        for (i, j) in dom.cells.vx.interior().collect::<Vec<_>>() {
            dom.cells.vx.set(i, j, 10.0 + i as f64 + 0.5 * j as f64);
        }
        compute(&dom.geom, &mut dom.cells);
        let (ni, nj) = (dom.cells.ni as isize, dom.cells.nj as isize);
        let g = &dom.cells.grad;
        for j in 0..nj {
            assert_eq!(g.get(-1, j).dudx, g.get(0, j).dudx);
            assert_eq!(g.get(-1, j).dtdx, 0.0); // 绝热壁
            assert_eq!(g.get(ni, j).dudx, 0.0); // 远场
        }
        for i in 0..ni {
            assert_eq!(g.get(i, -1).dudx, g.get(i, nj - 1).dudx);
            assert_eq!(g.get(i, nj).dudx, g.get(i, 0).dudx);
        }
    }
}
