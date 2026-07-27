//! 有限体积法的几何度量:单元面积/形心、面法向/中点、壁面距离。
//!
//! # 下标约定
//!
//! 记 `NI = n_rings-1` 为径向单元数、`NJ = n_theta` 为周向单元数。
//!
//! * 单元 `(i, j)`,`i ∈ [0,NI)`,`j ∈ [0,NJ)`;四个顶点是节点
//!   `(i,j) (i+1,j) (i+1,j+1) (i,j+1)`。
//! * **tau 面**(周向边,一圈圈的"波纹"):`i ∈ [0,NI]`,`j ∈ [0,NJ)`。
//!   tau 面 `i` 分隔单元 `i-1` 与 `i`,法向指向径向外侧。
//! * **n 面**(径向边,"波纹圈的直径"):`i ∈ [0,NI)`,`j ∈ [0,NJ)`。
//!   n 面 `j` 分隔单元 `j-1` 与 `j`,法向指向周向 +j 侧。
//!
//! 法向的**模长等于边长**(即面积加权法向),因此通量可以直接点乘法向而不必
//! 另外乘面积。
//!
//! 这套约定保证了度量闭合:`n_τ(i+1) − n_τ(i) + n_n(j+1) − n_n(j) ≡ 0`,
//! 它正是自由来流保持性的充要条件(见 `tests/properties.rs`)。

use crate::field::Field;
use crate::mesh::Mesh;

/// 单面的几何量。
#[derive(Clone, Copy, Debug, Default)]
pub struct FaceGeom {
    /// 面积加权法向的 x 分量(模长 = 边长)
    pub nx: f64,
    /// 面积加权法向的 y 分量
    pub ny: f64,
    /// 面中点 x
    pub mx: f64,
    /// 面中点 y
    pub my: f64,
}

impl FaceGeom {
    #[inline]
    pub fn length(&self) -> f64 {
        self.nx.hypot(self.ny)
    }
}

/// 全部与时间无关的几何量,setup 阶段算一次后只读。
#[derive(Clone, Debug)]
pub struct Geometry {
    pub ni: usize,
    pub nj: usize,
    /// 单元面积(二维下的"体积")
    pub vol: Field<f64>,
    /// 单元形心 x / y
    pub cx: Field<f64>,
    pub cy: Field<f64>,
    /// 单元中心到壁面的最近距离(S-A 模型的 d)
    pub wall_dist: Field<f64>,
    /// `1/V`。梯度、残差更新都要按体积归一,预先算好省掉逐单元逐级的除法
    pub inv_vol: Field<f64>,
    /// `1/d²`。S-A 源项里 `ν̃/(κ²d²)` 与 `(ν̃/d)²` 都要用
    pub inv_wall_dist_sq: Field<f64>,
    /// 周向面,`(NI+1) x NJ`
    pub tau: Field<FaceGeom>,
    /// 径向面,`NI x NJ`
    pub nrm: Field<FaceGeom>,
}

impl Geometry {
    /// 周向下标 +1 的回绕。
    #[inline(always)]
    pub fn jp1(&self, j: isize) -> isize {
        if j + 1 < self.nj as isize {
            j + 1
        } else {
            0
        }
    }

    pub fn build(mesh: &Mesh, halo: usize) -> Self {
        let (ni, nj) = (mesh.ni(), mesh.nj());
        let mut g = Self {
            ni,
            nj,
            // 单元量需要 halo:虚拟单元也参与 JST 模板,vol 用于源项缩放
            vol: Field::new(ni, nj, halo),
            cx: Field::new(ni, nj, halo),
            cy: Field::new(ni, nj, halo),
            wall_dist: Field::new(ni, nj, halo),
            inv_vol: Field::new(ni, nj, halo),
            inv_wall_dist_sq: Field::new(ni, nj, halo),
            // 面量不需要 halo:所有 kernel 只在 [0,NI]x[0,NJ) 上访问面
            tau: Field::new(ni + 1, nj, 0),
            nrm: Field::new(ni, nj, 0),
        };
        g.build_cells(mesh);
        g.build_faces(mesh);
        g.build_wall_distance();
        for i in 0..ni as isize {
            for j in 0..nj as isize {
                g.inv_vol.set(i, j, 1.0 / g.vol.get(i, j));
                let d = g.wall_dist.get(i, j);
                g.inv_wall_dist_sq.set(i, j, 1.0 / (d * d));
            }
        }
        g
    }

    /// 单元面积与形心。
    ///
    /// 面积用"对角线叉积"公式 `A = ½|AC × DB|` —— 对任意不自交的四边形都精确。
    /// 形心用多边形形心的标准公式(shoelace 加权)。
    fn build_cells(&mut self, mesh: &Mesh) {
        for i in 0..self.ni {
            for j in 0..self.nj {
                let a = mesh.node(i, j);
                let b = mesh.node(i + 1, j);
                let c = mesh.node(i + 1, j + 1);
                let d = mesh.node(i, j + 1);

                let (d1x, d1y) = (c.x - a.x, c.y - a.y);
                let (d2x, d2y) = (b.x - d.x, b.y - d.y);
                let vol = 0.5 * (d1x * d2y - d1y * d2x).abs();

                let cr = [
                    a.x * b.y - b.x * a.y,
                    b.x * c.y - c.x * b.y,
                    c.x * d.y - d.x * c.y,
                    d.x * a.y - a.x * d.y,
                ];
                let signed = 0.5 * (cr[0] + cr[1] + cr[2] + cr[3]);
                let sx = (a.x + b.x) * cr[0]
                    + (b.x + c.x) * cr[1]
                    + (c.x + d.x) * cr[2]
                    + (d.x + a.x) * cr[3];
                let sy = (a.y + b.y) * cr[0]
                    + (b.y + c.y) * cr[1]
                    + (c.y + d.y) * cr[2]
                    + (d.y + a.y) * cr[3];

                let (i, j) = (i as isize, j as isize);
                self.vol.set(i, j, vol);
                if signed.abs() > 1e-30 {
                    self.cx.set(i, j, sx / (6.0 * signed));
                    self.cy.set(i, j, sy / (6.0 * signed));
                }
            }
        }
    }

    /// 面法向与中点。
    fn build_faces(&mut self, mesh: &Mesh) {
        // tau 面:沿周向的边 (i,j) → (i,j+1),法向 (dy, −dx) 指向径向外侧
        for i in 0..=self.ni {
            for j in 0..self.nj {
                let a = mesh.node(i, j);
                let b = mesh.node(i, j + 1);
                let (dx, dy) = (b.x - a.x, b.y - a.y);
                self.tau.set(
                    i as isize,
                    j as isize,
                    FaceGeom {
                        nx: dy,
                        ny: -dx,
                        mx: 0.5 * (a.x + b.x),
                        my: 0.5 * (a.y + b.y),
                    },
                );
            }
        }
        // n 面:沿径向的边 (i,j) → (i+1,j),法向 (−dy, dx) 指向周向 +j 侧
        for i in 0..self.ni {
            for j in 0..self.nj {
                let a = mesh.node(i, j);
                let b = mesh.node(i + 1, j);
                let (dx, dy) = (b.x - a.x, b.y - a.y);
                self.nrm.set(
                    i as isize,
                    j as isize,
                    FaceGeom {
                        nx: -dy,
                        ny: dx,
                        mx: 0.5 * (a.x + b.x),
                        my: 0.5 * (a.y + b.y),
                    },
                );
            }
        }
    }

    /// 每个单元中心到壁面的最近距离。
    ///
    /// 壁面即第 0 层 tau 面。只在周向 ±`window` 的范围内搜索:O 型网格上最近的
    /// 壁面点几乎总在径向正下方,`window` 取到约 ±20% 周长已远超需要,可把
    /// 复杂度从 O(NI·NJ²) 降到 O(NI·NJ·window)。
    fn build_wall_distance(&mut self) {
        let nj = self.nj;
        let window = (15.max(nj / 5)).min(nj / 2) as isize;
        let wall: Vec<(f64, f64)> = (0..nj)
            .map(|j| {
                let f = self.tau.get(0, j as isize);
                (f.mx, f.my)
            })
            .collect();

        for i in 0..self.ni as isize {
            for j in 0..nj as isize {
                let (px, py) = (self.cx.get(i, j), self.cy.get(i, j));
                let mut best = f64::INFINITY;
                for dk in -window..=window {
                    let k = (j + dk).rem_euclid(nj as isize) as usize;
                    let d = ((px - wall[k].0).powi(2) + (py - wall[k].1).powi(2)).sqrt();
                    if d < best {
                        best = d;
                    }
                }
                self.wall_dist.set(i, j, best);
            }
        }
    }

    /// 全部单元面积之和。
    pub fn total_area(&self) -> f64 {
        self.vol.interior().map(|(i, j)| self.vol.get(i, j)).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fangdata() -> Geometry {
        let m = Mesh::parse(include_str!("../../fangdata.txt")).unwrap();
        Geometry::build(&m, 3)
    }

    /// 度量闭合 —— 自由来流保持性的充要几何条件。
    #[test]
    fn metric_closure() {
        let g = fangdata();
        let scale = (0..=g.ni)
            .flat_map(|i| (0..g.nj).map(move |j| (i, j)))
            .map(|(i, j)| g.tau.get(i as isize, j as isize).length())
            .fold(0.0f64, f64::max);
        for i in 0..g.ni as isize {
            for j in 0..g.nj as isize {
                let jp = g.jp1(j);
                let sx = g.tau.get(i + 1, j).nx - g.tau.get(i, j).nx + g.nrm.get(i, jp).nx
                    - g.nrm.get(i, j).nx;
                let sy = g.tau.get(i + 1, j).ny - g.tau.get(i, j).ny + g.nrm.get(i, jp).ny
                    - g.nrm.get(i, j).ny;
                assert!(sx.abs() < 1e-13 * scale, "closure x at ({i},{j}): {sx:e}");
                assert!(sy.abs() < 1e-13 * scale, "closure y at ({i},{j}): {sy:e}");
            }
        }
    }

    #[test]
    fn total_area_equals_polygon_annulus() {
        let m = Mesh::parse(include_str!("../../fangdata.txt")).unwrap();
        let g = Geometry::build(&m, 3);
        let shoelace = |ring: usize| -> f64 {
            0.5 * (0..m.n_theta())
                .map(|j| {
                    let (a, b) = (m.node(ring, j), m.node(ring, j + 1));
                    a.x * b.y - b.x * a.y
                })
                .sum::<f64>()
        };
        let want = shoelace(m.n_rings() - 1) - shoelace(0);
        assert!((g.total_area() - want).abs() < 1e-12 * want.abs());
    }

    #[test]
    fn volumes_positive() {
        let g = fangdata();
        assert!(g.vol.interior().all(|(i, j)| g.vol.get(i, j) > 0.0));
    }

    #[test]
    fn tau_normal_points_outward() {
        let g = fangdata();
        for i in 0..=g.ni as isize {
            for j in 0..g.nj as isize {
                let f = g.tau.get(i, j);
                assert!(f.nx * f.mx + f.ny * f.my > 0.0, "tau ({i},{j}) inward");
            }
        }
    }

    #[test]
    fn n_normal_points_counterclockwise() {
        let g = fangdata();
        for i in 0..g.ni as isize {
            for j in 0..g.nj as isize {
                let f = g.nrm.get(i, j);
                assert!(f.nx * -f.my + f.ny * f.mx > 0.0, "n ({i},{j}) flipped");
            }
        }
    }

    #[test]
    fn normal_magnitude_equals_edge_length() {
        let m = Mesh::parse(include_str!("../../fangdata.txt")).unwrap();
        let g = Geometry::build(&m, 3);
        for i in 0..=g.ni {
            for j in 0..g.nj {
                let (a, b) = (m.node(i, j), m.node(i, j + 1));
                let want = (b.x - a.x).hypot(b.y - a.y);
                let got = g.tau.get(i as isize, j as isize).length();
                assert!((got - want).abs() < 1e-14 * want);
            }
        }
    }

    #[test]
    fn wall_distance_positive_and_increasing() {
        let g = fangdata();
        for j in 0..g.nj as isize {
            let mut prev = 0.0;
            for i in 0..g.ni as isize {
                let d = g.wall_dist.get(i, j);
                assert!(d > prev, "wall distance not increasing at ({i},{j})");
                prev = d;
            }
        }
    }

    #[test]
    fn centroid_inside_bounding_box() {
        let m = Mesh::parse(include_str!("../../fangdata.txt")).unwrap();
        let g = Geometry::build(&m, 3);
        for i in 0..g.ni {
            for j in 0..g.nj {
                let pts = [
                    m.node(i, j),
                    m.node(i + 1, j),
                    m.node(i + 1, j + 1),
                    m.node(i, j + 1),
                ];
                let (xlo, xhi) = pts.iter().fold((f64::MAX, f64::MIN), |(l, h), p| {
                    (l.min(p.x), h.max(p.x))
                });
                let (ylo, yhi) = pts.iter().fold((f64::MAX, f64::MIN), |(l, h), p| {
                    (l.min(p.y), h.max(p.y))
                });
                let (cx, cy) = (g.cx.get(i as isize, j as isize), g.cy.get(i as isize, j as isize));
                assert!(cx >= xlo - 1e-12 && cx <= xhi + 1e-12);
                assert!(cy >= ylo - 1e-12 && cy <= yhi + 1e-12);
            }
        }
    }
}
