//! 带 halo(虚拟层)的二维数组,以及五分量守恒向量 [`Vec5`].
//!
//! # 为什么是 halo
//!
//! Python 基线把虚拟单元**追加**在物理数组之后:壁面 ghost 放在 `CellList[i_total..]`,
//! 远场 ghost 放在更后面,周向 ghost 追加在每行末尾。于是每个 kernel 都得自己写一遍
//! "如果 i==1 取 `CellList[i_total]`、如果 j==j_total 取 `CellList[j_total+IM+1]`……"
//! 这类映射 —— 全项目重复了十几次,而 `BUGS.md` 里 B4/B5/B6/B8 四个数值错误
//! **全部**出自这些手写映射的笔误。
//!
//! 这里改成:单元的下标空间直接扩展到 `[-H, N+H)`,虚拟层就住在负下标和越界下标上。
//! 边界条件收敛成唯一一处 [`crate::boundary::apply`],此后每个 kernel 都是不带任何
//! 特判的矩形循环。索引写错的整类 bug 在结构上被消掉了。
//!
//! ```text
//!      j = -3 -2 -1 | 0  1  ...  NJ-1 | NJ NJ+1 NJ+2
//! i = -3   ┌────────┼────────────────┼────────────┐
//!  ...     │  halo  │                │    halo    │
//! i = -1   │        │                │            │
//!          ├────────┼────────────────┼────────────┤
//! i =  0   │  halo  │    物理单元     │    halo    │
//!  ...     │        │   NI x NJ      │            │
//! i = NI-1 │        │                │            │
//!          ├────────┼────────────────┼────────────┤
//! i = NI   │  halo  │      halo      │    halo    │
//! ```

use std::ops::{Add, AddAssign, Index, IndexMut, Mul, Neg, Sub};

use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::ParallelSliceMut;

/// 五分量守恒向量 `[ρ, ρu, ρv, ρE, ρν̃]`。
///
/// 定义了完整的算术运算符,好让格式公式在代码里保持数学写法 ——
/// 例如 JST 耗散项可以直接写成 `lam * (d1u * eps2 - d3u * eps4)`。
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct Vec5(pub [f64; 5]);

/// 守恒向量的分量下标。
pub mod comp {
    /// 密度 ρ
    pub const RHO: usize = 0;
    /// x 方向动量 ρu
    pub const MX: usize = 1;
    /// y 方向动量 ρv
    pub const MY: usize = 2;
    /// 总能 ρE
    pub const RHO_E: usize = 3;
    /// 湍流工作变量 ρν̃
    pub const RHO_NU: usize = 4;
}

impl Vec5 {
    pub const ZERO: Self = Vec5([0.0; 5]);

    #[inline]
    pub const fn new(rho: f64, mx: f64, my: f64, rho_e: f64, rho_nu: f64) -> Self {
        Vec5([rho, mx, my, rho_e, rho_nu])
    }

    /// 各分量绝对值的最大值,用于收敛/容差判断。
    #[inline]
    pub fn amax(&self) -> f64 {
        self.0.iter().fold(0.0f64, |m, v| m.max(v.abs()))
    }

    #[inline]
    pub fn is_finite(&self) -> bool {
        self.0.iter().all(|v| v.is_finite())
    }
}

impl Index<usize> for Vec5 {
    type Output = f64;
    #[inline]
    fn index(&self, k: usize) -> &f64 {
        &self.0[k]
    }
}

impl IndexMut<usize> for Vec5 {
    #[inline]
    fn index_mut(&mut self, k: usize) -> &mut f64 {
        &mut self.0[k]
    }
}

macro_rules! impl_binop {
    ($trait:ident, $method:ident, $op:tt) => {
        impl $trait for Vec5 {
            type Output = Vec5;
            #[inline]
            fn $method(self, r: Vec5) -> Vec5 {
                let (a, b) = (self.0, r.0);
                Vec5([a[0] $op b[0], a[1] $op b[1], a[2] $op b[2], a[3] $op b[3], a[4] $op b[4]])
            }
        }
    };
}
impl_binop!(Add, add, +);
impl_binop!(Sub, sub, -);

impl Mul<f64> for Vec5 {
    type Output = Vec5;
    #[inline]
    fn mul(self, s: f64) -> Vec5 {
        let a = self.0;
        Vec5([a[0] * s, a[1] * s, a[2] * s, a[3] * s, a[4] * s])
    }
}

impl Mul<Vec5> for f64 {
    type Output = Vec5;
    #[inline]
    fn mul(self, v: Vec5) -> Vec5 {
        v * self
    }
}

impl Neg for Vec5 {
    type Output = Vec5;
    #[inline]
    fn neg(self) -> Vec5 {
        self * -1.0
    }
}

impl AddAssign for Vec5 {
    #[inline]
    fn add_assign(&mut self, r: Vec5) {
        *self = *self + r;
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// 带 halo 的二维数组,下标为有符号整数,合法范围 `[-halo, n+halo)`。
///
/// 内部是行主序的一维 `Vec`(i 为慢维、j 为快维),因此沿 j 遍历是连续访问。
#[derive(Clone, Debug)]
pub struct Field<T> {
    data: Vec<T>,
    ni: usize,
    nj: usize,
    halo: usize,
    stride: usize,
}

impl<T: Clone + Default> Field<T> {
    /// 建立 `ni x nj` 的物理区,四周各留 `halo` 层。
    pub fn new(ni: usize, nj: usize, halo: usize) -> Self {
        let stride = nj + 2 * halo;
        Self {
            data: vec![T::default(); (ni + 2 * halo) * stride],
            ni,
            nj,
            halo,
            stride,
        }
    }
}

impl<T> Field<T> {
    #[inline]
    pub fn ni(&self) -> usize {
        self.ni
    }
    #[inline]
    pub fn nj(&self) -> usize {
        self.nj
    }
    #[inline]
    pub fn halo(&self) -> usize {
        self.halo
    }

    /// 有符号下标 → 线性下标。越界时 panic(debug 与 release 都检查:
    /// 这里的下标算术正是最容易写错的地方,不值得为省一次比较冒风险)。
    #[inline(always)]
    pub fn offset(&self, i: isize, j: isize) -> usize {
        let h = self.halo as isize;
        debug_assert!(
            i >= -h && i < self.ni as isize + h && j >= -h && j < self.nj as isize + h,
            "index ({i}, {j}) out of halo range for {}x{} field with halo {}",
            self.ni,
            self.nj,
            self.halo
        );
        ((i + h) as usize) * self.stride + (j + h) as usize
    }

    #[inline(always)]
    pub fn at(&self, i: isize, j: isize) -> &T {
        &self.data[self.offset(i, j)]
    }

    #[inline(always)]
    pub fn at_mut(&mut self, i: isize, j: isize) -> &mut T {
        let o = self.offset(i, j);
        &mut self.data[o]
    }

    #[inline(always)]
    pub fn set(&mut self, i: isize, j: isize, v: T) {
        let o = self.offset(i, j);
        self.data[o] = v;
    }

    /// 第 `i` 行(含 halo 列)的可变切片。行之间互不重叠,是 rayon 并行的天然单位。
    #[inline]
    pub fn row_mut(&mut self, i: isize) -> &mut [T] {
        let start = self.offset(i, -(self.halo as isize));
        &mut self.data[start..start + self.stride]
    }

    /// 底层连续存储,含 halo。
    #[inline]
    pub fn raw(&self) -> &[T] {
        &self.data
    }

    #[inline]
    pub fn raw_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// 行内 j 从 `-halo` 起算的偏移量,配合 [`Field::rows_mut`] 使用。
    #[inline]
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// 按行切分成可并行的可变切片(含 halo 行)。
    #[inline]
    pub fn rows_mut(&mut self) -> std::slice::ChunksExactMut<'_, T> {
        let s = self.stride;
        self.data.chunks_exact_mut(s)
    }

    /// 遍历全部**物理**单元下标。
    #[inline]
    pub fn interior(&self) -> impl Iterator<Item = (isize, isize)> + '_ {
        let (ni, nj) = (self.ni as isize, self.nj as isize);
        (0..ni).flat_map(move |i| (0..nj).map(move |j| (i, j)))
    }

    /// 遍历全部可寻址下标(物理单元 + 虚拟层,含角落)。
    #[inline]
    pub fn all_indices(&self) -> impl Iterator<Item = (isize, isize)> + '_ {
        let h = self.halo as isize;
        let (ni, nj) = (self.ni as isize, self.nj as isize);
        (-h..ni + h).flat_map(move |i| (-h..nj + h).map(move |j| (i, j)))
    }
}

/// 一行的可变视图,支持有符号的 j 下标(与 [`Field`] 保持一致的写法)。
pub struct RowMut<'a, T> {
    data: &'a mut [T],
    halo: isize,
}

impl<T> Index<isize> for RowMut<'_, T> {
    type Output = T;
    #[inline(always)]
    fn index(&self, j: isize) -> &T {
        &self.data[(j + self.halo) as usize]
    }
}

impl<T> IndexMut<isize> for RowMut<'_, T> {
    #[inline(always)]
    fn index_mut(&mut self, j: isize) -> &mut T {
        &mut self.data[(j + self.halo) as usize]
    }
}

/// 单个并行任务至少要处理这么多单元。
///
/// 行是天然的并行单位,但一行只有 `NJ` 个单元 —— 网格较窄时,单行的计算量还
/// 抵不上一次任务派发的开销。实测在 128x256 的网格上不设下限时 24 线程比串行
/// **慢 3.5 倍**;按这个粒度合并行之后才转为正向收益。
const MIN_CELLS_PER_TASK: usize = 8192;

impl<T: Send> Field<T> {
    /// 按**物理行**并行迭代,产出 `(i, 该行的可变视图)`。
    ///
    /// 行与行之间不重叠,所以可以安全并行;kernel 只需保证"每个输出元素只被
    /// 写一次",输入则来自其他 `Field`(不同对象,借用检查天然不冲突)。
    /// 因为每个输出元素的值只由输入决定,结果与线程数无关 —— 并行不影响可复现性。
    ///
    /// 粒度由 [`MIN_CELLS_PER_TASK`] 控制,避免小网格上被派发开销吃掉。
    pub fn par_interior_rows_mut(
        &mut self,
    ) -> impl IndexedParallelIterator<Item = (isize, RowMut<'_, T>)> {
        let (halo, stride, ni, nj) = (self.halo, self.stride, self.ni, self.nj);
        let min_rows = (MIN_CELLS_PER_TASK / nj.max(1)).max(1);
        self.data
            .par_chunks_exact_mut(stride)
            .skip(halo)
            .take(ni)
            .enumerate()
            .with_min_len(min_rows)
            .map(move |(r, row)| {
                (
                    r as isize,
                    RowMut {
                        data: row,
                        halo: halo as isize,
                    },
                )
            })
    }

    /// 串行版本,便于在小规模或调试时避免线程开销。
    pub fn interior_rows_mut(&mut self) -> impl Iterator<Item = (isize, RowMut<'_, T>)> {
        let (halo, stride, ni) = (self.halo, self.stride, self.ni);
        self.data
            .chunks_exact_mut(stride)
            .skip(halo)
            .take(ni)
            .enumerate()
            .map(move |(r, row)| {
                (
                    r as isize,
                    RowMut {
                        data: row,
                        halo: halo as isize,
                    },
                )
            })
    }
}

impl<T: Copy> Field<T> {
    #[inline(always)]
    pub fn get(&self, i: isize, j: isize) -> T {
        self.data[self.offset(i, j)]
    }

    /// 把物理区按行主序拷成扁平 `Vec`(golden 比对用的顺序)。
    pub fn to_interior_vec(&self) -> Vec<T> {
        self.interior().map(|(i, j)| self.get(i, j)).collect()
    }
}

impl<T> Index<(isize, isize)> for Field<T> {
    type Output = T;
    #[inline(always)]
    fn index(&self, (i, j): (isize, isize)) -> &T {
        self.at(i, j)
    }
}

impl<T> IndexMut<(isize, isize)> for Field<T> {
    #[inline(always)]
    fn index_mut(&mut self, (i, j): (isize, isize)) -> &mut T {
        self.at_mut(i, j)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halo_indices_are_addressable() {
        let mut f: Field<f64> = Field::new(4, 5, 2);
        f.set(-2, -2, 1.0);
        f.set(5, 6, 2.0);
        assert_eq!(f.get(-2, -2), 1.0);
        assert_eq!(f.get(5, 6), 2.0);
    }

    #[test]
    fn distinct_indices_map_to_distinct_slots() {
        let f: Field<f64> = Field::new(3, 4, 1);
        let mut seen = std::collections::HashSet::new();
        for i in -1..4 {
            for j in -1..5 {
                assert!(seen.insert(f.offset(i, j)), "offset collision at ({i},{j})");
            }
        }
        assert_eq!(seen.len(), 5 * 6);
    }

    #[test]
    fn interior_iterates_row_major() {
        let f: Field<f64> = Field::new(2, 3, 1);
        let got: Vec<_> = f.interior().collect();
        assert_eq!(got, vec![(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]);
    }

    #[test]
    fn vec5_arithmetic_is_componentwise() {
        let a = Vec5::new(1.0, 2.0, 3.0, 4.0, 5.0);
        let b = Vec5::new(0.5, 0.5, 0.5, 0.5, 0.5);
        assert_eq!((a + b).0, [1.5, 2.5, 3.5, 4.5, 5.5]);
        assert_eq!((a - b).0, [0.5, 1.5, 2.5, 3.5, 4.5]);
        assert_eq!((a * 2.0).0, [2.0, 4.0, 6.0, 8.0, 10.0]);
        assert_eq!((2.0 * a).0, (a * 2.0).0);
        assert_eq!((-a).0, [-1.0, -2.0, -3.0, -4.0, -5.0]);
    }

    #[test]
    #[should_panic(expected = "out of halo range")]
    #[cfg(debug_assertions)]
    fn out_of_halo_panics() {
        let f: Field<f64> = Field::new(4, 4, 1);
        f.get(-2, 0);
    }
}
