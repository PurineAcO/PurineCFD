//! 甯?halo(铏氭嫙灞?鐨勪簩缁存暟缁?浠ュ強浜斿垎閲忓畧鎭掑悜閲?[`Vec5`].
//!
//! # 涓轰粈涔堟槸 halo
//!
//! Python 鍩虹嚎鎶婅櫄鎷熷崟鍏?*杩藉姞**鍦ㄧ墿鐞嗘暟缁勪箣鍚?澹侀潰 ghost 鏀惧湪 `CellList[i_total..]`,
//! 杩滃満 ghost 鏀惧湪鏇村悗闈?鍛ㄥ悜 ghost 杩藉姞鍦ㄦ瘡琛屾湯灏俱€備簬鏄瘡涓?kernel 閮藉緱鑷繁鍐欎竴閬?//! "濡傛灉 i==1 鍙?`CellList[i_total]`銆佸鏋?j==j_total 鍙?`CellList[j_total+IM+1]`鈥︹€?
//! 杩欑被鏄犲皠 鈥斺€?鍏ㄩ」鐩噸澶嶄簡鍗佸嚑娆?鑰?`BUGS.md` 閲?B4/B5/B6/B8 鍥涗釜鏁板€奸敊璇?//! **鍏ㄩ儴**鍑鸿嚜杩欎簺鎵嬪啓鏄犲皠鐨勭瑪璇€?//!
//! 杩欓噷鏀规垚:鍗曞厓鐨勪笅鏍囩┖闂寸洿鎺ユ墿灞曞埌 `[-H, N+H)`,铏氭嫙灞傚氨浣忓湪璐熶笅鏍囧拰瓒婄晫涓嬫爣涓娿€?//! 杈圭晫鏉′欢鏀舵暃鎴愬敮涓€涓€澶?[`crate::boundary::apply`],姝ゅ悗姣忎釜 kernel 閮芥槸涓嶅甫浠讳綍
//! 鐗瑰垽鐨勭煩褰㈠惊鐜€傜储寮曞啓閿欑殑鏁寸被 bug 鍦ㄧ粨鏋勪笂琚秷鎺変簡銆?//!
//! ```text
//!      j = -3 -2 -1 | 0  1  ...  NJ-1 | NJ NJ+1 NJ+2
//! i = -3   鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹尖攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹尖攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?//!  ...     鈹? halo  鈹?               鈹?   halo    鈹?//! i = -1   鈹?       鈹?               鈹?           鈹?//!          鈹溾攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹尖攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹尖攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?//! i =  0   鈹? halo  鈹?   鐗╃悊鍗曞厓     鈹?   halo    鈹?//!  ...     鈹?       鈹?  NI x NJ      鈹?           鈹?//! i = NI-1 鈹?       鈹?               鈹?           鈹?//!          鈹溾攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹尖攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹尖攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?//! i = NI   鈹? halo  鈹?     halo      鈹?   halo    鈹?//! ```

use std::ops::{Add, AddAssign, Index, IndexMut, Mul, Neg, Sub};

use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::ParallelSliceMut;

/// 浜斿垎閲忓畧鎭掑悜閲?`[蟻, 蟻u, 蟻v, 蟻E, 蟻谓虄]`銆?///
/// 瀹氫箟浜嗗畬鏁寸殑绠楁湳杩愮畻绗?濂借鏍煎紡鍏紡鍦ㄤ唬鐮侀噷淇濇寔鏁板鍐欐硶 鈥斺€?/// 渚嬪 JST 鑰楁暎椤瑰彲浠ョ洿鎺ュ啓鎴?`lam * (d1u * eps2 - d3u * eps4)`銆?#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct Vec5(pub [f64; 5]);

/// 瀹堟亽鍚戦噺鐨勫垎閲忎笅鏍囥€?pub mod comp {
    /// 瀵嗗害 蟻
    pub const RHO: usize = 0;
    /// x 鏂瑰悜鍔ㄩ噺 蟻u
    pub const MX: usize = 1;
    /// y 鏂瑰悜鍔ㄩ噺 蟻v
    pub const MY: usize = 2;
    /// 鎬昏兘 蟻E
    pub const RHO_E: usize = 3;
    /// 婀嶆祦宸ヤ綔鍙橀噺 蟻谓虄
    pub const RHO_NU: usize = 4;
}

impl Vec5 {
    pub const ZERO: Self = Vec5([0.0; 5]);

    #[inline]
    pub const fn new(rho: f64, mx: f64, my: f64, rho_e: f64, rho_nu: f64) -> Self {
        Vec5([rho, mx, my, rho_e, rho_nu])
    }

    /// 鍚勫垎閲忕粷瀵瑰€肩殑鏈€澶у€?鐢ㄤ簬鏀舵暃/瀹瑰樊鍒ゆ柇銆?    #[inline]
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

// 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

/// 甯?halo 鐨勪簩缁存暟缁?涓嬫爣涓烘湁绗﹀彿鏁存暟,鍚堟硶鑼冨洿 `[-halo, n+halo)`銆?///
/// 鍐呴儴鏄涓诲簭鐨勪竴缁?`Vec`(i 涓烘參缁淬€乯 涓哄揩缁?,鍥犳娌?j 閬嶅巻鏄繛缁闂€?#[derive(Clone, Debug)]
pub struct Field<T> {
    data: Vec<T>,
    ni: usize,
    nj: usize,
    halo: usize,
    stride: usize,
}

impl<T: Clone + Default> Field<T> {
    /// 寤虹珛 `ni x nj` 鐨勭墿鐞嗗尯,鍥涘懆鍚勭暀 `halo` 灞傘€?    pub fn new(ni: usize, nj: usize, halo: usize) -> Self {
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

    /// 鏈夌鍙蜂笅鏍?鈫?绾挎€т笅鏍囥€傝秺鐣屾椂 panic(debug 涓?release 閮芥鏌?
    /// 杩欓噷鐨勪笅鏍囩畻鏈鏄渶瀹规槗鍐欓敊鐨勫湴鏂?涓嶅€煎緱涓虹渷涓€娆℃瘮杈冨啋椋庨櫓)銆?    #[inline(always)]
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

    /// 绗?`i` 琛?鍚?halo 鍒?鐨勫彲鍙樺垏鐗囥€傝涔嬮棿浜掍笉閲嶅彔,鏄?rayon 骞惰鐨勫ぉ鐒跺崟浣嶃€?    #[inline]
    pub fn row_mut(&mut self, i: isize) -> &mut [T] {
        let start = self.offset(i, -(self.halo as isize));
        &mut self.data[start..start + self.stride]
    }

    /// 搴曞眰杩炵画瀛樺偍,鍚?halo銆?    #[inline]
    pub fn raw(&self) -> &[T] {
        &self.data
    }

    #[inline]
    pub fn raw_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// 琛屽唴 j 浠?`-halo` 璧风畻鐨勫亸绉婚噺,閰嶅悎 [`Field::rows_mut`] 浣跨敤銆?    #[inline]
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// 鎸夎鍒囧垎鎴愬彲骞惰鐨勫彲鍙樺垏鐗?鍚?halo 琛?銆?    #[inline]
    pub fn rows_mut(&mut self) -> std::slice::ChunksExactMut<'_, T> {
        let s = self.stride;
        self.data.chunks_exact_mut(s)
    }

    /// 閬嶅巻鍏ㄩ儴**鐗╃悊**鍗曞厓涓嬫爣銆?    #[inline]
    pub fn interior(&self) -> impl Iterator<Item = (isize, isize)> + '_ {
        let (ni, nj) = (self.ni as isize, self.nj as isize);
        (0..ni).flat_map(move |i| (0..nj).map(move |j| (i, j)))
    }

    /// 閬嶅巻鍏ㄩ儴鍙鍧€涓嬫爣(鐗╃悊鍗曞厓 + 铏氭嫙灞?鍚钀?銆?    #[inline]
    pub fn all_indices(&self) -> impl Iterator<Item = (isize, isize)> + '_ {
        let h = self.halo as isize;
        let (ni, nj) = (self.ni as isize, self.nj as isize);
        (-h..ni + h).flat_map(move |i| (-h..nj + h).map(move |j| (i, j)))
    }
}

/// 涓€琛岀殑鍙彉瑙嗗浘,鏀寔鏈夌鍙风殑 j 涓嬫爣(涓?[`Field`] 淇濇寔涓€鑷寸殑鍐欐硶)銆?pub struct RowMut<'a, T> {
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

/// 鍗曚釜骞惰浠诲姟鑷冲皯瑕佸鐞嗚繖涔堝鍗曞厓銆?///
/// 琛屾槸澶╃劧鐨勫苟琛屽崟浣?浣嗕竴琛屽彧鏈?`NJ` 涓崟鍏?鈥斺€?缃戞牸杈冪獎鏃?鍗曡鐨勮绠楅噺杩?/// 鎶典笉涓婁竴娆′换鍔℃淳鍙戠殑寮€閿€銆傚疄娴嬪湪 128x256 鐨勭綉鏍间笂涓嶈涓嬮檺鏃?24 绾跨▼姣斾覆琛?/// **鎱?3.5 鍊?*;鎸夎繖涓矑搴﹀悎骞惰涔嬪悗鎵嶈浆涓烘鍚戞敹鐩娿€?const MIN_CELLS_PER_TASK: usize = 8192;

impl<T: Send> Field<T> {
    /// 鎸?*鐗╃悊琛?*骞惰杩唬,浜у嚭 `(i, 璇ヨ鐨勫彲鍙樿鍥?`銆?    ///
    /// 琛屼笌琛屼箣闂翠笉閲嶅彔,鎵€浠ュ彲浠ュ畨鍏ㄥ苟琛?kernel 鍙渶淇濊瘉"姣忎釜杈撳嚭鍏冪礌鍙
    /// 鍐欎竴娆?,杈撳叆鍒欐潵鑷叾浠?`Field`(涓嶅悓瀵硅薄,鍊熺敤妫€鏌ュぉ鐒朵笉鍐茬獊)銆?    /// 鍥犱负姣忎釜杈撳嚭鍏冪礌鐨勫€煎彧鐢辫緭鍏ュ喅瀹?缁撴灉涓庣嚎绋嬫暟鏃犲叧 鈥斺€?骞惰涓嶅奖鍝嶅彲澶嶇幇鎬с€?    ///
    /// 绮掑害鐢?[`MIN_CELLS_PER_TASK`] 鎺у埗,閬垮厤灏忕綉鏍间笂琚淳鍙戝紑閿€鍚冩帀銆?    pub fn par_interior_rows_mut(
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

    /// 涓茶鐗堟湰,渚夸簬鍦ㄥ皬瑙勬ā鎴栬皟璇曟椂閬垮厤绾跨▼寮€閿€銆?    pub fn interior_rows_mut(&mut self) -> impl Iterator<Item = (isize, RowMut<'_, T>)> {
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

    /// 鎶婄墿鐞嗗尯鎸夎涓诲簭鎷锋垚鎵佸钩 `Vec`(golden 姣斿鐢ㄧ殑椤哄簭)銆?    pub fn to_interior_vec(&self) -> Vec<T> {
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
