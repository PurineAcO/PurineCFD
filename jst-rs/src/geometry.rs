//! 鏈夐檺浣撶Н娉曠殑鍑犱綍搴﹂噺:鍗曞厓闈㈢Н/褰㈠績銆侀潰娉曞悜/涓偣銆佸闈㈣窛绂汇€?//!
//! # 涓嬫爣绾﹀畾
//!
//! 璁?`NI = n_rings-1` 涓哄緞鍚戝崟鍏冩暟銆乣NJ = n_theta` 涓哄懆鍚戝崟鍏冩暟銆?//!
//! * 鍗曞厓 `(i, j)`,`i 鈭?[0,NI)`,`j 鈭?[0,NJ)`;鍥涗釜椤剁偣鏄妭鐐?//!   `(i,j) (i+1,j) (i+1,j+1) (i,j+1)`銆?//! * **tau 闈?*(鍛ㄥ悜杈?涓€鍦堝湀鐨?娉㈢汗"):`i 鈭?[0,NI]`,`j 鈭?[0,NJ)`銆?//!   tau 闈?`i` 鍒嗛殧鍗曞厓 `i-1` 涓?`i`,娉曞悜鎸囧悜寰勫悜澶栦晶銆?//! * **n 闈?*(寰勫悜杈?"娉㈢汗鍦堢殑鐩村緞"):`i 鈭?[0,NI)`,`j 鈭?[0,NJ)`銆?//!   n 闈?`j` 鍒嗛殧鍗曞厓 `j-1` 涓?`j`,娉曞悜鎸囧悜鍛ㄥ悜 +j 渚с€?//!
//! 娉曞悜鐨?*妯￠暱绛変簬杈归暱**(鍗抽潰绉姞鏉冩硶鍚?,鍥犳閫氶噺鍙互鐩存帴鐐逛箻娉曞悜鑰屼笉蹇?//! 鍙﹀涔橀潰绉€?//!
//! 杩欏绾﹀畾淇濊瘉浜嗗害閲忛棴鍚?`n_蟿(i+1) 鈭?n_蟿(i) + n_n(j+1) 鈭?n_n(j) 鈮?0`,
//! 瀹冩鏄嚜鐢辨潵娴佷繚鎸佹€х殑鍏呰鏉′欢(瑙?`tests/properties.rs`)銆?
use crate::field::Field;
use crate::mesh::Mesh;

/// 鍗曢潰鐨勫嚑浣曢噺銆?#[derive(Clone, Copy, Debug, Default)]
pub struct FaceGeom {
    /// 闈㈢Н鍔犳潈娉曞悜鐨?x 鍒嗛噺(妯￠暱 = 杈归暱)
    pub nx: f64,
    /// 闈㈢Н鍔犳潈娉曞悜鐨?y 鍒嗛噺
    pub ny: f64,
    /// 闈腑鐐?x
    pub mx: f64,
    /// 闈腑鐐?y
    pub my: f64,
}

impl FaceGeom {
    #[inline]
    pub fn length(&self) -> f64 {
        self.nx.hypot(self.ny)
    }
}

/// 鍏ㄩ儴涓庢椂闂存棤鍏崇殑鍑犱綍閲?setup 闃舵绠椾竴娆″悗鍙銆?#[derive(Clone, Debug)]
pub struct Geometry {
    pub ni: usize,
    pub nj: usize,
    /// 鍗曞厓闈㈢Н(浜岀淮涓嬬殑"浣撶Н")
    pub vol: Field<f64>,
    /// 鍗曞厓褰㈠績 x / y
    pub cx: Field<f64>,
    pub cy: Field<f64>,
    /// 鍗曞厓涓績鍒板闈㈢殑鏈€杩戣窛绂?S-A 妯″瀷鐨?d)
    pub wall_dist: Field<f64>,
    /// `1/V`銆傛搴︺€佹畫宸洿鏂伴兘瑕佹寜浣撶Н褰掍竴,棰勫厛绠楀ソ鐪佹帀閫愬崟鍏冮€愮骇鐨勯櫎娉?    pub inv_vol: Field<f64>,
    /// `1/d虏`銆係-A 婧愰」閲?`谓虄/(魏虏d虏)` 涓?`(谓虄/d)虏` 閮借鐢?    pub inv_wall_dist_sq: Field<f64>,
    /// 鍛ㄥ悜闈?`(NI+1) x NJ`
    pub tau: Field<FaceGeom>,
    /// 寰勫悜闈?`NI x NJ`
    pub nrm: Field<FaceGeom>,
}

impl Geometry {
    /// 鍛ㄥ悜涓嬫爣 +1 鐨勫洖缁曘€?    #[inline(always)]
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
            // 鍗曞厓閲忛渶瑕?halo:铏氭嫙鍗曞厓涔熷弬涓?JST 妯℃澘,vol 鐢ㄤ簬婧愰」缂╂斁
            vol: Field::new(ni, nj, halo),
            cx: Field::new(ni, nj, halo),
            cy: Field::new(ni, nj, halo),
            wall_dist: Field::new(ni, nj, halo),
            inv_vol: Field::new(ni, nj, halo),
            inv_wall_dist_sq: Field::new(ni, nj, halo),
            // 闈㈤噺涓嶉渶瑕?halo:鎵€鏈?kernel 鍙湪 [0,NI]x[0,NJ) 涓婅闂潰
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

    /// 鍗曞厓闈㈢Н涓庡舰蹇冦€?    ///
    /// 闈㈢Н鐢?瀵硅绾垮弶绉?鍏紡 `A = 陆|AC 脳 DB|` 鈥斺€?瀵逛换鎰忎笉鑷氦鐨勫洓杈瑰舰閮界簿纭€?    /// 褰㈠績鐢ㄥ杈瑰舰褰㈠績鐨勬爣鍑嗗叕寮?shoelace 鍔犳潈)銆?    fn build_cells(&mut self, mesh: &Mesh) {
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

    /// 闈㈡硶鍚戜笌涓偣銆?    fn build_faces(&mut self, mesh: &Mesh) {
        // tau 闈?娌垮懆鍚戠殑杈?(i,j) 鈫?(i,j+1),娉曞悜 (dy, 鈭抎x) 鎸囧悜寰勫悜澶栦晶
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
        // n 闈?娌垮緞鍚戠殑杈?(i,j) 鈫?(i+1,j),娉曞悜 (鈭抎y, dx) 鎸囧悜鍛ㄥ悜 +j 渚?        for i in 0..self.ni {
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

    /// 姣忎釜鍗曞厓涓績鍒板闈㈢殑鏈€杩戣窛绂汇€?    ///
    /// 澹侀潰鍗崇 0 灞?tau 闈€傚彧鍦ㄥ懆鍚?卤`window` 鐨勮寖鍥村唴鎼滅储:O 鍨嬬綉鏍间笂鏈€杩戠殑
    /// 澹侀潰鐐瑰嚑涔庢€诲湪寰勫悜姝ｄ笅鏂?`window` 鍙栧埌绾?卤20% 鍛ㄩ暱宸茶繙瓒呴渶瑕?鍙妸
    /// 澶嶆潅搴︿粠 O(NI路NJ虏) 闄嶅埌 O(NI路NJ路window)銆?    fn build_wall_distance(&mut self) {
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

    /// 鍏ㄩ儴鍗曞厓闈㈢Н涔嬪拰銆?    pub fn total_area(&self) -> f64 {
        self.vol.interior().map(|(i, j)| self.vol.get(i, j)).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fangdata() -> Geometry {
        let m = Mesh::parse(include_str!("../fangdata.txt")).unwrap();
        Geometry::build(&m, 3)
    }

    /// 搴﹂噺闂悎 鈥斺€?鑷敱鏉ユ祦淇濇寔鎬х殑鍏呰鍑犱綍鏉′欢銆?    #[test]
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
        let m = Mesh::parse(include_str!("../fangdata.txt")).unwrap();
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
        let m = Mesh::parse(include_str!("../fangdata.txt")).unwrap();
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
        let m = Mesh::parse(include_str!("../fangdata.txt")).unwrap();
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

