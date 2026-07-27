//! O 鍨嬬綉鏍肩殑璇诲彇銆?//!
//! 鏂囦欢鏍煎紡涓?Python 鍩虹嚎鐨?`meshreading.read_mesh` 瀹屽叏涓€鑷?:
//!
//! ```text
//! i_total j_total
//! x y          鈫?閫愮幆銆佺幆鍐呴€嗘椂閽?鍏?i_total*j_total 琛?//! ```
//!
//! 鑻ユ瘡鐜灏捐妭鐐归噸鍚?灏佸彛缃戞牸),鑷姩鍓婃帀閲嶅鐨勬湯鐐瑰苟鎶?`j_total` 鍑?1銆?
use std::fmt;
use std::path::Path;

/// 缃戞牸璇诲彇閿欒銆侾ython 鍩虹嚎鍦ㄨ繖浜涙儏褰㈠彧 `print` 涓€鍙ュ氨 `return`,璁╁悗缁唬鐮?/// 鍦ㄥ崐鍒濆鍖栫殑鍏ㄥ眬鐘舵€佷笂缁х画璺?杩欓噷鏀规垚鏄惧紡閿欒绫诲瀷,璋冪敤鏂瑰繀椤诲鐞嗐€?#[derive(Debug)]
pub enum MeshError {
    Io(std::io::Error),
    BadHeader(String),
    Empty,
    CountMismatch { expected: usize, got: usize },
    BadPoint { line: usize, text: String },
    TooSmall { ni: usize, nj: usize },
}

impl fmt::Display for MeshError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::BadHeader(h) => write!(f, "bad header {h:?}: expected `i_total j_total`"),
            Self::Empty => write!(f, "mesh file contains no points"),
            Self::CountMismatch { expected, got } => {
                write!(f, "expected {expected} points from header, found {got}")
            }
            Self::BadPoint { line, text } => write!(f, "line {line}: cannot parse point {text:?}"),
            Self::TooSmall { ni, nj } => write!(
                f,
                "mesh {ni}x{nj} too small: the JST 4th-order stencil needs \
                 at least 4 rings and 8 circumferential points"
            ),
        }
    }
}

impl std::error::Error for MeshError {}

impl From<std::io::Error> for MeshError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// 鑺傜偣鍧愭爣銆?#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

/// O 鍨嬬綉鏍肩殑鑺傜偣闃靛垪銆?///
/// 鑺傜偣鎸?`n_rings x n_theta` 鎺掑垪,鐜唴**涓嶅皝鍙?*(鏈偣涓庨鐐逛笉閲嶅悎),鍛ㄥ悜鐢?/// 绱㈠紩鍙栨ā瀹炵幇鍥炵粫銆傚崟鍏冩暟涓?`(n_rings-1) x n_theta`銆?#[derive(Clone, Debug)]
pub struct Mesh {
    n_rings: usize,
    n_theta: usize,
    nodes: Vec<Point>,
}

impl Mesh {
    /// 鐜暟(寰勫悜鑺傜偣灞傛暟)銆?    #[inline]
    pub fn n_rings(&self) -> usize {
        self.n_rings
    }
    /// 鍛ㄥ悜鑺傜偣鏁般€?    #[inline]
    pub fn n_theta(&self) -> usize {
        self.n_theta
    }
    /// 寰勫悜鍗曞厓鏁?`NI = n_rings - 1`銆?    #[inline]
    pub fn ni(&self) -> usize {
        self.n_rings - 1
    }
    /// 鍛ㄥ悜鍗曞厓鏁?`NJ = n_theta`銆?    #[inline]
    pub fn nj(&self) -> usize {
        self.n_theta
    }
    #[inline]
    pub fn n_cells(&self) -> usize {
        self.ni() * self.nj()
    }

    /// 鍙栬妭鐐?鍛ㄥ悜鑷姩鍙栨ā鍥炵粫銆?    #[inline]
    pub fn node(&self, ring: usize, theta: usize) -> Point {
        debug_assert!(ring < self.n_rings);
        self.nodes[ring * self.n_theta + theta % self.n_theta]
    }

    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, MeshError> {
        Self::parse(&std::fs::read_to_string(path.as_ref())?)
    }

    pub fn parse(text: &str) -> Result<Self, MeshError> {
        let mut lines = text.lines();
        let header = lines.next().ok_or(MeshError::Empty)?.trim();
        let mut it = header.split_whitespace();
        let (ni, nj) = match (
            it.next().and_then(|s| s.parse::<usize>().ok()),
            it.next().and_then(|s| s.parse::<usize>().ok()),
        ) {
            (Some(a), Some(b)) => (a, b),
            _ => return Err(MeshError::BadHeader(header.to_string())),
        };

        let mut nodes = Vec::with_capacity(ni * nj);
        for (k, line) in lines.enumerate() {
            let t = line.trim();
            if t.is_empty() {
                continue;
            }
            let mut p = t.split_whitespace();
            match (
                p.next().and_then(|s| s.parse::<f64>().ok()),
                p.next().and_then(|s| s.parse::<f64>().ok()),
            ) {
                (Some(x), Some(y)) => nodes.push(Point { x, y }),
                _ => {
                    return Err(MeshError::BadPoint {
                        line: k + 2,
                        text: t.to_string(),
                    })
                }
            }
        }

        if nodes.is_empty() {
            return Err(MeshError::Empty);
        }
        if nodes.len() != ni * nj {
            return Err(MeshError::CountMismatch {
                expected: ni * nj,
                got: nodes.len(),
            });
        }

        let mut mesh = Self {
            n_rings: ni,
            n_theta: nj,
            nodes,
        };
        mesh.trim_closing_duplicate();

        if mesh.n_rings < 4 || mesh.n_theta < 8 {
            return Err(MeshError::TooSmall {
                ni: mesh.n_rings,
                nj: mesh.n_theta,
            });
        }
        Ok(mesh)
    }

    /// 鑻ユ瘡鐜灏捐妭鐐归噸鍚?鍒犳帀閲嶅鐨勬湯鐐广€?    fn trim_closing_duplicate(&mut self) {
        if self.n_theta < 2 {
            return;
        }
        let closed = (0..self.n_rings).all(|r| {
            let a = self.nodes[r * self.n_theta];
            let b = self.nodes[r * self.n_theta + self.n_theta - 1];
            (a.x - b.x).abs() <= 1e-10 && (a.y - b.y).abs() <= 1e-10
        });
        if !closed {
            return;
        }
        let new_nj = self.n_theta - 1;
        let mut trimmed = Vec::with_capacity(self.n_rings * new_nj);
        for r in 0..self.n_rings {
            let base = r * self.n_theta;
            trimmed.extend_from_slice(&self.nodes[base..base + new_nj]);
        }
        self.nodes = trimmed;
        self.n_theta = new_nj;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring_mesh(rings: usize, theta: usize, closed: bool) -> String {
        let per = if closed { theta + 1 } else { theta };
        let mut s = format!("{rings} {per}\n");
        for r in 0..rings {
            let rad = 1.0 + r as f64;
            for t in 0..per {
                let a = 2.0 * std::f64::consts::PI * (t % theta) as f64 / theta as f64;
                s += &format!("{:.10} {:.10}\n", rad * a.cos(), rad * a.sin());
            }
        }
        s
    }

    #[test]
    fn parses_open_mesh() {
        let m = Mesh::parse(&ring_mesh(5, 16, false)).unwrap();
        assert_eq!((m.n_rings(), m.n_theta()), (5, 16));
        assert_eq!(m.n_cells(), 4 * 16);
    }

    #[test]
    fn trims_closed_mesh() {
        let m = Mesh::parse(&ring_mesh(5, 16, true)).unwrap();
        assert_eq!(m.n_theta(), 16);
        assert_eq!(m.n_cells(), 4 * 16);
    }

    #[test]
    fn node_wraps_in_theta() {
        let m = Mesh::parse(&ring_mesh(5, 16, false)).unwrap();
        assert_eq!(m.node(0, 16), m.node(0, 0));
        assert_eq!(m.node(2, 17), m.node(2, 1));
    }

    #[test]
    fn rejects_bad_header() {
        assert!(matches!(
            Mesh::parse("not a header\n1.0 2.0\n"),
            Err(MeshError::BadHeader(_))
        ));
    }

    #[test]
    fn rejects_count_mismatch() {
        assert!(matches!(
            Mesh::parse("4 8\n1.0 2.0\n"),
            Err(MeshError::CountMismatch { .. })
        ));
    }

    #[test]
    fn rejects_too_small() {
        assert!(matches!(
            Mesh::parse(&ring_mesh(3, 16, false)),
            Err(MeshError::TooSmall { .. })
        ));
    }

    #[test]
    fn reads_repository_mesh() {
        let m = Mesh::parse(include_str!("../fangdata.txt")).unwrap();
        assert_eq!((m.n_rings(), m.n_theta()), (10, 12));
        assert_eq!(m.n_cells(), 108);
        assert!((m.node(0, 0).x - 1.0).abs() < 1e-12);
    }
}

