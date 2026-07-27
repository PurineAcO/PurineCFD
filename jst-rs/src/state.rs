//! 娴佸満鐘舵€佺殑瀛樺偍甯冨眬銆?//!
//! # 鍒嗙粍 SoA
//!
//! 涓嶆槸"姣忎釜鏍囬噺涓€涓暟缁?(绾?SoA),涔熶笉鏄?Python 閭ｆ牱"姣忎釜鍗曞厓涓€涓璞?(AoS),
//! 鑰屾槸**鎸変娇鐢ㄦ柟寮忓垎缁?*:鎬绘槸琚竴璧疯鍐欑殑閲忔墦鍖呮垚涓€涓皬缁撴瀯浣?鍚勭粍涔嬮棿浠嶇劧
//! 鏄嫭绔嬬殑杩炵画鏁扮粍銆?//!
//! * [`Grad`] 鈥斺€?鍏釜姊害鍒嗛噺銆傜矘鎬ч」涓庢簮椤逛竴娆¤鐢ㄥ叏閮ㄥ叓涓?鎵撳寘鍚?//!   `gradient::compute` 鍙啓涓€涓暟缁?涓嶅繀 `zip` 鍏釜骞惰杩唬鍣?//!   (rayon 鐨勫璺?`zip` 瑕佹眰鍚勮矾鍚屾鍒囧垎,寮€閿€杩滈珮浜庢敹鐩?銆?//! * [`TurbAux`] 鈥斺€?`渭, 蠂, fv1`,S-A 鐨勪笁涓腑闂撮噺,鎬绘槸鍚屾椂浜х敓銆佸悓鏃舵秷璐广€?//! * [`DiffTensor`] 鈥斺€?鎵╂暎寮犻噺鐨勪袱鍒?闈笂鍋氬钩鍧囨椂鎴愬浣跨敤銆?//! * [`Vec5`] 鈥斺€?浜斾釜瀹堟亽鍒嗛噺,鎵€鏈夐€氶噺杩愮畻閮戒綔鐢ㄥ湪鏁寸粍涓娿€?//!
//! 杩欐牱姣忎釜 kernel 閮芥槸"璇昏嫢骞叉暟缁勩€佸啓**涓€涓?*鏁扮粍",鍊熺敤妫€鏌ョ洿鎺ヨ瘉鏄庢棤鍒悕,
//! rayon 骞惰鏃笉闇€瑕?`unsafe` 涔熶笉闇€瑕佸厠闅嗕腑闂寸粨鏋溿€?
use crate::config::Config;
use crate::field::{Field, Vec5};
use crate::geometry::Geometry;

pub type F64Field = Field<f64>;
pub type Vec5Field = Field<Vec5>;

/// 鍗曞厓涓婄殑 Green-Gauss 姊害銆?#[derive(Clone, Copy, Debug, Default)]
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

/// S-A 妯″瀷鐨勯€愬崟鍏冧腑闂撮噺銆?#[derive(Clone, Copy, Debug, Default)]
pub struct TurbAux {
    /// 鍒嗗瓙绮樺害 渭(Sutherland)
    pub mu: f64,
    /// 蠂 = 蟻谓虄/渭
    pub chi: f64,
    /// 闃诲凹鍑芥暟 fv1
    pub fv1: f64,
}

/// 绮樻€?婀嶆祦鎵╂暎寮犻噺鐨勪袱鍒椼€?#[derive(Clone, Copy, Debug, Default)]
pub struct DiffTensor {
    pub x: Vec5,
    pub y: Vec5,
}

/// 闈笂鐨勮嚜閫傚簲鑰楁暎绯绘暟銆?#[derive(Clone, Copy, Debug, Default)]
pub struct Eps {
    /// 浜岄樁(婵€娉?绯绘暟 蔚虏
    pub e2: f64,
    /// 鍥涢樁(鑳屾櫙)绯绘暟 蔚鈦?    pub e4: f64,
}

/// 鍗曞厓涓績閲忋€?#[derive(Clone, Debug)]
pub struct Cells {
    pub ni: usize,
    pub nj: usize,

    // 鈹€鈹€ 瀹堟亽閲?鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    /// `[蟻, 蟻u, 蟻v, 蟻E, 蟻谓虄]`
    pub u: Vec5Field,
    /// 鏃堕棿姝ュ紑濮嬫椂鐨勫畧鎭掗噺(RK 鍚勭骇閮藉熀浜庡畠鏇存柊)
    pub u_former: Vec5Field,

    // 鈹€鈹€ 鍘熷鍙橀噺 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    pub rho: F64Field,
    pub p: F64Field,
    pub t: F64Field,
    pub vx: F64Field,
    pub vy: F64Field,
    /// 鍗曚綅璐ㄩ噺鎬昏兘
    pub e: F64Field,
    /// 鍗曚綅璐ㄩ噺鎬荤創
    pub h: F64Field,
    /// 澹伴€?    pub c: F64Field,
    /// 婀嶆祦宸ヤ綔鍙橀噺 谓虄
    pub nut: F64Field,

    // 鈹€鈹€ 瀵煎嚭閲?鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    pub grad: Field<Grad>,
    pub aux: Field<TurbAux>,
    pub diff: Field<DiffTensor>,

    // 鈹€鈹€ 娈嬪樊鍒嗛」 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    /// 瀵规祦閫氶噺
    pub fc: Vec5Field,
    /// 绮樻€?婀嶆祦鎵╂暎閫氶噺
    pub fv: Vec5Field,
    /// JST 浜哄伐绮樻€?    pub fd: Vec5Field,
    /// S-A 婧愰」(鍙湁绗?5 鍒嗛噺闈為浂,鏁呭瓨鏍囬噺)
    pub src: F64Field,

    /// 褰撳湴鏃堕棿姝?閫愬崟鍏?JST 璋卞崐寰勯渶瑕?
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

    /// 鐢卞師濮嬪彉閲忚閰嶅畧鎭掗噺 `U`銆?    #[inline]
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

    /// 閫愬崟鍏冨啓鍏ヤ竴缁勪竴鑷寸殑鍘熷鍙橀噺骞惰閰嶅畧鎭掗噺銆?    #[inline]
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

    /// 鐢ㄦ潵娴佹潯浠跺垵濮嬪寲鍏ㄩ儴**鐗╃悊**鍗曞厓銆?    pub fn initialize(&mut self, cfg: &Config) {
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

    /// 鎶?*鍏ㄩ儴**鍗曞厓(鍚櫄鎷熷眰)缃负鍚屼竴鍧囧寑鐘舵€併€?    ///
    /// 杩欐槸鑷敱鏉ユ祦淇濇寔鎬ч獙璇佺殑鍓嶆彁:鏍煎紡鑻ョ鏁ｄ竴鑷?鍧囧寑鍦轰笅瀵规祦娈嬪樊銆佷汉宸?    /// 绮樻€т笌姊害閮藉簲绮剧‘涓?0銆傛敞鎰忓畠浼氳鐩栬櫄鎷熷眰,鍥犳**涓嶈兘**鍏堣皟鐢?    /// [`crate::boundary::apply`] 鈥斺€?鍥哄闀滃儚浼氳璐村澶勪笉鍐嶅潎鍖€(閭ｆ槸鐗╃悊涓婃纭?    /// 鐨?浣嗕笉鏄繖閲岃妫€楠岀殑鎬ц川)銆?    pub fn set_uniform(&mut self, cfg: &Config, rho: f64, vx: f64, vy: f64, p: f64, nut: f64) {
        let s = PrimState::from_primitives(cfg, rho, vx, vy, p, nut);
        for (i, j) in self.rho.all_indices().collect::<Vec<_>>() {
            self.write_state(i, j, &s);
        }
    }
}

/// 涓€缁勮嚜娲界殑鍘熷鍙橀噺銆?#[derive(Clone, Copy, Debug, Default)]
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
    /// 鐢?(蟻, u, v, p, 谓虄) 琛ュ叏 T銆丒銆丠銆乧銆?    pub fn from_primitives(cfg: &Config, rho: f64, vx: f64, vy: f64, p: f64, nut: f64) -> Self {
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

/// 闈炵墿鐞嗙姸鎬?瀵嗗害鎴栧帇鍔涢潪姝?銆?#[derive(Debug, Clone, Copy)]
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

/// 闈笂鐨勫伐浣滈噺銆?///
/// 娉ㄦ剰杩欓噷**涓嶅瓨**闈笂鐨勫畧鎭掗噺 鈥斺€?瀹冨彧鏄绠楁棤绮橀€氶噺鐨勪腑闂村€?鐩存帴鍦?/// [`crate::convection`] 鐨勫惊鐜噷鐢ㄥ眬閮ㄥ彉閲忕畻鎺?鐪佷竴涓暟缁勫拰涓€閬嶈瀛樸€?#[derive(Clone, Debug)]
pub struct FaceWork {
    /// 鏃犵矘閫氶噺 F路n
    pub flux: Vec5Field,
    /// 绮樻€?婀嶆祦鎵╂暎閫氶噺
    pub diff: Vec5Field,
    /// JST 浜哄伐绮樻€?    pub dissipation: Vec5Field,
    /// 闈㈣氨鍗婂緞 位f
    pub lambda: F64Field,
    /// 鑷€傚簲鑰楁暎绯绘暟
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

/// tau 闈?+ n 闈㈢殑宸ヤ綔閲?澶栧姞 JST 婵€娉㈡帰娴嬪櫒銆?#[derive(Clone, Debug)]
pub struct Faces {
    /// 鍛ㄥ悜闈?`(NI+1) x NJ`
    pub tau: FaceWork,
    /// 寰勫悜闈?`NI x NJ`
    pub nrm: FaceWork,
    /// 浠?*鍗曞厓**涓轰腑蹇冪殑鍘嬪姏鎺㈡祴鍣?i 鏂瑰悜
    pub sensor_i: F64Field,
    /// 鍚屼笂,j 鏂瑰悜
    pub sensor_j: F64Field,
    /// 閫愬崟鍏冪殑 `V/螖t_local`,闈㈣氨鍗婂緞鍙栧畠鐨勪袱渚у钩鍧?閬垮厤鍦ㄩ潰寰幆閲岄噸澶嶅仛闄ゆ硶)
    pub spec_ratio: F64Field,
}

impl Faces {
    pub fn new(ni: usize, nj: usize, halo: usize) -> Self {
        Self {
            tau: FaceWork::new(ni + 1, nj),
            nrm: FaceWork::new(ni, nj),
            // 鎺㈡祴鍣ㄤ互鍗曞厓涓轰腑蹇?蔚虏 鐨勫洓鐐瑰彇鍊奸渶瑕?[-2, N+1],halo=3 瓒冲
            sensor_i: Field::new(ni, nj, halo),
            sensor_j: Field::new(ni, nj, halo),
            spec_ratio: Field::new(ni, nj, halo),
        }
    }
}

/// 渚夸簬鏁翠綋浼犻€掔殑姹傝В鍩熴€?#[derive(Clone, Debug)]
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
        let cfg = Config::from_str(include_str!("../config.json")).unwrap();
        let mesh = Mesh::parse(include_str!("../fangdata.txt")).unwrap();
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
        let cfg = Config::from_str(include_str!("../config.json")).unwrap();
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

