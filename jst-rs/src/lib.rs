//! # JST + Spalart-Allmaras 浜岀淮鏈夐檺浣撶Н姹傝В鍣?//!
//! O 鍨嬬粨鏋勭綉鏍间笂鐨勫彲鍘?Navier-Stokes 姹傝В鍣?绌洪棿鐢ㄦ牸蹇冩湁闄愪綋绉腑蹇冩牸寮?+
//! JST 鏍囬噺浜哄伐绮樻€?婀嶆祦鐢?Spalart-Allmaras 涓€鏂圭▼妯″瀷,鏃堕棿鐢ㄦ樉寮忎簲绾?//! Runge-Kutta 鎺ㄨ繘鍒板畾甯搞€?//!
//! 鏈?crate 鏄悓鐩綍 Python 瀹炵幇鐨勯噸鍐欍€傛暟鍊兼牸寮忛€愰」瀵归綈(`tests/golden.rs`
//! 鐩存帴璇?Python 瀵煎嚭鐨勫弬鑰冩暟鎹仛浜ゅ弶楠岃瘉),浣嗘灦鏋勫仛浜嗕袱澶勬牴鏈€ц皟鏁?
//!
//! ## 1. Halo 鍖栫殑涓嬫爣绌洪棿
//!
//! Python 鎶婅櫄鎷熷崟鍏冭拷鍔犲湪鏁扮粍灏鹃儴,浜庢槸姣忎釜 kernel 閮借鑷繁鍐欎竴閬?//! "澹侀潰鍙栬繖涓€佽繙鍦哄彇閭ｄ釜銆佸垏鍓茬嚎鍙栧彟涓€涓?鐨勬槧灏勩€傝繖绫绘墜鍐欐槧灏勮础鐚簡
//! `BUGS.md` 閲岀殑鍥涗釜鏁板€奸敊璇€傝繖閲屾妸鍗曞厓涓嬫爣鎵╁睍鍒?`[-H, N+H)`,铏氭嫙鍗曞厓浣忓湪
//! 璐熶笅鏍囦笂,杈圭晫鏉′欢鏀舵暃鎴愬敮涓€鐨?[`boundary::apply`],鍏朵綑 kernel 鍏ㄦ槸鏃犵壒鍒ょ殑
//! 鐭╁舰寰幆 鈥斺€?涓€鏁寸被 bug 鍦ㄧ粨鏋勪笂琚秷闄ゃ€?//!
//! ## 2. SoA + 鎸夎骞惰
//!
//! 姣忎釜鐗╃悊閲忎竴涓繛缁?`Vec`(瑙?[`field::Field`])銆傞『搴忔壂鎻忓彧鍔犺浇鐪熸鐢ㄥ埌鐨勯噺,
//! 缂栬瘧鍣ㄥ彲鑷姩鍚戦噺鍖?鑰?杈撳嚭涓€涓暟缁勩€佽緭鍏ヨ嫢骞插埆鐨勬暟缁?杩欎竴妯″紡璁╁€熺敤妫€鏌?//! 鐩存帴璇佹槑鏃犲埆鍚?rayon 骞惰涓嶉渶瑕佷换浣?`unsafe`銆傛墍鏈夊苟琛?kernel 閮芥槸
//! 銆屾瘡涓緭鍑哄厓绱犲彧鍐欎竴娆°€佸€煎彧鐢辫緭鍏ュ喅瀹氥€?鍥犳缁撴灉涓庣嚎绋嬫暟鏃犲叧,閫愪綅鍙鐜般€?//!
//! ## 妯″潡鍒嗗眰
//!
//! ```text
//! mesh 鈹€鈻?geometry 鈹€鈹?//! config 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹尖攢鈻?state 鈹€鈻?solver 鈹€鈻?(bin/jst)
//!                   鈹?     鈻?//!         boundary 鈹€鈹?     鈹? kernels:
//!                          鈹溾攢鈹€ timestep     褰撳湴/鍏ㄥ眬鏃堕棿姝?//!                          鈹溾攢鈹€ convection   鏃犵矘瀵规祦閫氶噺
//!                          鈹溾攢鈹€ gradient     Green-Gauss 姊害
//!                          鈹溾攢鈹€ viscous      绮樻€у簲鍔?+ 婀嶆祦鎵╂暎
//!                          鈹溾攢鈹€ dissipation  JST 浜哄伐绮樻€?//!                          鈹斺攢鈹€ source       S-A 婧愰」
//! ```
//!
//! 姣忎釜 kernel 閮芥槸鎺ユ敹 `&Geometry`銆乣&Cells`銆乣&mut 杈撳嚭` 鐨勮嚜鐢卞嚱鏁?鍙互鍗曠嫭
//! 娴嬭瘯涓庡熀鍑?鏇挎崲鍏朵腑浠讳綍涓€涓?姣斿鎶婁腑蹇冩牸寮忔崲鎴?Roe 杩庨)涓嶇壍鍔ㄥ叾浣欓儴鍒嗐€?//!
//! ## 鐢ㄦ硶
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
