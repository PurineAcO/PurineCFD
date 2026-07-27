//! 鍛戒护琛屽叆鍙ｃ€?
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use jst::{Config, Mesh, Solver};

#[derive(Parser, Debug)]
#[command(name = "jst", about = "2D JST + Spalart-Allmaras finite-volume solver")]
struct Cli {
    /// 缃戞牸鏂囦欢
    #[arg(long, default_value = "fangdata.txt")]
    mesh: PathBuf,
    /// 閰嶇疆鏂囦欢
    #[arg(long, default_value = "config.json")]
    config: PathBuf,
    /// 瑕嗙洊閰嶇疆涓殑鏈€澶ц凯浠ｆ鏁?    #[arg(long)]
    steps: Option<usize>,
    /// 缁撴灉 CSV 杈撳嚭璺緞
    #[arg(long, default_value = "result.csv")]
    out: PathBuf,
    /// 娈嬪樊鍘嗗彶杈撳嚭璺緞
    #[arg(long, default_value = "res.log")]
    reslog: PathBuf,
    /// 鍙墦鍗版眹鎬?涓嶉€愭鍒峰睆
    #[arg(long)]
    quiet: bool,
    /// 绾跨▼鏁?0 = 鎸夌綉鏍艰妯¤嚜鍔ㄩ€夊彇,瑙?`choose_threads`)
    #[arg(long, default_value_t = 0)]
    threads: usize,
    /// 涓嶅啓浠讳綍杈撳嚭鏂囦欢(鍩哄噯娴嬭瘯鐢?
    #[arg(long)]
    no_output: bool,
}

/// 姣忎釜绾跨▼鑷冲皯瑕佹憡鍒拌繖涔堝鍗曞厓鎵嶅€煎緱寮€銆?///
/// 鏈眰瑙ｅ櫒鏄?*璁垮瓨甯﹀鍙楅檺**鐨?姣忎釜鍗曞厓姣忕骇瑕佹祦杩囩害 2 KB 鐨勭姸鎬侀噺,绠楁湳寮哄害
/// 寰堜綆銆傚洜姝ゆ湁鏁堢嚎绋嬫暟鐢卞唴瀛橀€氶亾鏁拌€岄潪鏍稿績鏁板喅瀹?鈥斺€?绾跨▼鍐嶅鍙細澧炲姞浜夌敤銆?/// 鍦?i7-14650HX(12 鏍?/ 24 绾跨▼,鍙岄€氶亾)涓婂疄娴?
///
/// ```text
///   32 768 鍗曞厓:  1绾跨▼ 23.0 / 2绾跨▼ 15.7 / 4绾跨▼ 17.2 / 8绾跨▼ 24.0  ms/姝?///  294 912 鍗曞厓:  1绾跨▼ 213  / 4绾跨▼ 94.2 / 8绾跨▼ 86.3 / 24绾跨▼ 123  ms/姝?/// ```
///
/// 鏈€浼樼偣閮借惤鍦?姣忕嚎绋?16K鈥?7K 鍗曞厓"闄勮繎,鏁呭彇 24K 浣滀负缁忛獙鍊笺€?/// 鍐呭瓨閫氶亾鏇村鐨勬湇鍔″櫒鍙互鐢?`--threads` 鎵嬪姩璋冮珮銆?const CELLS_PER_THREAD: usize = 24_576;

fn choose_threads(n_cells: usize, available: usize) -> usize {
    n_cells.div_ceil(CELLS_PER_THREAD).clamp(1, available.max(1))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let cfg = Config::from_path(&cli.config)?;
    let mesh = Mesh::from_path(&cli.mesh)?;
    println!(
        "mesh {}: {} rings x {} points -> {} cells",
        cli.mesh.display(),
        mesh.n_rings(),
        mesh.n_theta(),
        mesh.n_cells()
    );

    let threads = if cli.threads > 0 {
        cli.threads
    } else {
        choose_threads(
            mesh.n_cells(),
            std::thread::available_parallelism().map_or(1, |n| n.get()),
        )
    };
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()?;
    println!("threads: {threads}");

    let mut solver = Solver::new(cfg, &mesh);

    let mut log = if cli.no_output {
        None
    } else {
        let mut f = std::io::BufWriter::new(std::fs::File::create(&cli.reslog)?);
        writeln!(f, "step,residual")?;
        Some(f)
    };

    let t0 = Instant::now();
    let quiet = cli.quiet;
    let report = solver.run(cli.steps, |step, res| {
        if !quiet {
            println!("step:{step:6} | residual:{res:.6e}");
        }
        if let Some(f) = log.as_mut() {
            let _ = writeln!(f, "{step},{res:.6e}");
        }
    })?;
    let wall = t0.elapsed();

    println!(
        "steps: {}  final residual: {:.6e}{}",
        report.steps,
        report.residual,
        if report.converged { "  (converged)" } else { "" }
    );
    println!("totaltime (physical): {:.6e} s", report.totaltime);
    println!(
        "wall clock: {:.6} s  ({:.4} ms/step, {:.1} ns/cell/step)",
        wall.as_secs_f64(),
        wall.as_secs_f64() / report.steps as f64 * 1e3,
        wall.as_secs_f64() / report.steps as f64 / solver.n_cells() as f64 * 1e9
    );

    if !cli.no_output {
        solver.write_csv(&cli.out)?;
        println!("all data is written in {}", cli.out.display());
    }
    Ok(())
}
