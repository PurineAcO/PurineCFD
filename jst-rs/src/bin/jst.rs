//! 命令行入口。

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use jst::{Config, Mesh, Solver};

#[derive(Parser, Debug)]
#[command(name = "jst", about = "2D JST + Spalart-Allmaras finite-volume solver")]
struct Cli {
    /// 网格文件
    #[arg(long, default_value = "fangdata.txt")]
    mesh: PathBuf,
    /// 配置文件
    #[arg(long, default_value = "config.json")]
    config: PathBuf,
    /// 覆盖配置中的最大迭代步数
    #[arg(long)]
    steps: Option<usize>,
    /// 结果 CSV 输出路径
    #[arg(long, default_value = "result.csv")]
    out: PathBuf,
    /// 残差历史输出路径
    #[arg(long, default_value = "res.log")]
    reslog: PathBuf,
    /// 只打印汇总,不逐步刷屏
    #[arg(long)]
    quiet: bool,
    /// 线程数(0 = 按网格规模自动选取,见 `choose_threads`)
    #[arg(long, default_value_t = 0)]
    threads: usize,
    /// 不写任何输出文件(基准测试用)
    #[arg(long)]
    no_output: bool,
}

/// 每个线程至少要摊到这么多单元才值得开。
///
/// 本求解器是**访存带宽受限**的:每个单元每级要流过约 2 KB 的状态量,算术强度
/// 很低。因此有效线程数由内存通道数而非核心数决定 —— 线程再多只会增加争用。
/// 在 i7-14650HX(12 核 / 24 线程,双通道)上实测:
///
/// ```text
///   32 768 单元:  1线程 23.0 / 2线程 15.7 / 4线程 17.2 / 8线程 24.0  ms/步
///  294 912 单元:  1线程 213  / 4线程 94.2 / 8线程 86.3 / 24线程 123  ms/步
/// ```
///
/// 最优点都落在"每线程 16K–37K 单元"附近,故取 24K 作为经验值。
/// 内存通道更多的服务器可以用 `--threads` 手动调高。
const CELLS_PER_THREAD: usize = 24_576;

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
