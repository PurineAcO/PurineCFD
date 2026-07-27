"""JST + Spalart-Allmaras 二维 O 型网格有限体积求解器 —— 驱动脚本."""

import argparse
import time

import meshreading as mr
import geometry as geo
import initialize as ini
import classconfig as cc
import solvesupple as ss
import solvemain as sm


def setup(meshfile="fangdata.txt", debugfile="output.txt"):
    """读网格 → 建几何 → 初始化流场 → 建守恒量 → 建虚拟网格."""
    mr.read_mesh(meshfile)
    geo.geometry_main(debugfile)
    ini.initialization_main()
    ss.formvars_main()
    # BUGFIX: 虚拟网格必须在时间推进之前建立一次.原代码写在 RK 级循环里
    #         (`if step==1`),会在第 1 步的每一级重复 append 出多余的 ghost 行.
    ss.riemann_main()
    ss.imagination_mesh_create()


def run(max_step=None, reslog="res.log", verbose=True):
    """时间推进主循环,返回 (实际步数, 最终残差)."""
    limit = cc.iteration if max_step is None else max_step
    # BUGFIX: 原以 "a" 模式打开,多次运行会把残差历史首尾相接
    with open(reslog, "w", encoding="utf-8") as f:
        f.write("step,residual\n")

    residual = float("nan")
    step = 0
    # BUGFIX: range(1, iteration) 少跑一步,应为 iteration+1
    for step in range(1, limit + 1):
        sm.RK(step)
        residual = ss.calc_residual()
        if verbose:
            print(f"step:{step:6d} | residual:{residual:.6e}")
        with open(reslog, "a", encoding="utf-8") as f:
            f.write(f"{step},{residual:.6e}\n")
        if residual < cc.targetres:
            break
    return step, residual


def write_result(path="result.csv"):
    """导出物理单元的流场.仅覆盖 i = 1…i_total-1, j = 1…j_total."""
    # BUGFIX: 原循环写作 range(1, i_total+1),越过物理单元上界(i_total-1)
    #         把壁面 ghost 也写了出来;且字段名 `cell.Dent` 根本不存在
    #         (应为 `cell.rho`),运行到此处必然 AttributeError.
    with open(path, "w", encoding="utf-8") as f:
        f.write("i,j,x,y,rho,p,T,u,v,miubl\n")
        # BUGFIX: 原实现对每一行重新 open/close 文件一次,O(N) 次系统调用
        for i in range(1, cc.i_total):
            for j in range(1, cc.j_total + 1):
                cell: cc.cell_class = cc.CellList[i][j]
                f.write(
                    f"{i},{j},"
                    f"{cell.x:.8e},{cell.y:.8e},"
                    f"{cell.rho:.8e},"
                    f"{cell.p:.8e},"
                    f"{cell.T:.8e},"
                    f"{cell.u:.8e},"
                    f"{cell.v:.8e},"
                    f"{cell.miubl:.8e}\n"
                )


def main():
    ap = argparse.ArgumentParser(description="JST/S-A 2D finite-volume solver")
    ap.add_argument("--mesh", default="fangdata.txt")
    ap.add_argument("--steps", type=int, default=None, help="覆盖 config.json 的最大迭代步数")
    ap.add_argument("--out", default="result.csv")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    setup(args.mesh)
    t0 = time.perf_counter()
    step, residual = run(max_step=args.steps, verbose=not args.quiet)
    wall = time.perf_counter() - t0

    print(f"steps: {step}  final residual: {residual:.6e}")
    print(f"totaltime (physical): {cc.totaltime:.6e} s")
    print(f"wall clock: {wall:.3f} s  ({wall / max(step, 1) * 1e3:.3f} ms/step)")
    write_result(args.out)
    print(f"all data is written in {args.out}")


if __name__ == "__main__":
    main()
