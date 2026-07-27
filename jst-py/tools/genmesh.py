#!/usr/bin/env python3
"""O 型网格生成器.

生成的网格族与仓库自带的 `fangdata.txt` 完全同源：内边界是半轴 (a_w, b_w) 的椭圆
柱，外边界是半径 R∞ 的圆，中间沿径向插值。取默认参数并令 (i_total, j_total)
= (10, 12)、径向线性分布时，可**逐位复现** `fangdata.txt`（见
`tests/test_meshgen.py::test_reproduces_fangdata`），因此可以放心地把它当作
分辨率扫描与性能基准的网格来源。

    uv run python tools/genmesh.py --ni 129 --nj 256 -o meshes/cyl_129x256.txt

网格文件格式（与 `meshreading.read_mesh` 一致）::

    i_total j_total
    x y            ← 逐环、环内逆时针，共 i_total*j_total 行
    ...
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path


def gen_o_mesh(
    ni: int,
    nj: int,
    a_wall: float = 1.0,
    b_wall: float = 0.5,
    r_far: float = 5.0,
    stretch: float = 1.0,
) -> tuple[int, int, list[tuple[float, float]]]:
    """构造椭圆柱 → 远场圆的 O 型网格.

    Args:
        ni: 径向节点环数(含壁面环与远场环).
        nj: 周向节点数(不封口,末点与首点不重合).
        a_wall, b_wall: 壁面椭圆的半长轴/半短轴.
        r_far: 远场圆半径.
        stretch: 径向拉伸比. 1.0 为均匀分布; >1 时近壁加密,
            第 k 层间距按 `stretch**k` 递增(几何级数).

    Returns:
        `(ni, nj, points)`，`points` 按 i 外循环、j 内循环排列.
    """
    if ni < 4:
        raise ValueError("ni must be >= 4 (JST 的四阶耗散模板需要至少 4 层)")
    if nj < 8:
        raise ValueError("nj must be >= 8")
    if stretch <= 0:
        raise ValueError("stretch must be > 0")

    # 径向参数 s ∈ [0, 1]：s=0 贴壁，s=1 远场
    if abs(stretch - 1.0) < 1e-12:
        svals = [i / (ni - 1) for i in range(ni)]
    else:
        # 几何级数间距，归一化到 [0, 1]
        widths = [stretch**k for k in range(ni - 1)]
        total = sum(widths)
        svals, acc = [0.0], 0.0
        for w in widths:
            acc += w
            svals.append(acc / total)

    points: list[tuple[float, float]] = []
    for s in svals:
        a = a_wall + s * (r_far - a_wall)
        b = b_wall + s * (r_far - b_wall)
        for j in range(nj):
            theta = 2.0 * math.pi * j / nj
            points.append((a * math.cos(theta), b * math.sin(theta)))
    return ni, nj, points


def write_mesh(path: str | Path, ni: int, nj: int, points: list[tuple[float, float]]) -> None:
    lines = [f"{ni} {nj}\n"]
    lines += [f"{x:.10f} {y:.10f}\n" for x, y in points]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ni", type=int, default=10, help="径向节点环数")
    ap.add_argument("--nj", type=int, default=12, help="周向节点数")
    ap.add_argument("--a-wall", type=float, default=1.0)
    ap.add_argument("--b-wall", type=float, default=0.5)
    ap.add_argument("--r-far", type=float, default=5.0)
    ap.add_argument("--stretch", type=float, default=1.0, help="径向几何拉伸比(>1 近壁加密)")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    ni, nj, pts = gen_o_mesh(args.ni, args.nj, args.a_wall, args.b_wall, args.r_far, args.stretch)
    write_mesh(args.out, ni, nj, pts)
    print(f"wrote {args.out}: {ni} rings x {nj} points = {len(pts)} nodes, "
          f"{(ni - 1) * nj} cells")


if __name__ == "__main__":
    main()
