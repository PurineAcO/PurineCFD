"""几何离散的正确性：度量闭合、面积守恒、法向定向、壁面距离."""

from __future__ import annotations

import math

import numpy as np
import pytest

import classconfig as cc
from conftest import physical_cells, setup_case


def _shoelace(ring: int) -> float:
    """第 `ring` 层节点多边形的有向面积（逆时针为正）."""
    s = 0.0
    for j in range(1, cc.j_total + 1):
        jn = j + 1 if j < cc.j_total else 1
        a, b = cc.NodeList[ring][j], cc.NodeList[ring][jn]
        s += a.x * b.y - b.x * a.y
    return 0.5 * s


def test_metric_closure(bare_case):
    """单元四个面的外法向之和必须精确为 0.

    这是自由来流保持性的**充要几何条件**：Fc = Σ±F(U)·n 对均匀 U 退化为
    F(U)·(Σ±n)，只有闭合才恒等于 0。任何一个法向写错都会在这里暴露。
    """
    scale = max(abs(f.nx) + abs(f.ny)
                for i in range(1, cc.i_total + 1)
                for f in (cc.Facelist_tau[i][j] for j in range(1, cc.j_total + 1)))
    worst = 0.0
    for i in range(1, cc.i_total):
        for j in range(1, cc.j_total + 1):
            jp1 = j + 1 if j < cc.j_total else 1
            sx = (cc.Facelist_tau[i + 1][j].nx - cc.Facelist_tau[i][j].nx
                  + cc.FaceList_n[i][jp1].nx - cc.FaceList_n[i][j].nx)
            sy = (cc.Facelist_tau[i + 1][j].ny - cc.Facelist_tau[i][j].ny
                  + cc.FaceList_n[i][jp1].ny - cc.FaceList_n[i][j].ny)
            worst = max(worst, abs(sx), abs(sy))
    assert worst < 1e-13 * scale, f"metric not closed: max |Σn| = {worst:.3e}"


def test_total_area_matches_polygon_annulus(bare_case):
    """所有单元面积之和 == 外环多边形面积 − 内环多边形面积（解析恒等式）."""
    total = sum(c.vol for c in physical_cells())
    expected = _shoelace(cc.i_total) - _shoelace(1)
    assert total == pytest.approx(expected, rel=1e-13)


def test_all_volumes_positive(bare_case):
    assert all(c.vol > 0 for c in physical_cells())


def test_centroid_inside_bounding_box(bare_case):
    """单元形心必须落在其四个顶点的包围盒内."""
    for i in range(1, cc.i_total):
        for j in range(1, cc.j_total + 1):
            jn = j + 1 if j < cc.j_total else 1
            xs = [cc.NodeList[i][j].x, cc.NodeList[i + 1][j].x,
                  cc.NodeList[i + 1][jn].x, cc.NodeList[i][jn].x]
            ys = [cc.NodeList[i][j].y, cc.NodeList[i + 1][j].y,
                  cc.NodeList[i + 1][jn].y, cc.NodeList[i][jn].y]
            c = cc.CellList[i][j]
            assert min(xs) - 1e-12 <= c.x <= max(xs) + 1e-12
            assert min(ys) - 1e-12 <= c.y <= max(ys) + 1e-12


def test_tau_normal_points_outward(bare_case):
    """周向面(tau)的法向应指向径向外侧：n·(面心 − 原点) > 0."""
    for i in range(1, cc.i_total + 1):
        for j in range(1, cc.j_total + 1):
            f = cc.Facelist_tau[i][j]
            assert f.nx * f.mx + f.ny * f.my > 0, f"tau face {(i, j)} normal points inward"


def test_n_normal_is_counterclockwise(bare_case):
    """径向面(n)的法向应指向周向 +j 侧：与切向 (−y, x) 同号."""
    for i in range(1, cc.i_total):
        for j in range(1, cc.j_total + 1):
            f = cc.FaceList_n[i][j]
            assert f.nx * (-f.my) + f.ny * f.mx > 0, f"n face {(i, j)} normal flipped"


def test_face_normal_magnitude_equals_edge_length(bare_case):
    """有限体积法中法向的模长必须等于边长（面积加权法向）."""
    for i in range(1, cc.i_total + 1):
        for j in range(1, cc.j_total + 1):
            jn = j + 1 if j < cc.j_total else 1
            a, b = cc.NodeList[i][j], cc.NodeList[i][jn]
            f = cc.Facelist_tau[i][j]
            assert math.hypot(f.nx, f.ny) == pytest.approx(math.hypot(b.x - a.x, b.y - a.y), rel=1e-14)


def test_wall_distance_positive_and_increasing(bare_case):
    """壁面距离恒正，且沿径向单调增加（本网格族为同心外扩）."""
    for j in range(1, cc.j_total + 1):
        prev = -1.0
        for i in range(1, cc.i_total):
            sad = cc.CellList[i][j].sad
            assert sad > 0
            assert sad > prev, f"sad not increasing at {(i, j)}"
            prev = sad


def test_geometry_independent_of_mesh_resolution(tmp_path):
    """网格加密时，总面积应收敛到解析的「圆 − 椭圆」面积."""
    import sys
    sys.path.insert(0, str((__import__("pathlib").Path(__file__).resolve().parent.parent)))
    from tools.genmesh import gen_o_mesh, write_mesh

    analytic = math.pi * 5.0 * 5.0 - math.pi * 1.0 * 0.5
    errs = []
    for nj in (32, 128):
        p = tmp_path / f"m{nj}.txt"
        write_mesh(p, *gen_o_mesh(8, nj))
        setup_case(p, with_ghosts=False)
        errs.append(abs(sum(c.vol for c in physical_cells()) - analytic))
    cc.reset_state()
    # 多边形逼近圆的面积误差 ~ O(1/nj²)：nj 翻两番，误差应降到 ~1/16
    assert errs[1] < errs[0] / 10, f"area not converging: {errs}"


def test_metric_closure_on_stretched_mesh(tmp_path):
    """在近壁加密（几何拉伸）的网格上度量闭合同样成立."""
    import sys
    sys.path.insert(0, str((__import__("pathlib").Path(__file__).resolve().parent.parent)))
    from tools.genmesh import gen_o_mesh, write_mesh

    p = tmp_path / "stretched.txt"
    write_mesh(p, *gen_o_mesh(16, 48, stretch=1.25))
    setup_case(p, with_ghosts=False)
    worst = 0.0
    for i in range(1, cc.i_total):
        for j in range(1, cc.j_total + 1):
            jp1 = j + 1 if j < cc.j_total else 1
            worst = max(
                worst,
                abs(cc.Facelist_tau[i + 1][j].nx - cc.Facelist_tau[i][j].nx
                    + cc.FaceList_n[i][jp1].nx - cc.FaceList_n[i][j].nx),
                abs(cc.Facelist_tau[i + 1][j].ny - cc.Facelist_tau[i][j].ny
                    + cc.FaceList_n[i][jp1].ny - cc.FaceList_n[i][j].ny),
            )
    cc.reset_state()
    assert worst < 1e-12
