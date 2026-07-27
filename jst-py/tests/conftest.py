"""pytest 公共夹具.

Python 基线用的是**模块级全局状态**（`cc.CellList` 等），所以每个用例都必须先
`cc.reset_state()` 再重新装配，否则算例之间会互相污染。`setup_case` 把这套流程
封成一个函数，并刻意绕开 `initialize_output` / `geometry_debug` 之类的文件写出。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import classconfig as cc  # noqa: E402
import geometry as geo  # noqa: E402
import initialize as ini  # noqa: E402
import meshreading as mr  # noqa: E402
import solvesupple as ss  # noqa: E402

FANGDATA = ROOT / "fangdata.txt"
GOLDEN_DIR = Path(__file__).parent / "golden"


def setup_case(mesh: str | Path = FANGDATA, *, with_ghosts: bool = True) -> None:
    """把全局状态重置成「刚初始化完」的算例，可选是否建立虚拟网格."""
    cc.reset_state()
    mr.read_mesh(str(mesh))
    geo.calc_cell_vol()
    geo.calc_cell_center()
    geo.calc_face_direction_tau()
    geo.calc_face_direction_n()
    geo.calc_most_near_walldistance()
    ini.initialization()
    cc.shockwave_tau = np.zeros((cc.i_total + cc.IM + 1, cc.j_total + 1))
    cc.shockwave_n = np.zeros((cc.i_total, cc.j_total + cc.IM + 1))
    cc.density_table = np.zeros((cc.i_total + 1, cc.j_total + 1))
    ss.formvars_main()
    if with_ghosts:
        ss.riemann_main()
        ss.imagination_mesh_create()


def physical_cells():
    """按行主序遍历物理单元 (i = 1…i_total-1, j = 1…j_total)."""
    for i in range(1, cc.i_total):
        for j in range(1, cc.j_total + 1):
            yield cc.CellList[i][j]


def all_cells():
    """遍历物理单元 + 全部虚拟单元."""
    for i in range(1, len(cc.CellList)):
        for j in range(1, len(cc.CellList[i])):
            yield cc.CellList[i][j]


def set_uniform(rho: float, u: float, v: float, p: float, nut: float) -> None:
    """把**所有**单元（含 ghost）置为同一个均匀状态.

    这是自由来流保持性(free-stream preservation)测试的前提：格式若离散一致，
    均匀场下的对流残差、人工粘性、梯度都应精确为 0.
    """
    E = p / (rho * (cc.gamma - 1)) + 0.5 * (u * u + v * v)
    T = p / (cc.R * rho)
    for cell in all_cells():
        cell.rho, cell.u, cell.v, cell.p = rho, u, v, p
        cell.E, cell.T, cell.miubl = E, T, nut
        cell.H = E + p / rho
        cell.c = np.sqrt(cc.gamma * cc.R * T)
        cell.ma = np.hypot(u, v) / cell.c
        cell.formvars()


@pytest.fixture
def case():
    """默认算例：仓库自带的 10x12 椭圆柱 O 型网格."""
    setup_case()
    yield cc
    cc.reset_state()


@pytest.fixture
def bare_case():
    """只有几何 + 初始场，不建虚拟网格."""
    setup_case(with_ghosts=False)
    yield cc
    cc.reset_state()
