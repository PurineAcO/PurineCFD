"""网格生成器与网格读取."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import classconfig as cc  # noqa: E402
import meshreading as mr  # noqa: E402
from conftest import FANGDATA  # noqa: E402
from tools.genmesh import gen_o_mesh, write_mesh  # noqa: E402


def test_reproduces_fangdata():
    """默认参数 + (10,12) 必须逐点复现仓库自带的 fangdata.txt.

    这保证「新生成的网格」与「原始输入」属于同一族，扫分辨率做性能基准时
    不会偷换算例。
    """
    _, _, pts = gen_o_mesh(10, 12)
    ref = [tuple(map(float, ln.split()))
           for ln in FANGDATA.read_text().splitlines()[1:] if ln.strip()]
    assert len(pts) == len(ref) == 120
    for (gx, gy), (rx, ry) in zip(pts, ref):
        assert gx == pytest.approx(rx, abs=1e-9)
        assert gy == pytest.approx(ry, abs=1e-9)


def test_header_matches_point_count(tmp_path):
    p = tmp_path / "m.txt"
    ni, nj, pts = gen_o_mesh(7, 20)
    write_mesh(p, ni, nj, pts)
    head = p.read_text().splitlines()[0].split()
    assert list(map(int, head)) == [7, 20]
    assert len(p.read_text().splitlines()) == 1 + 7 * 20


def test_rings_are_not_closed(tmp_path):
    """网格不封口：每环末点不得与首点重合，否则 read_mesh 会自动削掉一列."""
    _, nj, pts = gen_o_mesh(6, 16)
    first, last = pts[0], pts[nj - 1]
    assert math.hypot(first[0] - last[0], first[1] - last[1]) > 1e-6


def test_read_mesh_roundtrip(tmp_path):
    p = tmp_path / "m.txt"
    write_mesh(p, *gen_o_mesh(9, 24))
    cc.reset_state()
    mr.read_mesh(str(p))
    assert (cc.i_total, cc.j_total, cc.meshcnt) == (9, 24, 216)
    assert cc.NodeList[1][1].x == pytest.approx(1.0)
    assert cc.NodeList[1][1].y == pytest.approx(0.0)
    cc.reset_state()


def test_read_mesh_detects_closed_rings(tmp_path):
    """首尾重合的网格应被自动检测并把 j_total 减 1."""
    ni, nj, pts = gen_o_mesh(5, 16)
    closed = []
    for i in range(ni):
        ring = pts[i * nj:(i + 1) * nj]
        closed.extend(ring + [ring[0]])
    p = tmp_path / "closed.txt"
    p.write_text(f"{ni} {nj + 1}\n" + "".join(f"{x:.10f} {y:.10f}\n" for x, y in closed))
    cc.reset_state()
    mr.read_mesh(str(p))
    assert cc.j_total == nj
    assert cc.meshcnt == ni * nj
    cc.reset_state()


def test_stretching_clusters_near_wall(tmp_path):
    """stretch > 1 时近壁第一层间距应明显小于均匀分布."""
    _, nj, uni = gen_o_mesh(16, 32, stretch=1.0)
    _, _, str_ = gen_o_mesh(16, 32, stretch=1.3)
    d_uni = math.hypot(uni[nj][0] - uni[0][0], uni[nj][1] - uni[0][1])
    d_str = math.hypot(str_[nj][0] - str_[0][0], str_[nj][1] - str_[0][1])
    assert d_str < 0.5 * d_uni


def test_rejects_degenerate_sizes():
    with pytest.raises(ValueError):
        gen_o_mesh(3, 32)       # JST 四阶模板需要至少 4 层
    with pytest.raises(ValueError):
        gen_o_mesh(10, 4)
