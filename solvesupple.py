import classconfig as cc
import output as ot
import math

def formvars_main():
    """执行守恒量计算,并将结果存储在`CellList` 中."""
    for i in range(1, cc.i_total,1):  # 一定要注意i的范围是什么
        for j in range(1,cc.j_total+1,1):
            cc.CellList[i][j].formvars()
    ot.formvars_main_output()

def min_timestep():
    """计算各单元的当地时间步长 (`localdt`), 找出全局最小值,
    然后将所有单元的实际推进时间步 dt 统一设为该最小值,
    并累加到 `totaltime`."""
    mintime = float('inf')

    # 第一轮:计算各单元 localdt, 记录最小值
    for j in range(1, cc.j_total + 1):
        jp1 = j + 1 if j < cc.j_total else 1          # 周向回绕
        for i in range(1, cc.i_total):
            cell : cc.cell_class= cc.CellList[i][j]
            A = 0.5 * (cc.Facelist_tau[i][j].ni+ cc.Facelist_tau[i+1][j].ni)
            B = 0.5 * (cc.Facelist_tau[i][j].nj+ cc.Facelist_tau[i+1][j].nj)
            C = 0.5 * (cc.FaceList_n[j][i].ni+ cc.FaceList_n[jp1][i].ni)
            D = 0.5 * (cc.FaceList_n[j][i].nj+ cc.FaceList_n[jp1][i].nj)
            E = abs(cell.u * A + cell.v * B)
            F = abs(cell.u * C + cell.v * D)
            G = math.sqrt(A * A + B * B)
            L = math.sqrt(C * C + D * D)
            cell.localdt = cc.CFL * cell.vol / (E + F + cell.c * (G + L))

            if cell.localdt < mintime:
                mintime = cell.localdt

    # 第二轮:所有单元均使用全局最小时间步推进
    for j in range(1, cc.j_total + 1):
        for i in range(1, cc.i_total):
            cc.CellList[i][j].dt = mintime

    cc.totaltime += mintime
    ot.min_timestep_output()

def IM_wall():
    """设置内壁面假想网格边界条件,使用基于镜像速度的方法"""

    for im in range(1, cc.IM + 1):
        ghost_row = [[]]                       # j=0 占位
        for j in range(1, cc.j_total + 1):
            gcell : cc.cell_class = cc.cell_class((cc.i_total + im - 1, j))

            # 标量: 从壁面直接复制
            gcell.rho = cc.CellList[1][j].rho
            gcell.p   = cc.CellList[1][j].p
            gcell.T   = cc.CellList[1][j].T
            gcell.E   = cc.CellList[1][j].E
            gcell.H   = cc.CellList[1][j].H
            gcell.c   = cc.CellList[1][j].c

            # 速度 / 湍流粘度: 取对应内层的相反数 (镜像反射)
            gcell.u     = -cc.CellList[im][j].u
            gcell.v     = -cc.CellList[im][j].v
            gcell.miubl = -cc.CellList[im][j].miubl
            gcell.ma = (math.sqrt(cc.CellList[im][j].u ** 2 +
                                  cc.CellList[im][j].v ** 2) / cc.CellList[1][j].c)
            gcell.formvars()
            ghost_row.append(gcell)
        cc.CellList.append(ghost_row)


def IM_far():
    """设置压力远场假想网格边界条件"""

    for im in range(1, cc.IM + 1):
        ghost_row = [[]]             # j=0 占位
        for j in range(1, cc.j_total + 1):
            gcell = cc.cell_class((cc.i_total + im - 1, j))
            gcell.copy_flow_fields(cc.CellList[cc.i_total - 1][j])
            gcell.formvars()
            ghost_row.append(gcell)
        cc.CellList.append(ghost_row)


def IM_LR():
    """设置 O 型网格切割线两侧的周期假想网格 (j 方向周期边界)

    左侧 ghost ← 右侧物理端 (j = j_total, j_total-1, ...)
    右侧 ghost ← 左侧物理端 (j = 1, 2, ...)"""

    for i in range(1, cc.i_total):
        # ── 左侧假想网格 ──
        for im in range(1, cc.IM + 1):
            gcell = cc.cell_class((i, cc.j_total + im))
            gcell.copy_flow_fields(cc.CellList[i][cc.j_total - im + 1])
            gcell.formvars()
            cc.CellList[i].append(gcell)

        # ── 右侧假想网格 ──
        for im in range(1, cc.IM + 1):
            gcell = cc.cell_class((i, cc.j_total + cc.IM + im))
            gcell.copy_flow_fields(cc.CellList[i][im])
            gcell.formvars()
            cc.CellList[i].append(gcell)

def formIM():
    IM_wall()
    IM_far()
    IM_LR()

# def res_and_ustep():
#     """通量推进和残差计算,残差依靠`density_table`,同时推进守恒通量"""
#     for i in range(1, cc.i_total, 1):
#         for j in range(1, cc.j_total+1, 1):
#             thiscell = cc.CellList[i][j]
#             # save the density used to calc residual
#             cc.density_table[i][j] = thiscell.rho
#             # step on the conservative variables
#             thiscell.U_former = thiscell.U.copy()

