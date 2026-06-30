import classconfig as cc
import output as ot
import math

def formvars(cell: cc.cell_class):
    """计算每个单元的守恒量,并存储在`CellList` 中."""
    cell.U[1] = cell.rho
    cell.U[2] = cell.rho * cell.u
    cell.U[3] = cell.rho * cell.v
    cell.U[4] = cell.rho * cell.E
    cell.U[5] = cell.rho * cell.miubl

def formvars_main():
    """执行守恒量计算,并将结果存储在`CellList` 中."""
    for i in range(1, cc.i_total,1):  # 一定要注意i的范围是什么
        for j in range(1,cc.j_total+1,1):
            formvars(cc.CellList[i][j])
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
            cell = cc.CellList[i][j]
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

    # 分配假想网格的空间,位于`CellList`的尾部`1~IM`行.
    for im in range(1, cc.IM + 1):
        ghost_row = [[]]                       # j=0 占位
        for j in range(1, cc.j_total + 1):
            ghost_row.append(cc.cell_class((cc.i_total + im - 1, j)))
        cc.CellList.append(ghost_row)

    for j in range(1, cc.j_total + 1):
        for im in range(1, cc.IM + 1):
            # 标量将直接被复制过去
            cc.CellList[cc.i_total + im - 1][j].rho = cc.CellList[1][j].rho
            cc.CellList[cc.i_total + im - 1][j].p = cc.CellList[1][j].p
            cc.CellList[cc.i_total + im - 1][j].T = cc.CellList[1][j].T
            cc.CellList[cc.i_total + im - 1][j].E = cc.CellList[1][j].E
            cc.CellList[cc.i_total + im - 1][j].H = cc.CellList[1][j].H
            cc.CellList[cc.i_total + im - 1][j].c = cc.CellList[1][j].c
            
            # 速度和湍流函数将被取相反数
            cc.CellList[cc.i_total + im - 1][j].u = - cc.CellList[im][j].u
            cc.CellList[cc.i_total + im - 1][j].v = - cc.CellList[im][j].v
            cc.CellList[cc.i_total + im - 1][j].miubl = -cc.CellList[im][j].miubl
            cc.CellList[cc.i_total + im - 1][j].ma = (math.sqrt(cc.CellList[im][j].u**2 + 
                                                    cc.CellList[im][j].v**2) / cc.CellList[1][j].c)

# def res_and_ustep():
#     """通量推进和残差计算,残差依靠`density_table`,同时推进守恒通量"""
#     for i in range(1, cc.i_total, 1):
#         for j in range(1, cc.j_total+1, 1):
#             thiscell = cc.CellList[i][j]
#             # save the density used to calc residual
#             cc.density_table[i][j] = thiscell.rho
#             # step on the conservative variables
#             thiscell.U_former = thiscell.U.copy()

