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

# def res_and_ustep():
#     """通量推进和残差计算,残差依靠`density_table`,同时推进守恒通量"""
#     for i in range(1, cc.i_total, 1):
#         for j in range(1, cc.j_total+1, 1):
#             thiscell = cc.CellList[i][j]
#             # save the density used to calc residual
#             cc.density_table[i][j] = thiscell.rho
#             # step on the conservative variables
#             thiscell.U_former = thiscell.U.copy()

