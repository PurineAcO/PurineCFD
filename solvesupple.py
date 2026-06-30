import classconfig as cc
import output as ot

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

def res_and_ustep():
    """通量推进和残差计算,残差依靠`density_table`,同时推进守恒通量"""
    for i in range(1, cc.i_total, 1):
        for j in range(1, cc.j_total+1, 1):
            thiscell = cc.CellList[i][j]
            # save the density used to calc residual
            cc.density_table[i][j] = thiscell.rho
            # step on the conservative variables
            thiscell.U_former = thiscell.U.copy()

