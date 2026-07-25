import solvesupple as ss
import classconfig as cc

def solvering(step):
    for i in (1,cc.i_total):
        for j in (1,cc.j_total+1):
            cell : cc.cell_class = cc.CellList[i][j]
            cc.density_table[i][j] = cell.rho
            cell.U_former = cell.U

    for k in range(1.5):
        mintime = ss.min_timestep()
        ss.riemann_main()
        if step==1:ss.imagination_mesh_create()
        else: ss.imagination_mesh_update()
        ss.calc_convect()
        ss.calc_diffusion()
        ss.calc_dissipation()
        ss.calc_source()
        for i in range(1,cc.i_total):
            for j in range(1,cc.j_total+1):
                cell:cc.cell_class = cc.CellList[i][j]
                cell.URK = (cell.Fc-cell.Fv-cell.Fd-cell.S)/cell.vol
                cell.U = cell.U_former-cc.RK[k]*mintime*cell.URK
        ss.form_vars()