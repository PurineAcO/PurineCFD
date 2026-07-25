import meshreading as mr
import geometry as geo
import initialize as ini
import classconfig as cc
import solvesupple as ss
import solvemain as sm

mr.read_mesh("fangdata.txt")
geo.geometry_main("output.txt")
ini.initialization_main()
ss.formvars_main()

for step in range(1,cc.iteration):
    sm.RK(step)
    residual = ss.calc_residual()
    print(f"step:{step:6d} | residual:{residual:.6e}")
    with open ("res.log","a") as f:
        f.write(f"{step},{residual:.6e}\n")
    if residual < cc.targetres:
        break

print("totaltime:",cc.totaltime)

with open("result.csv", "w", encoding="utf-8") as f:
     f.write("cell_index,rho,p,T,u,v,miubl\n")

for i in range(1, cc.i_total + 1):
    for j in range(1, cc.j_total + 1):
        cell: cc.cell_class = cc.CellList[i][j]
        line = (
            f"{cell.index},"
            f"{cell.Dent:.8e},"
            f"{cell.P:.8e},"
            f"{cell.T:.8e},"
            f"{cell.u:.8e},"
            f"{cell.v:.8e},"
            f"{cell.miubl:.8e}\n"
        )
        with open("result.csv", "a", encoding="utf-8") as f:
            f.write(line)

print("all data is written in result.csv")
