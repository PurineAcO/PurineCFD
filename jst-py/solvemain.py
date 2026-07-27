import solvesupple as ss
import classconfig as cc


def RK(step):
    """显式多级 Runge-Kutta 推进一个时间步.

    每个时间步内时间步长固定(在级循环外求一次),各级依次重算残差:
        U^(k) = U^n - α_k·Δt·R(U^(k-1)) / vol
    """
    # ── 保存步初状态,用于残差统计与 RK 级更新 ──────────────────
    # BUGFIX: 原代码写作 `for i in (1, cc.i_total)` / `for j in (1, cc.j_total+1)`,
    #         这是在遍历长度为 2 的**元组**而非 range,既漏掉了绝大多数单元,
    #         又会以 j = j_total+1 越界写 density_table((i_total+1, j_total+1) 形状).
    for i in range(1, cc.i_total):
        for j in range(1, cc.j_total + 1):
            cell: cc.cell_class = cc.CellList[i][j]
            cc.density_table[i][j] = cell.rho
            # BUGFIX: 必须深拷贝,否则 U_former 与 U 指向同一 ndarray
            cell.U_former = cell.U.copy()

    # BUGFIX: 时间步长在整个 RK 步内保持不变;原实现每级都重算并把 Δt
    #         累加进 totaltime,导致物理时间被放大 RK_STAGES 倍.
    mintime = ss.min_timestep()
    cc.totaltime += mintime

    # BUGFIX: 原为 range(1, 5),只跑了 4 级,丢弃了最后一级 RK[5] = 1,
    #         格式实际退化且不相容(末级系数必须为 1).
    for k in range(1, cc.RK_STAGES + 1):
        ss.riemann_main()          # 远场黎曼不变量边界
        ss.imagination_mesh_update()  # 虚拟(ghost)网格同步
        ss.calc_convect()          # 对流通量
        # BUGFIX: 原求解器从未调用 calc_grad,导致 ugrad/vgrad/Tgrad/miublgrad
        #         恒为 0 —— 粘性扩散项与 S-A 源项全部失效,方程退化为无粘 Euler.
        ss.calc_grad()             # Green-Gauss 梯度
        ss.calc_diffusion()        # 粘性/湍流扩散项
        ss.calc_dissipation()      # JST 人工粘性
        ss.calc_source()           # S-A 源项
        for i in range(1, cc.i_total):
            for j in range(1, cc.j_total + 1):
                cell: cc.cell_class = cc.CellList[i][j]
                cell.URK = (cell.Fc - cell.Fv - cell.Fd - cell.S) / cell.vol
                cell.U = cell.U_former - cc.RK[k] * mintime * cell.URK
        ss.form_vars()
