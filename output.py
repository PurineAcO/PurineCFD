import classconfig as cc

def geometry_debug(debugoutput=cc.outputfile):
    """将几何结构的相关信息输出到文件 `debugoutput` 中, 包括:\n  
    单元体积、中心坐标和到壁面的最近距离,周向边和径向边的法向量和中点坐标."""
    with open(debugoutput,'w') as f:
        for i in range(1,cc.i_total,1):
            for j in range(1,cc.j_total+1,1):
                cell = cc.CellList[i][j]
                f.write(f"cell index: ({cell.index}), cell volume: {cell.vol}\n")
                f.write(f"cell center: ({cell.x}, {cell.y})\n")
                f.write(f"most near wall distance: {cell.sad}\n")
                f.write("-------------------------------------\n")

        for i in range(1,cc.i_total+1,1):
            for j in range(1,cc.j_total+1,1):
                face_tau = cc.Facelist_tau[i][j]
                f.write(f"face_tau index: ({face_tau.index}), normal vector: ({face_tau.nx}, {face_tau.ny})\n")
                f.write(f"face_tau middle point: ({face_tau.mx}, {face_tau.my})\n")
                f.write("-------------------------------------\n")

        for i in range(1, cc.i_total ):
            for j in range(1, cc.j_total + 1):
                face_n = cc.FaceList_n[i][j]
                f.write(f"face_n index: ({face_n.index}), normal vector: ({face_n.nx}, {face_n.ny})\n")
                f.write(f"face_n middle point: ({face_n.mx}, {face_n.my})\n")
                f.write("-------------------------------------\n")

def mesh_visualization(savepath=None, show_centers=True, show_tau=True, show_n=True):
    """可视化网格,如果不指定保存路径`savepath`则直接显示图像.\n
    用参数`show_centers`控制是否显示单元中心.用参数`show_tau`和`show_n`控制显示周向/径向法向."""
    import matplotlib.pyplot as plt
    import os

    fig, ax = plt.subplots(figsize=(10, 10), dpi=120)

    # circumferential lines (constant i)
    for i in range(1, cc.i_total + 1):
        xs, ys = [], []
        for j in range(1, cc.j_total + 1):
            xs.append(cc.NodeList[i][j].x)
            ys.append(cc.NodeList[i][j].y)
        xs.append(cc.NodeList[i][1].x)
        ys.append(cc.NodeList[i][1].y)
        ax.plot(xs, ys, color='steelblue', linewidth=0.8, alpha=0.8)

    # radial lines (constant j)
    for j in range(1, cc.j_total + 1):
        xs, ys = [], []
        for i in range(1, cc.i_total + 1):
            xs.append(cc.NodeList[i][j].x)
            ys.append(cc.NodeList[i][j].y)
        ax.plot(xs, ys, color='steelblue', linewidth=0.8, alpha=0.8)

    # domain extent for auto-scaling arrows
    outer = cc.i_total
    xs_outer = [cc.NodeList[outer][j].x for j in range(1, cc.j_total + 1)]
    ys_outer = [cc.NodeList[outer][j].y for j in range(1, cc.j_total + 1)]
    domain_size = max(max(xs_outer) - min(xs_outer), max(ys_outer) - min(ys_outer))
    arrow_scale = domain_size * 0.03
    arrow_width = domain_size * 0.0003

    # subsample target: ~15 arrows per direction
    skip_i = max(1, (cc.i_total + 14) // 15)
    skip_j = max(1, (cc.j_total + 14) // 15)

    # cell centers
    if show_centers:
        cx, cy = [], []
        for i in range(1, cc.i_total):
            for j in range(1, cc.j_total + 1):
                cx.append(cc.CellList[i][j].x)
                cy.append(cc.CellList[i][j].y)
        ax.scatter(cx, cy, c='crimson', s=18, zorder=5, label='cell center')

    # face_tau normals (circumferential edge → radial outward)
    if show_tau:
        tx, ty, tni, tnj = [], [], [], []
        for i in range(1, cc.i_total + 1, skip_i):
            for j in range(1, cc.j_total + 1, skip_j):
                jn = j + 1 if j < cc.j_total else 1
                mx = (cc.NodeList[i][j].x + cc.NodeList[i][jn].x) * 0.5
                my = (cc.NodeList[i][j].y + cc.NodeList[i][jn].y) * 0.5
                tx.append(mx); ty.append(my)
                tni.append(cc.Facelist_tau[i][j].nx)
                tnj.append(cc.Facelist_tau[i][j].ny)
        ax.quiver(tx, ty, tni, tnj, color='darkcyan', scale=1/arrow_scale,
                  scale_units='xy', width=arrow_width, zorder=6, label='face_tau normal')

    # face_n normals (radial edge → circumferential)
    if show_n:
        nx, ny, nni, nnj = [], [], [], []
        for j in range(1, cc.j_total + 1, skip_j):
            for i in range(1, cc.i_total, skip_i):
                mx = (cc.NodeList[i][j].x + cc.NodeList[i+1][j].x) * 0.5
                my = (cc.NodeList[i][j].y + cc.NodeList[i+1][j].y) * 0.5
                nx.append(mx); ny.append(my)
                nni.append(cc.FaceList_n[j][i].nx)
                nnj.append(cc.FaceList_n[j][i].ny)
        ax.quiver(nx, ny, nni, nnj, color='darkorange', scale=1/arrow_scale,
                  scale_units='xy', width=arrow_width, zorder=6, label='face_n normal')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'O mesh — {cc.i_total} rings x {cc.j_total} radial points')
    ax.set_aspect('equal')
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)

    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches='tight')
        print(f'mesh visualization saved to: {os.path.abspath(savepath)}')
    else:
        plt.show()
    plt.close(fig)



def initialize_output(debugoutput=cc.outputfile):
    """标准初始化的输出"""
    with open(debugoutput, "a") as f:
        cellexample = cc.CellList[1][1] # 可调节监视的网格单元
        f.write("standard initialization is completed.\n")
        f.write(f"for example,{cellexample.index}\n")
        f.write(f"cell center: ({cellexample.x}, {cellexample.y})\n")
        f.write(f"cell volume: {cellexample.vol}\n")
        f.write(f"most near wall distance: {cellexample.sad}\n")
        f.write(f"cell density: {cellexample.rho}\n")
        f.write(f"cell pressure: {cellexample.p}\n")
        f.write(f"cell temperature: {cellexample.T}\n")
        f.write(f"cell velocity: ({cellexample.u}, {cellexample.v})\n")
        f.write(f"cell energy: {cellexample.E}\n")
        f.write(f"cell enthalpy: {cellexample.H}\n")
        f.write(f"cell sound speed: {cellexample.c}\n")
        f.write(f"cell Mach number: {cellexample.ma}\n")
        f.write(f"cell dynamic viscosity: {cellexample.miu}\n")
        f.write(f"cell kinematic viscosity: {cellexample.miubl}\n")
        f.write("All the cell is initialized as the same value.\n")
        f.write(f"All the next is about cell {cellexample.index},you can change the index.\n")
        f.write("-------------------------------------\n")

def formvars_main_output(debugoutput=cc.outputfile):
    """输出守恒量计算结果"""
    with open(debugoutput, "a") as f:
        cellexample = cc.CellList[1][1] # 可调节监视的网格单元
        f.write(f"cell U: {cellexample.U}\n")

def min_timestep_output(debugoutput=cc.outputfile):
    """输出最小时间步长"""
    with open(debugoutput, "a") as f:
        cellexample = cc.CellList[1][1] # 可调节监视的网格单元
        f.write(f"global dt: {cellexample.dt:.6e}\n")
        f.write(f"totaltime: {cc.totaltime:.6e}\n")