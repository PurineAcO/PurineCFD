import classconfig as cc
import boundary as bd
from grad import green_gauss
import turbulence as tb
import dissipation as hs
import output as ot
import math

def formvars_main():
    """执行守恒量计算,并将结果存储在`CellList` 中."""
    for i in range(1, cc.i_total,1):  # 一定要注意i的范围是什么
        for j in range(1,cc.j_total+1,1):
            cell : cc.cell_class = cc.CellList[i][j]
            cell.formvars()

def min_timestep():
    """计算各单元的当地时间步长 (`localdt`), 找出全局最小值,
    然后将所有单元的实际推进时间步 dt 统一设为该最小值,并累加到 `totaltime`."""

    mintime = float('inf')
    # 第一轮:计算各单元 localdt, 记录最小值
    for j in range(1, cc.j_total + 1):
        jp1 = j + 1 if j < cc.j_total else 1          # 周向回绕
        for i in range(1, cc.i_total):
            cell : cc.cell_class= cc.CellList[i][j]
            A = 0.5 * (cc.Facelist_tau[i][j].nx+ cc.Facelist_tau[i+1][j].nx)
            B = 0.5 * (cc.Facelist_tau[i][j].ny+ cc.Facelist_tau[i+1][j].ny)
            C = 0.5 * (cc.FaceList_n[i][j].nx+ cc.FaceList_n[i][jp1].nx)
            D = 0.5 * (cc.FaceList_n[i][j].ny+ cc.FaceList_n[i][jp1].ny)
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
    return mintime

def riemann_main():
    """使用压力远场更新边界条件"""
    for j in range(1,cc.j_total+1):
        bd.riemann(j)

def imagination_mesh_create():
    """设立虚拟网格"""
    # 设置壁面虚拟网格,使用镜像法
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

    # 设置远场虚拟网格,有关数据从边界条件计算!
    for im in range(1, cc.IM + 1):
        ghost_row = [[]]             # j=0 占位
        for j in range(1, cc.j_total + 1):
            face : cc.face_class = cc.Facelist_tau[cc.i_total][j]
            gcell = cc.cell_class((cc.i_total + im - 1, j))
            gcell.rho = face.rho
            gcell.E = face.E
            gcell.p = face.p
            gcell.T = face.T
            # gcell.H
            gcell.u = face.u
            gcell.v = face.v
            gcell.ma = (face.u**2+face.v**2)/(cc.gamma*cc.R*face.T)
            gcell.miubl = face.miubl
            gcell.formvars()
            ghost_row.append(gcell)
        cc.CellList.append(ghost_row)

    # 设置 O 型网格切割线两侧的周期假想网格 (j 方向周期边界)
    # 左侧 ghost ← 右侧物理端 (j = j_total, j_total-1, ...)
    # 右侧 ghost ← 左侧物理端 (j = 1, 2, ...)
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

def imagination_mesh_update():
    """更新虚拟网格"""
    # 设置壁面虚拟网格,使用镜像法
    for im in range(1, cc.IM + 1):
        ghost_row = [[]]                       # j=0 占位
        for j in range(1, cc.j_total + 1):
            gcell : cc.cell_class = cc.CellList[cc.i_total + im - 1][j]

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

    # 设置远场虚拟网格,有关数据从边界条件计算!
    for im in range(1, cc.IM + 1):
        ghost_row = [[]]             # j=0 占位
        for j in range(1, cc.j_total + 1):
            face : cc.face_class = cc.Facelist_tau[cc.i_total][j]
            # gcell = cc.cell_class((cc.i_total + im - 1, j))
            gcell :cc.cell_class = cc.CellList[cc.i_total-1+cc.IM+im][j]
            gcell.rho = face.rho
            gcell.E = face.E
            gcell.p = face.p
            gcell.T = face.T
            # gcell.H
            gcell.u = face.u
            gcell.v = face.v
            gcell.ma = (face.u**2+face.v**2)/(cc.gamma*cc.R*face.T)
            gcell.miubl = face.miubl
            gcell.formvars()

    # 设置 O 型网格切割线两侧的周期假想网格 (j 方向周期边界)
    # 左侧 ghost ← 右侧物理端 (j = j_total, j_total-1, ...)
    # 右侧 ghost ← 左侧物理端 (j = 1, 2, ...)
    for i in range(1, cc.i_total):
        # ── 左侧假想网格 ──
        for im in range(1, cc.IM + 1):
            gcell :cc.cell_class = cc.CellList[i][cc.j_total+im]
            gcell.copy_flow_fields(cc.CellList[i][cc.j_total - im + 1])
            gcell.formvars()

        # ── 右侧假想网格 ──
        for im in range(1, cc.IM + 1):
            gcell:cc.cell_class = cc.CellList[i][cc.j_total+cc.IM+im]
            gcell.copy_flow_fields(cc.CellList[i][im])
            gcell.formvars()
            cc.CellList[i].append(gcell)

def calc_convect():
    """邢程对流项"""
    # 处理壁面处的面上守恒量,face_tau,由首层tau网格和第一层壁面虚拟网格平均
    for j in range(1,cc.j_total+1):
        face : cc.face_class = cc.Facelist_tau[1][j]
        face.form_face_conserved_1stbounded(cc.CellList[1][j], cc.CellList[cc.i_total][j])

    # 处理远场处的面上守恒量,face_tau,由最外层tau网格和第一层远场虚拟网格平均
    for j in range(1,cc.j_total+1):
        face : cc.face_class = cc.Facelist_tau[cc.i_total][j]
        face.form_face_conserved_1stbounded(cc.CellList[cc.i_total-1][j], cc.CellList[cc.i_total+cc.IM][j])

    # 处理左周期边界处的面上守恒量,face_n
    for i in range(1,cc.i_total):
        face : cc.face_class = cc.FaceList_n[i][cc.j_total]
        face.form_face_conserved_1stbounded(cc.CellList[i][cc.j_total], cc.CellList[i][cc.j_total+1])
        
    # 处理右周期边界处的面上守恒量,face_n
    for i in range(1,cc.i_total):
        face : cc.face_class = cc.FaceList_n[i][1]
        face.form_face_conserved_1stbounded(cc.CellList[i][1], cc.CellList[i][cc.j_total+cc.IM+1])

    # 处理正常地方的面上守恒量,时刻提醒自己face_tau是(i_total,j_total),face_n是(i_total-1,j_total)
    for i in range(2,cc.i_total): # (1,j)和(i_total,j)的face通量已经处理好了
        for j in range(1,cc.j_total+1):
            face : cc.face_class = cc.Facelist_tau[i][j]
            face.form_face_conserved_1stbounded(cc.CellList[i][j], cc.CellList[i-1][j])

    for i in range(1,cc.i_total):
        for j in range(2,cc.j_total+1): # (i,1)和(i,j_total)的face通量已经处理好了
            face : cc.face_class = cc.FaceList_n[i][j]
            face.form_face_conserved_1stbounded(cc.CellList[i][j],cc.CellList[i][j-1])

    # 现在开始构造对流项:
    for i in range(1,cc.i_total+1):
        for j in range(1,cc.j_total+1):
            face : cc.face_class = cc.Facelist_tau[i][j]
            face.form_flux()

    for i in range(1,cc.i_total):
        for j in range(1,cc.j_total+1):
            face : cc.face_class = cc.FaceList_n[i][j]
            face.form_flux()

    # 邢程单元体的总通量,请注意以(i,j)变大为正.
    for i in range(1,cc.i_total):
        for j in range(1,cc.j_total+1):
            cell: cc.cell_class = cc.CellList[i][j]
            jp1 = j + 1 if j < cc.j_total else 1  
            cell.Fc = (cc.FaceList_n[i][jp1].Flux-cc.FaceList_n[i][j].Flux+
                    cc.Facelist_tau[i+1][j].Flux-cc.Facelist_tau[i][j].Flux)

def calc_grad():
    """Green-Gauss Based 梯度(和*Fluent*不同,这里没有加梯度限制器)"""
    for i in range(1,cc.i_total):
        for j in range(1,cc.j_total+1):
            cell : cc.cell_class = cc.CellList[i][j]

            # 处理各种特殊情况
            i_down = cc.i_total if i==1 else i-1
            i_up = cc.i_total+cc.IM if i==cc.i_total-1 else i+1
            j_left = cc.j_total+1 if j==1 else j-1
            j_right = cc.j_total+cc.IM+1 if j==cc.j_total else j+1
            jpright = j + 1 if j < cc.j_total else 1

            # 找到邻接网格
            cell_up : cc.cell_class = cc.CellList[i_up][j]
            cell_down : cc.cell_class = cc.CellList[i_down][j]
            cell_left : cc.cell_class = cc.CellList[i][j_left]
            cell_right : cc.cell_class = cc.CellList[i][j_right]

            # 找到邻接面
            face_up : cc.face_class = cc.Facelist_tau[i+1][j]
            face_down : cc.face_class = cc.Facelist_tau[i][j]
            face_left : cc.face_class = cc.FaceList_n[i][j]
            face_right : cc.face_class = cc.FaceList_n[i][jpright]

            # 使用一阶中心差分建构邻接面上的物理量
            face_up.form_face_vars_1stbounded(cell,cell_up)
            face_down.form_face_vars_1stbounded(cell,cell_down)
            face_left.form_face_vars_1stbounded(cell,cell_left)
            face_right.form_face_vars_1stbounded(cell,cell_right)

            # 建立当前网格的Green-Guass梯度
            green_gauss(cell,face_up,face_down,face_right,face_left)

    # 对壁面虚拟网格进行处理
    for j in range(1,cc.j_total+1):
        cell : cc.cell_class = cc.CellList[cc.i_total][j]
        cell.copy_grad(cc.CellList[1][j],ifT=False)

    # 对远场虚拟网格进行处理
    for j in range(1,cc.j_total+1):
        cell : cc.cell_class = cc.CellList[cc.i_total+cc.IM][j]
        cell.copy_grad(cc.CellList[cc.i_total-1][j],ifu=False,ifv=False,ifT=False,ifmiubl=False)

    # 对左周期处的虚拟网格进行处理
    for i in range(1,cc.i_total):
        cell : cc.cell_class = cc.CellList[i][cc.j_total+1]
        cell.copy_grad(cc.CellList[i][cc.j_total])

    # 对右周期处的虚拟网格进行处理
    for i in range(1,cc.i_total):
        cell : cc.cell_class = cc.CellList[i][cc.j_total+cc.IM+1]
        cell.copy_grad(cc.CellList[i][1])

def calc_diffusion():
    """邢程因湍流模型引起的扩散项"""
    # 计算壁面假想网格 S-A湍流模型各个参量
    for j in range(1,cc.j_total+1):
        cell : cc.cell_class = cc.CellList[cc.i_total][j]
        tb.Spalart_Allmaras(cell)

    # 计算远场假想网格 S-A湍流模型各个参量
    for j in range(1,cc.j_total+1):
        cell : cc.cell_class = cc.CellList[cc.i_total+cc.IM][j]
        tb.Spalart_Allmaras(cell)

    # 计算左周期处的虚拟网格 S-A湍流模型各个参量
    for i in range(1,cc.i_total):
        cell : cc.cell_class = cc.CellList[i][cc.j_total+1]
        tb.Spalart_Allmaras(cell)

    # 计算右周期处的虚拟网格 S-A湍流模型各个参量
    for i in range(1,cc.i_total):
        cell : cc.cell_class = cc.CellList[i][cc.j_total+cc.IM+1]
        tb.Spalart_Allmaras(cell)

    # 计算正常网格的 S-A湍流模型各个参量
    for i in range(1,cc.i_total):
        for j in range(1,cc.j_total+1):
            cell : cc.cell_class = cc.CellList[i][j]
            tb.Spalart_Allmaras(cell)

    # 处理facelist_tau的面上扩散项
    for i in range(1,cc.i_total+1):
        for j in range(1,cc.j_total+1):
            face : cc.face_class = cc.Facelist_tau[i][j]
            # 处理特殊情况
            i_down = cc.i_total if i==1 else i-1
            i_up = cc.i_total + cc.IM if i==cc.i_total else i
            # 找到邻接网格
            cell_up : cc.cell_class = cc.CellList[i_up][j]
            cell_down : cc.cell_class = cc.CellList[i_down][j]
            # 邢程扩散项
            tb.form_face_diffusion_1stbounded(face,cell_up, cell_down)

    # 处理facelist_n的面上扩散项
    for i in range(1,cc.i_total):
        for j in range(1,cc.j_total+1):
            face : cc.face_class = cc.FaceList_n[i][j]
            # 处理特殊情况
            j_left = cc.j_total+1 if j==1 else j-1
            j_right = cc.j_total+cc.IM+1 if j==cc.j_total else j
            # 找到邻接网格
            cell_left : cc.cell_class = cc.CellList[i][j_left]
            cell_right : cc.cell_class = cc.CellList[i][j_right]
            # 邢程扩散项
            tb.form_face_diffusion_1stbounded(face,cell_left, cell_right)

    # 邢程cell的湍流扩散项
    for i in range(1,cc.i_total):
        for j in range(1,cc.j_total+1):
            cell : cc.cell_class = cc.CellList[i][j]
            jp1 = j+1 if j<cc.j_total else 1
            face_up : cc.face_class = cc.Facelist_tau[i+1][j]
            face_down : cc.face_class= cc.Facelist_tau[i][j]
            face_left : cc.face_class = cc.FaceList_n[i][j]
            face_right : cc.face_class = cc.FaceList_n[i][jp1]
            # 邢程湍流扩散项
            cell.Fv = (face_up.DiffuTurb - face_down.DiffuTurb+
                    face_right.DiffuTurb - face_left.DiffuTurb)

def calc_source():
    """邢程因湍流模型引起的源项"""
    for i in range(1,cc.i_total):
        for j in range(1,cc.j_total+1):
            cell : cc.cell_class = cc.CellList[i][j]
            tb.form_source_term(cell)

def calc_dissipation():
    """邢程人工粘性项,JST不是一个稳定的格式,使用4阶粘性进行耗散"""
    # tau边界上的谱半径近似
    for i in range(1,cc.i_total+1):
        for j in range(1,cc.j_total+1):
            face : cc.face_class = cc.Facelist_tau[i][j]
            if i==1:hs.Spectral_Radius(face,cc.CellList[1][j],cc.CellList[1][j])
            elif i==cc.i_total:hs.Spectral_Radius(face,cc.CellList[cc.i_total-1][j],cc.CellList[cc.i_total-1][j])
            else: hs.Spectral_Radius(face,cc.CellList[i][j],cc.CellList[i-1][j])

    # n边界上的谱半径近似
    for i in range(1,cc.i_total):
        for j in range(1,cc.j_total+1):
            face : cc.face_class = cc.FaceList_n[i][j]
            j_left = cc.j_total if j==1 else j-1
            hs.Spectral_Radius(face,cc.CellList[i][j],cc.CellList[i][j_left])

    # tau边界上激波捕捉
    # ! 此处为了偷懒.把cc.IM==3当成默认的了.
    # 此外,使用激波捕捉要求整个网格的横纵单元数不得低于4,先处理边界处情况
    for j in range(1,cc.j_total+1):
        hs.shockwave_catcher((1,j),"tau",cc.CellList[cc.i_total+2][j],
                            cc.CellList[cc.i_total+1][j],cc.CellList[cc.i_total][j])
        hs.shockwave_catcher((2,j),"tau",cc.CellList[cc.i_total+1][j],
                            cc.CellList[cc.i_total][j],cc.CellList[1][j])
        hs.shockwave_catcher((3,j),"tau",cc.CellList[cc.i_total][j],
                            cc.CellList[1][j],cc.CellList[2][j])
        for i in range(4,cc.i_total+1):
            hs.shockwave_catcher((i,j),"tau",cc.CellList[i-3][j],
                                cc.CellList[i-2][j],cc.CellList[i-1][j])
        hs.shockwave_catcher((cc.i_total+1,j),"tau",cc.CellList[cc.i_total-2][j],
                            cc.CellList[cc.i_total-1][j],cc.CellList[cc.i_total+3][j])
        hs.shockwave_catcher((cc.i_total+2,j),"tau",cc.CellList[cc.i_total-1][j],
                            cc.CellList[cc.i_total+3][j],cc.CellList[cc.i_total+4][j])
        hs.shockwave_catcher((cc.i_total+3,j),"tau",cc.CellList[cc.i_total+3][j],
                            cc.CellList[cc.i_total+4][j],cc.CellList[cc.i_total+5][j])

    # n边界上的激波捕捉
    for i in range(1,cc.i_total):
        hs.shockwave_catcher((i,1),"n",cc.CellList[i][cc.j_total+3],
                            cc.CellList[i][cc.j_total+2],cc.CellList[i][cc.j_total+1])
        hs.shockwave_catcher((i,2),"n",cc.CellList[i][cc.j_total+2],
                            cc.CellList[i][cc.j_total+1],cc.CellList[i][1])
        hs.shockwave_catcher((i,3),"n",cc.CellList[i][cc.j_total+1],
                            cc.CellList[i][1],cc.CellList[i][2])
        for j in range(4,cc.j_total+2):
            hs.shockwave_catcher((i,j),"n",cc.CellList[i][j-3],
                                cc.CellList[i][j-2],cc.CellList[i][j-1])
        hs.shockwave_catcher((i,cc.j_total+2),"n",cc.CellList[i][cc.j_total-1],
                            cc.CellList[i][cc.j_total],cc.CellList[i][cc.j_total+4])
        hs.shockwave_catcher((i,cc.j_total+2),"n",cc.CellList[i][cc.j_total],
                            cc.CellList[i][cc.j_total+4],cc.CellList[i][cc.j_total+5])
        hs.shockwave_catcher((i,cc.j_total+2),"n",cc.CellList[i][cc.j_total+4],
                            cc.CellList[i][cc.j_total+5],cc.CellList[i][cc.j_total+6])

    # n边界上的阻尼系数
    for i in range(1,cc.i_total):
        for j in range(1,cc.j_total+1):
            hs.adaptive_dissipation(cc.FaceList_n[i][j],"n")

    # tau边界上的阻尼系数
    for i in range(1,cc.i_total+1):
        for j in range(1,cc.j_total+1):
            hs.adaptive_dissipation(cc.Facelist_tau[i][j],"tau")

    # n边界上人工粘性构建
    for i in range(1,cc.i_total):
        for j in range(1,cc.j_total+1):
            face: cc.face_class = cc.FaceList_n[i][j]
            if j==1:
                hs.form_JST_dissipation_term(face,cc.CellList[i][cc.j_total+1],
                                            cc.CellList[i][cc.j_total+2],
                                            cc.CellList[i][1],cc.CellList[i][2])
            elif j==2:
                hs.form_JST_dissipation_term(face,cc.CellList[i][1],
                                            cc.CellList[i][cc.j_total+1],
                                            cc.CellList[i][2],cc.CellList[i][3])
            elif j==cc.j_total:
                hs.form_JST_dissipation_term(face,cc.CellList[i][j-1],
                                            cc.CellList[i][j-2],cc.CellList[i][j],
                                            cc.CellList[i][cc.j_total+cc.IM+1])
            else:
                hs.form_JST_dissipation_term(face,cc.CellList[i][j-1],
                                            cc.CellList[i][j-2],cc.CellList[i][j],
                                            cc.CellList[i][j+1])

    # tau边界上的人工粘性构建
    for i in range(1,cc.i_total+1):
        for j in range(1,cc.j_total+1):
            face : cc.face_class = cc.Facelist_tau[i][j]
            if i == 1:
                hs.form_JST_dissipation_term(face,cc.CellList[cc.i_total][j],
                                            cc.CellList[cc.i_total+1][j],
                                            cc.CellList[1][j],cc.CellList[2][j])
            elif i == 2:
                hs.form_JST_dissipation_term(face,cc.CellList[1][j],
                                            cc.CellList[cc.i_total][j],cc.CellList[2][j],
                                            cc.CellList[3][j])
            elif i == cc.i_total-1:
                hs.form_JST_dissipation_term(face,cc.CellList[i-1][j],cc.CellList[i-2][j],
                                            cc.CellList[i][j],cc.CellList[cc.i_total+cc.IM][j])
            elif i == cc.i_total:
                hs.form_JST_dissipation_term(face,cc.CellList[i-1][j],cc.CellList[i-2][j],
                                            cc.CellList[cc.i_total+cc.IM][j],
                                            cc.CellList[cc.i_total+cc.IM+1][j])
            else:
                hs.form_JST_dissipation_term(face,cc.CellList[i-1][j],cc.CellList[i-2][j],
                                            cc.CellList[i][j],cc.CellList[i+1][j])

    # 邢程所有单元的人工粘性
    for i in range(1,cc.i_total):
        for j in range(1,cc.j_total+1):
            jp1 = j+1 if j<cc.j_total else 1
            cell : cc.cell_class = cc.CellList[i][j]
            face_up : cc.face_class = cc.Facelist_tau[i+1][j]
            face_down : cc.face_class = cc.Facelist_tau[i][j]
            face_left : cc.face_class = cc.FaceList_n[i][j]
            face_right : cc.face_class = cc.FaceList_n[i][jp1]
            cell.Fd = (face_up.Dissipation + face_right.Dissipation -
                    face_down.Dissipation - face_left.Dissipation)

def form_vars():
    """还原基本物理量"""
    for i in range(1,cc.i_total):
        for j in range(1,cc.j_total+1):
            cell : cc.cell_class = cc.CellList[i][j]
            cell.form_physic_vars()

def calc_residual():
    """计算残差,基于密度"""
    residual = 0
    for i in range(1,cc.i_total):
        for j in range(1,cc.j_total+1):
            cell : cc.cell_class = cc.CellList[i][j]
            residual += (cell.rho-cc.density_table[i][j])**2
    return math.sqrt(residual/(cc.i_total-1)/cc.j_total)