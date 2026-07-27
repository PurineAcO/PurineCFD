import classconfig as cc
import numpy as np

def ifsupersonic():
    """判断是否超越了声速"""
    return True if np.linalg.norm(np.array([cc.ull,cc.vll]))>= cc.cll else False

def form_vars(face:cc.face_class,c,vel_n,vel_tau,edgelength):
    """邢程各个物理亮"""
    face.T = c**2/cc.R/cc.gamma
    face.rho = cc.rholl * (face.T/cc.T) ** (1/(cc.gamma-1))
    face.p = face.rho * cc.R * face.T
    face.u = (vel_tau * face.ny + vel_n *face.nx)/edgelength
    face.v = (-vel_tau * face.nx + vel_n * face.ny)/edgelength
    face.E = face.p/(face.rho*(cc.gamma-1)) + 0.5*(face.u**2+face.v**2)

def turbence_nu_calc(j):
    """计算湍流变量的插值解"""
    LS = [0,0,0,0]
    face : cc.face_class = cc.Facelist_tau[cc.i_total][j]
    cell_in :list[cc.cell_class] = [cc.CellList[cc.i_total-1][j],
                                    cc.CellList[cc.i_total-2][j], 
                                    cc.CellList[cc.i_total-3][j]]
    for delta in range(1,4):
        LS[delta]=np.linalg.norm(np.array([cell_in[delta-1].x-face.mx,
                                           cell_in[delta-1].y-face.my]))
    # BUGFIX: 三点 Lagrange 外插的分母在两单元中心到面距离相等(退化网格)时为 0.
    #         此时退化为最近单元的一阶外插.
    d12, d13, d23 = LS[1]-LS[2], LS[1]-LS[3], LS[2]-LS[3]
    if abs(d12) < 1e-14 or abs(d13) < 1e-14 or abs(d23) < 1e-14:
        return max(cell_in[0].miubl, 1e-10)

    prembl = (LS[2]*LS[3]/(d12*d13)*cell_in[0].miubl +
            LS[1]*LS[3]/(-d12*d23)*cell_in[1].miubl +
            LS[1]*LS[2]/(d23*(-d13))*cell_in[2].miubl)

    return prembl if prembl>1e-10 else 1e-10

def riemann(j):
    """依据黎曼不变量求解"""
    face : cc.face_class = cc.Facelist_tau[cc.i_total][j]
    cell_in : cc.cell_class = cc.CellList[cc.i_total-1][j]
    edge_length = np.linalg.norm(np.array([face.nx,face.ny]))
    vel_ll = np.array([cc.ull,cc.vll])
    vel_in = np.array([cell_in.u,cell_in.v])
    tauer = np.array([face.nx,face.ny])/edge_length
    ner = np.array([-face.ny,face.nx])/edge_length
    vel_tau_ll = np.linalg.det(np.vstack([vel_ll,tauer])) # 来流切向速度
    vel_n_ll = np.linalg.det(np.vstack([vel_ll,ner]))     # 来流法向速度
    vel_tau_in = np.linalg.det(np.vstack([vel_in,tauer])) # 内部切向速度
    vel_n_in = np.linalg.det(np.vstack([vel_in,ner]))     # 内部法向速度

    if not ifsupersonic():
        R_in = vel_n_in + 2*cell_in.c/(cc.gamma-1)
        R_ll = vel_n_ll - 2*cc.cll/(cc.gamma-1)
        vel_n_face = 1/2*(R_in+R_ll)
        c_face = (cc.gamma-1)/4*(R_in-R_ll)
        if vel_n_face <= 0:
            form_vars(face,c_face,vel_n_face,vel_tau_ll,edge_length)
            # BUGFIX: 入流处原先置 ν̃ = 0.S-A 的远场边界应给定来流工作变量,
            #         置零会让湍流粘度被持续冲刷掉,整个流场退化为层流.
            face.miubl = cc.miublll
        else:
            form_vars(face,c_face,vel_n_face,vel_tau_in,edge_length)
            face.miubl = turbence_nu_calc(j)
    else:
        if vel_n_ll<=0:
            form_vars(face,cc.cll,vel_n_ll,vel_tau_ll,edge_length)
            # BUGFIX: 入流处原先置 ν̃ = 0.S-A 的远场边界应给定来流工作变量,
            #         置零会让湍流粘度被持续冲刷掉,整个流场退化为层流.
            face.miubl = cc.miublll
        else:
            form_vars(face,cell_in.c,vel_n_in,vel_tau_in,edge_length)
            face.miubl = turbence_nu_calc(j)