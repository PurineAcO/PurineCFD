import classconfig as cc
import numpy as np
import math

def Spalart_Allmaras(cell:cc.cell_class):
    """计算Spalart-Allmaras湍流模型引起的各种东西"""

    mu = (cc.mu0 * (cell.T/cc.T0)**1.5
            * (cc.T0+cc.Ts)/(cell.T+cc.Ts))   # 计算分子粘度μ,基于Suthland公式
    cell.chi = cell.U[5]/mu                   # 计算χ,修正粘度比
    cell.fv1 = (cell.chi**3)/(cell.chi**3+cc.Cv1)  # 计算阻尼函数fv1
    mu_t = cell.U[5] * cell.fv1                    # 形成湍流粘度μt
    mu_eff = mu + mu_t                        # 计算有效粘度μeff
    lambda_eff = mu/cc.Pr + mu_t/cc.Prt       # 计算有效导热系数λeff
    tau_xx = mu_eff * (4/3*cell.ugrad[1]-2/3*cell.vgrad[2]) # 计算切应力τxx
    tau_yy = mu_eff * (4/3*cell.vgrad[2]-2/3*cell.ugrad[1]) # 计算切应力τyy
    tau_xy = mu_eff * (cell.ugrad[2]+cell.vgrad[1])         # 计算切应力τxy
    q_x = lambda_eff * cc.cp * cell.Tgrad[1]                # 计算热流q_x
    q_y = lambda_eff * cc.cp * cell.Tgrad[2]                # 计算热流q_y
    cell.DiffuTurb[1] = [0,0]   
    cell.DiffuTurb[2] = [tau_xx, tau_xy]   
    cell.DiffuTurb[3] = [tau_xy, tau_yy]   
    cell.DiffuTurb[4] = [cell.u*tau_xx + cell.v*tau_xy + q_x,
                        cell.u*tau_xy + cell.v*tau_yy + q_y]  
    cell.DiffuTurb[5] = [cc.sigma * (mu + cell.U[5]) * cell.miublgrad[1],
                        cc.sigma * (mu + cell.U[5]) * cell.miublgrad[2]] 
    
def form_face_diffusion_1stbounded(face:cc.face_class,cell_1:cc.cell_class,cell_2:cc.cell_class):
    """根据相邻单元的湍流扩散项计算面上的湍流扩散项`DiffuTurb`.采用一阶中心差分"""
    face_diff = (cell_1.DiffuTurb + cell_2.DiffuTurb) / 2.0  
    normal = np.array([face.nx, face.ny])
    face.DiffuTurb = face_diff @ normal   

def form_source_term(cell:cc.cell_class):
    """计算单元的湍流源项`S`"""
    # the first term
    ft2 = cc.Ct3 * math.exp(-cc.Ct4 * cell.chi**2)  # 生产项修正函数ft2
    fv2 = 1-cell.chi/(1+cell.chi*cell.fv1)          # 涡量修正函数fv2
    Omega = 1/2 * (cell.ugrad[2] - cell.vgrad[1])   # 计算涡量Omega
    S = cc.fv3 * math.sqrt(2) * abs(Omega)          # 计算涡量S
    Sbl = (S + cell.U[5]/(cell.U[1] * 
        cell.sad**2 * cc.kappa**2)*fv2)             # 计算修正涡量Sbl
    P = cc.Cb1 * (1-ft2) * Sbl * cell.U[5]          # 计算生成项P
    # the second term
    r = min(cell.U[5]/cell.U[1] /Sbl /(cc.kappa**2 * cell.sad**2),cc.rmax) # 无量纲sad
    g = r + cc.Cw2 * (r**6 - r)
    fw = g * ((1+cc.Cw3**6)/(g**6 + cc.Cw3**6))**(1/6) # 壁面阻尼函数fw
    D = ((cc.Cw1 * fw - cc.Cb1 /(cc.kappa**2) * ft2) * 
         cell.U[1] *(cell.U[5]/cell.U[1]/cell.sad)**2) # 计算破坏项D
    # the third term
    G = cc.Cb2 * cc.sigma * cell.U[1] *(np.linalg.norm(cell.miublgrad))**2 
    # form the final source term
    cell.S = np.array([0,0,0,0,0,P-D+G])* cell.vol