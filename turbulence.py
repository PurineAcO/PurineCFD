import classconfig as cc
import numpy as np

def Spalart_Allmaras(cell:cc.cell_class):
    """计算Spalart-Allmaras湍流模型引起的各种东西"""

    mu = (cc.mu0 * (cell.T/cc.T0)**1.5
            * (cc.T0+cc.Ts)/(cell.T+cc.Ts)) # 计算分子粘度μ,基于Suthland公式
    chi = cell.U[5]/mu                      # 计算χ,修正粘度比
    fv1 = (chi**3)/(chi**3+cc.Cv1)          # 计算阻尼函数fv1
    mu_t = cell.U[5] * fv1                  # 形成湍流粘度μt
    mu_eff = mu + mu_t                      # 计算有效粘度μeff
    lambda_eff = mu/cc.Pr + mu_t/cc.Prt     # 计算有效导热系数λeff
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
    cell.DiffuTurb[5] = [cc.sigma * (mu + cell.U[5] + cc.Cb2*cell.U[5]) * cell.miublgrad[1],
                        cc.sigma * (mu + cell.U[5] + cc.Cb2*cell.U[5]) * cell.miublgrad[2]] 