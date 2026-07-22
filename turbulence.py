import classconfig as cc

def Spalart_Allmaras(cell:cc.cell_class):
    """计算Spalart-Allmaras湍流模型引起的各种东西"""
    mu = (cc.mu0 * (cell.T/cc.T0)**1.5
            * (cc.T0+cc.Ts)/(cell.T+cc.Ts)) # 计算分子粘度μ,基于Suthland公式
    chi = cell.U[5]/mu                      # 计算χ,修正粘度比
    fv1 = (chi**3)/(chi**3+cc.Cv1)          # 计算阻尼函数fv1
    mu_t = mu * fv1                         # 形成湍流粘度μt
    mu_eff = mu + mu_t                      # 计算有效粘度μeff
    lambda_eff = mu/cc.Pr + mu_t/cc.Prt     # 计算有效导热系数λeff

    return mu_eff, lambda_eff

def Spalart_Allmaras_Diffusion(cell:cc.cell_class):
    """计算Spalart-Allmaras湍流模型的扩散项"""
    mu_eff,lambda_eff = Spalart_Allmaras(cell)
    tau_xx = mu_eff * (4/3*cell.ugrad[1]-2/3*cell.vgrad[2])
    tau_yy = mu_eff * (4/3*cell.vgrad[2]-2/3*cell.ugrad[1])
    tau_xy = mu_eff * (cell.ugrad[2]+cell.vgrad[1])
    ...
