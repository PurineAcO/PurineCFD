from __future__ import annotations   # BUGFIX: cell_class 在自身类体内被前向引用

import numpy as np
import json
import math
import os

# ── 从 JSON 加载物理参数与模拟设置 ──────────────────────────
_config_path = os.path.join(os.path.dirname(__file__), 'config.json')
try:
    with open(_config_path, 'r', encoding='utf-8') as _f:
        _cfg = json.load(_f)
except FileNotFoundError:
    print(f"错误：找不到配置文件 {_config_path}")
    print("请确保 config.json 与 classconfig.py 在同一目录下。")
    exit(1)
except json.JSONDecodeError as e:
    print(f"错误：配置文件 {_config_path} 格式错误")
    print(f"详细信息：{e}")
    exit(1)

# 物理常数(理想气体、Suthland)
gamma = _cfg['physics']['gamma']
R     = _cfg['physics']['R']
T0    = _cfg['physics']['T0']          # Sutherland 参考温度
Ts    = _cfg['physics']['Ts']          # Sutherland 常数温度
mu0   = _cfg['physics']['mu0']
P0    = _cfg['physics']['P0']
c0    = _cfg['physics']['c0']
cv    = R/(gamma-1)
cp    = gamma*cv

# 湍流模型(S-A)
Cv1 = _cfg['spalart_allmaras']['Cv1']  # 阻尼常数I,一般取值为7.1
Pr = _cfg['spalart_allmaras']['Pr']    # 普朗特数,一般取值为0.71
Prt = _cfg['spalart_allmaras']['Prt']  # 湍流普朗特数,一般取值为0.9
Cv1_cubed = Cv1 ** 3                   # fv1 = χ³/(χ³+Cv1³),预先缓存 Cv1³
sigma = _cfg['spalart_allmaras']['sigma']  # 湍流模型参数σ的倒数,一般取值为1.5
Cb1 = _cfg['spalart_allmaras']['Cb1']  # 湍流模型参数Cb1,一般取值为0.1355
Cb2 = _cfg['spalart_allmaras']['Cb2']  # 湍流模型参数Cb2,一般取值为0.622
Cw2 = _cfg['spalart_allmaras']['Cw2']  # 湍流模型参数Cw2,一般取值为0.3
Cw3 = _cfg['spalart_allmaras']['Cw3']  # 湍流模型参数Cw3,一般取值为2.0
Ct3 = _cfg['spalart_allmaras']['Ct3']  # 湍流模型参数Ct3,一般取值为1.2
Ct4 = _cfg['spalart_allmaras']['Ct4']  # 湍流模型参数Ct4,一般取值为0.5
fv3 = _cfg['spalart_allmaras']['fv3']  # 湍流模型参数fv3,一般取值为1.0
kappa = _cfg['spalart_allmaras']['kappa']  # 湍流模型参数kappa,一般取值为0.41(也有取为0.4187的)
rmax = _cfg['spalart_allmaras']['rmax']  # 湍流模型参数rmax,一般取值为10

Cw1 = Cb1/(kappa**2) + (1+Cb2)*sigma    # 湍流模型参数Cw1

# JST耗散项
k2 = _cfg['dissipation']['k2']  # JST二阶耗散项,一般取值为0.5
k4 = _cfg['dissipation']['k4']  # JST四阶耗散项,一般取值为0.0078125

# 模拟状态,标识ll的是次生变量.
AOA = _cfg['simulation']['AOA'] # 来流攻角
Ma = _cfg['simulation']['Ma']   # 来流马赫数
T = _cfg['simulation']['T']     # 来流静温
P = _cfg['simulation']['P']     # 来流静压
cll = math.sqrt(gamma* R *T)    # 来流声速
ull = cll*Ma*math.cos(math.radians(AOA)) # 来流x方向速度
vll = cll*Ma*math.sin(math.radians(AOA)) # 来流y方向速度
rholl = P/(R*T)                 # 来流密度
mull  = mu0 * (T/T0)**1.5 * (T0+Ts)/(T+Ts)  # 来流分子粘度(Sutherland)
# 来流湍流工作变量 ν̃∞.与 initialize.py 的场初始化保持一致(0.1·ν∞),
# BUGFIX: 原 boundary.riemann 在入流处强制 miubl=0,会使湍流模型在整个流场退化.
miublll = 0.1 * mull / rholl

# Tll = T/(1+(gamma-1)/2*Ma**2)    # 来流总温
# Pll = P*(Tll/T)**(gamma/(gamma-1))# 来流总压
# cll = math.sqrt(gamma*R*T)       # 来流声速
# ull = cll*Ma*math.cos(math.radians(AOA)) # 来流x方向速度
# vll = cll*Ma*math.sin(math.radians(AOA)) # 来流y方向速度
# rholl = Pll/(R*Tll)                       # 来流密度
# Ell = Pll/((gamma-1)*rholl) + (ull**2+vll**2)/2 # 来流能量
# Hll = Ell + Pll/rholl                           # 来流焓
# mull = mu0 * (Tll/T0)**1.5 * (T0+Ts)/(Tll+Ts)   # 来流粘度
# miublll = 0.1*mull/rholl                        # 来流动力粘度

# 求解器设置
CFL   = _cfg['simulation']['CFL']
IM    = _cfg['simulation']['IM']    # ghost cell layers.
RK    = (0,0.25,1/6,0.375,0.5,1)      # Runge-Kutta params (5 stages: RK[1..5])
RK_STAGES = 5                         # BUGFIX: 原求解器只推进 4 级,丢掉了 RK[5]=1
_solver = _cfg.get('solver', {})
iteration = _solver.get('iteration', 10000)   # max iteration
targetres = _solver.get('targetres', 1e-10)   # target residual

# area for the global variables
i_total = 0
j_total = 0
meshcnt = 0
NodeList = [[]]
CellList = [[]]
FaceList_n = [[]]
Facelist_tau = [[]]

# global accumulated simulation time
totaltime = 0.0

# output file
outputfile = "output.txt"


def reset_state():
    """清空全部模块级全局状态,使同一进程内可以连续跑多个算例(测试需要)."""
    global i_total, j_total, meshcnt, totaltime
    global NodeList, CellList, FaceList_n, Facelist_tau
    global shockwave_tau, shockwave_n, density_table
    i_total = 0
    j_total = 0
    meshcnt = 0
    totaltime = 0.0
    NodeList = [[]]
    CellList = [[]]
    FaceList_n = [[]]
    Facelist_tau = [[]]
    shockwave_tau = None
    shockwave_n = None
    density_table = None

#area for the class definition
class node_class:
    def __init__(self,index):
        self.index = index
        self.x = 0       # node x
        self.y = 0       # node y

class cell_class:
    def __init__(self,index):
        self.index  = index  # cell index (i,j)
        self.x = 0          # cell center x
        self.y = 0          # cell center y
        self.vol = 0        # cell volume(for 2D,it iterally means area)
        self.sad = 0        # cell to wall distance
        self.rho = 0        # density
        self.p = 0          # pressure
        self.T = 0          # temperature
        self.u = 0          # x-component of velocity
        self.v = 0          # y-component of velocity
        self.E = 0          # total energy per unit mass
        self.H = 0          # specific enthalpy per unit mass
        self.c = 0          # speed of sound per unit mass
        self.ma = 0         # Mach number
        self.miu = 0        # dynamic viscosity
        self.miubl = 0      # turbulent viscosity
        self.chi = 0        # turbulent viscosity ratio
        self.fv1 = 0        # damping function fv1
        self.localdt = 0    # locally computed time step (per-cell)
        self.dt = 0         # actual time step used for advancement (= global min)
        self.U = np.zeros(6) # conservative variables
        self.U_former = np.zeros(6) # former conservative variables
        self.Fc = np.zeros(6) # cell flux variables
        self.Tgrad = np.zeros(3) # temperature gradient
        self.ugrad = np.zeros(3) # velocity gradient
        self.vgrad = np.zeros(3) # velocity gradient
        self.miublgrad = np.zeros(3) # turbulent viscosity gradient
        self.DiffuTurb = np.zeros((6,2)) # turbulent diffusion matrix
        self.Fv = np.zeros(6) # turbulent diffusion term
        self.S = np.zeros(6)  # turbulent source term
        self.Fd = np.zeros(6) # JST dissipation term
        self.URK = np.zeros(6) # Runge-Kutta conservative term

    def copy_flow_fields(self, src:cell_class):
        """将 `src` 的流场量复制到 `self`, 不覆盖几何属性 (index/x/y/vol/sad)."""
        self.rho   = src.rho
        self.p     = src.p
        self.E     = src.E
        self.T     = src.T
        self.H     = src.H
        self.u     = src.u
        self.v     = src.v
        self.ma    = src.ma
        self.miubl = src.miubl

    def formvars(self):
        """根据原始变量计算守恒量 U[1~5]."""
        self.U[1] = self.rho
        self.U[2] = self.rho * self.u
        self.U[3] = self.rho * self.v
        self.U[4] = self.rho * self.E
        self.U[5] = self.rho * self.miubl

    def form_physic_vars(self):
        """根据守恒量还原物理量"""
        # BUGFIX: 原实现引用 self.FU —— cell_class 根本没有该属性(那是 face_class 的),
        #         任何一次 RK 推进后调用都会 AttributeError.正确的守恒量是 self.U.
        self.rho = self.U[1]
        if self.rho <= 1e-15:
            raise FloatingPointError(f"non-positive density {self.rho:.3e} at cell {self.index}")
        self.u = self.U[2] / self.rho
        self.v = self.U[3] / self.rho
        self.miubl = self.U[5] / self.rho
        self.E = self.U[4] / self.rho
        self.p = (gamma-1)*(self.U[4]-self.rho*(self.u**2+self.v**2)*0.5)
        if self.p <= 1e-15:
            raise FloatingPointError(f"non-positive pressure {self.p:.3e} at cell {self.index}")
        self.H = self.E+self.p/self.rho
        self.T = self.p/(R*self.rho)
        self.c = math.sqrt(R*gamma*self.T)
        self.ma = math.sqrt(self.u**2+self.v**2)/self.c

    def copy_grad(self,src:cell_class,ifu=True,ifv=True,ifT=True,ifmiubl=True):
        """将 `src` 的梯度复制到 `self`, 可选择复制 ugrad, vgrad, Tgrad, miublgrad"""
        # BUGFIX: 使用 .copy() 而非直接绑定,避免 ghost 与内部单元共享同一 ndarray
        self.ugrad = src.ugrad.copy() if ifu else np.zeros(3)
        self.vgrad = src.vgrad.copy() if ifv else np.zeros(3)
        self.Tgrad = src.Tgrad.copy() if ifT else np.zeros(3)
        self.miublgrad = src.miublgrad.copy() if ifmiubl else np.zeros(3)

class face_class:
    def __init__(self,index):
        self.index = index
        self.rho = 0       # density
        self.p = 0         # pressure
        self.u = 0         # x-component of velocity
        self.v = 0         # y-component of velocity
        self.T = 0         # temperature
        self.miubl = 0     # turbulent viscosity
        self.E = 0         # total energy per unit mass
        self.nx = 0        # normal direction n
        self.ny = 0        # normal direction tau
        self.mx = 0        # middle point x
        self.my = 0        # middle point y
        self.FU = np.zeros(6) # face conservative variables
        self.Flux = np.zeros(6) # face flux variables
        self.DiffuTurb = np.zeros(6) # face turbulent diffusion term
        self.miublgrad = np.zeros(3) # turbulent viscosity gradient
        self.lambda_f = 0            # face spectral radius
        self.shockwave = 0           # face shockwave indicator
        self.epsilon = np.zeros(3)   # adaptive dissipation coefficient
        self.Dissipation = np.zeros(6) # JST dissipation term

    def form_face_conserved_1stbounded(self,cell_1:cell_class, cell_2:cell_class):
        """根据相邻单元的守恒量计算面上的守恒量*U*.采用一阶中心差分"""
        self.FU = 0.5 * (cell_1.U + cell_2.U)
    
    def form_face_vars_1stbounded(self,cell_1:cell_class, cell_2:cell_class):
        """根据相邻单元的守恒量计算面上的物理量*ϕ*(含*̃ν,u,v,T*).采用一阶中心差分"""
        self.u = (cell_1.u+cell_2.u) / 2
        self.v = (cell_1.v+cell_2.v) / 2
        self.miubl = (cell_1.miubl+cell_2.miubl) / 2
        self.T = (cell_1.T+cell_2.T) / 2
    
    def form_flux(self):
        """根据基本物理量计算通量项"""
        self.rho = self.FU[1]
        if self.rho <= 1e-15:
            raise FloatingPointError(f"non-positive density {self.rho:.3e} at face {self.index}")
        self.u = self.FU[2] / self.rho
        self.v = self.FU[3] / self.rho
        self.miubl = self.FU[5] / self.rho
        self.E = self.FU[4]/ self.rho
        self.p = (gamma-1)*(self.FU[4]-self.rho*(self.u**2+self.v**2)*0.5)
        normal_vel = self.nx * self.u + self.ny * self.v  # 法向传播速度因子
        self.Flux[1] = self.rho * normal_vel
        self.Flux[2] = self.rho * self.u * normal_vel + self.p * self.nx
        self.Flux[3] = self.rho * self.v * normal_vel + self.p * self.ny
        self.Flux[4] = (self.rho * self.E + self.p) * normal_vel
        self.Flux[5] = self.rho * self.miubl * normal_vel


# some temp variables
shockwave_tau = ...
shockwave_n = ...
density_table = ...