import numpy as np
import json
import os

# ── 从 JSON 加载物理参数与模拟设置 ──────────────────────────
_config_path = os.path.join(os.path.dirname(__file__), 'config.json')
with open(_config_path, 'r', encoding='utf-8') as _f:
    _cfg = json.load(_f)

# 物理常数
gamma = _cfg['physics']['gamma']
R     = _cfg['physics']['R']
T0    = _cfg['physics']['T0']
Ts    = _cfg['physics']['Ts']
mu0   = _cfg['physics']['mu0']
P0    = _cfg['physics']['P0']
c0    = _cfg['physics']['c0']

# 模拟状态
AOA   = _cfg['simulation']['AOA']
Ma    = _cfg['simulation']['Ma']

# 求解器设置
CFL   = _cfg['simulation']['CFL']
IM    = _cfg['simulation']['IM']    # ghost cell layers

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
        self.localdt = 0    # locally computed time step (per-cell)
        self.dt = 0         # actual time step used for advancement (= global min)
        self.U = np.zeros(6) # conservative variables
        self.U_former = np.zeros(6) # former conservative variables 

    def copy_flow_fields(self, src):
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
        """根据原始变量计算守恒量 U[1..5]."""
        self.U[1] = self.rho
        self.U[2] = self.rho * self.u
        self.U[3] = self.rho * self.v
        self.U[4] = self.rho * self.E
        self.U[5] = self.rho * self.miubl

class face_class:
    def __init__(self,index):
        self.index = index
        self.rho = 0       # density
        self.p = 0         # pressure
        self.u = 0         # x-component of velocity
        self.v = 0         # y-component of velocity
        self.miubl = 0     # turbulent viscosity
        self.E = 0         # total energy per unit mass
        self.nx = 0        # normal direction n
        self.ny = 0        # normal direction tau
        self.mx = 0        # middle point x
        self.my = 0        # middle point y
        self.FU = np.zeros(6) # face conservative variables
        self.Flux = np.zeros(6) # face flux variables

    def form_face_conserved(self,cell_1:cell_class, cell_2:cell_class):
        """根据相邻单元的守恒量计算面上的守恒量.采用一阶中心差分"""
        self.FU = 0.5 * (cell_1.U + cell_2.U)
    
    def get_vars(self):
        """拆分面上守恒量为基本物理量"""
        self.rho = self.FU[1]
        if self.rho == 0: print("rho is 0 at face",self.index)
        self.u = self.FU[2] / self.rho
        self.v = self.FU[3] / self.rho
        self.miubl = self.FU[5] / self.rho
        self.E = self.FU[4]/ self.rho
        self.p = (gamma-1)*(self.FU[4]-self.rho*(self.u**2+self.v**2)*0.5)

    def form_flux(self):
        """根据基本物理量计算通量项"""
        self.get_vars()
        normal_vel = self.nx * self.u + self.ny * self.v  # 法向传播速度因子
        self.Flux[1] = self.rho * normal_vel
        self.Flux[2] = self.rho * self.u * normal_vel + self.p * self.nx
        self.Flux[3] = self.rho * self.v * normal_vel + self.p * self.ny
        self.Flux[4] = (self.rho * self.E + self.p) * normal_vel
        self.Flux[5] = self.rho * self.miubl * normal_vel


# some temp variables

density_table = np.zeros((i_total+1,j_total+1))