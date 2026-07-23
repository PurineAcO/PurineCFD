import numpy as np
import json
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
T0    = _cfg['physics']['T0']
Ts    = _cfg['physics']['Ts']
mu0   = _cfg['physics']['mu0']
P0    = _cfg['physics']['P0']
c0    = _cfg['physics']['c0']
cv    = R/(gamma-1)
cp    = gamma*cv

# 湍流模型(S-A)
Cv1 = _cfg['spalart_allmaras']['Cv1']  # 阻尼常数I,一般取值为7.1
Pr = _cfg['spalart_allmaras']['Pr']    # 普朗特数,一般取值为0.71
Prt = _cfg['spalart_allmaras']['Prt']  # 湍流普朗特数,一般取值为0.9
sigma = _cfg['spalart_allmaras']['sigma']  # 湍流模型参数σ的倒数,一般取值为1.5
Cb2 = _cfg['spalart_allmaras']['Cb2']  # 湍流模型参数Cb2,一般取值为0.622

# 模拟状态
AOA   = _cfg['simulation']['AOA'] 
Ma    = _cfg['simulation']['Ma']

# 求解器设置
CFL   = _cfg['simulation']['CFL']
IM    = _cfg['simulation']['IM']    # ghost cell layers.

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
        self.Fc = np.zeros(6) # cell flux variables
        self.Tgrad = np.zeros(3) # temperature gradient
        self.ugrad = np.zeros(3) # velocity gradient
        self.vgrad = np.zeros(3) # velocity gradient
        self.miublgrad = np.zeros(3) # turbulent viscosity gradient
        self.DiffuTurb = np.zeros((6,2)) # turbulent diffusion matrix
        self.Fv = np.zeros(6) # turbulent diffusion term 

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

    def green_gauss(self,face1:face_class,face2:face_class,
                    face3:face_class,face4:face_class):
        """基于Green-Guass的梯度构建"""
        u_vec = np.array([face1.u,face2.u,face3.u,face4.u])
        v_vec = np.array([face1.v,face2.v,face3.v,face4.v])
        miubl_vec = np.array([face1.miubl,face2.miubl,face3.miubl,face4.miubl])
        T_vec = np.array([face1.T,face2.T,face3.T,face4.T])
        nx_vec = np.array([face1.nx,-face2.nx,face3.nx,-face4.nx])
        ny_vec = np.array([face1.ny,-face2.ny,face3.ny,-face4.ny])
        self.ugrad = np.array([0,np.dot(u_vec,nx_vec),np.dot(u_vec,ny_vec)])/self.vol
        self.vgrad = np.array([0,np.dot(v_vec,nx_vec),np.dot(v_vec,ny_vec)])/self.vol
        self.miublgrad = np.array([0,np.dot(miubl_vec,nx_vec),np.dot(miubl_vec,ny_vec)])/self.vol
        self.Tgrad = np.array([0,np.dot(T_vec,nx_vec),np.dot(T_vec,ny_vec)])/self.vol

    def copy_grad(self,src:cell_class,ifu=True,ifv=True,ifT=True,ifmiubl=True):
        """将 `src` 的梯度复制到 `self`, 可选择复制 ugrad, vgrad, Tgrad, miublgrad"""
        self.ugrad = src.ugrad if ifu else np.zeros(3)
        self.vgrad = src.vgrad if ifv else np.zeros(3)
        self.Tgrad = src.Tgrad if ifT else np.zeros(3)
        self.miublgrad = src.miublgrad if ifmiubl else np.zeros(3)

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

    def form_face_conserved_1stbounded(self,cell_1:cell_class, cell_2:cell_class):
        """根据相邻单元的守恒量计算面上的守恒量*U*.采用一阶中心差分"""
        self.FU = 0.5 * (cell_1.U + cell_2.U)

    def form_face_diffusion_1stbounded(self,cell_1:cell_class,cell_2:cell_class):
        """根据相邻单元的湍流扩散项计算面上的湍流扩散项`DiffuTurb`.采用一阶中心差分"""
        face_diff = (cell_1.DiffuTurb + cell_2.DiffuTurb) / 2.0  
        normal = np.array([self.nx, self.ny])
        self.DiffuTurb = face_diff @ normal   
    
    def form_face_vars_1stbounded(self,cell_1:cell_class, cell_2:cell_class):
        """根据相邻单元的守恒量计算面上的物理量*ϕ*(含*̃ν,u,v,T*).采用一阶中心差分"""
        self.u = (cell_1.u+cell_2.u) / 2
        self.v = (cell_1.v+cell_2.v) / 2
        self.miubl = (cell_1.miubl+cell_2.miubl) / 2
        self.T = (cell_1.T+cell_2.T) / 2
    
    def form_flux(self):
        """根据基本物理量计算通量项"""
        self.rho = self.FU[1]
        if self.rho <= 1e-15: print("rho is 0 at face",self.index); exit(6)
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

density_table = np.zeros((i_total+1,j_total+1))