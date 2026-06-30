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
CFL   = _cfg['simulation']['CFL']

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
        self.index = index  # cell index (i,j)
        self.x = 0          # cell center x
        self.y = 0          # cell center y   
        self.vol = 0        # cell volume(for 2D,it iterally means area)
        self.sad = 0        # cell to wall distance
        self.rho = 0
        self.p = 0
        self.T = 0
        self.u = 0
        self.v = 0
        self.E = 0
        self.H = 0
        self.c = 0
        self.ma = 0
        self.miu = 0
        self.miubl = 0
        self.localdt = 0    # locally computed time step (per-cell)
        self.dt = 0         # actual time step used for advancement (= global min)
        self.U = [0,0,0,0,0,0] # conservative variables
        self.U_former = [0,0,0,0,0,0] # former conservative variables

class face_class:
    def __init__(self,index):
        self.index = index
        self.ni = 0        # normal direction n
        self.nj = 0        # normal direction tau
        self.mx = 0        # middle point x
        self.my = 0        # middle point y

# some temp variables

density_table = np.zeros((i_total+1,j_total+1))