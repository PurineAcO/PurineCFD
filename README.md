# PurineCFD
---

## 目录

- [项目概览](#项目概览)
- [1. 配置 — `config.json` / `classconfig.py`](#1-配置--configjson--classconfigpy)
  - [1.1 物理常数](#11-物理常数)
  - [1.2 模拟设置](#12-模拟设置)
  - [1.3 全局数据结构](#13-全局数据结构)
  - [1.4 数据类](#14-数据类)
- [2. 网格读取 — `meshreading.py`](#2-网格读取--meshreadingpy)
- [3. 几何计算 — `geometry.py`](#3-几何计算--geometrypy)
  - [3.1 `calc_cell_vol()`](#31-calc_cell_vol)
  - [3.2 `calc_cell_center()`](#32-calc_cell_center)
  - [3.3 `calc_face_direction_tau()`](#33-calc_face_direction_tau)
  - [3.4 `calc_face_direction_n()`](#34-calc_face_direction_n)
  - [3.5 `calc_most_near_walldistance()`](#35-calc_most_near_walldistance)
- [4. 流场初始化 — `initialize.py`](#4-流场初始化--initializepy)
- [5. 输出 — `output.py`](#5-输出--输出outputpy)
  - [5.1 `geometry_debug()`](#51-geometry_debug)
  - [5.2 `initialize_output()`](#52-initialize_output)
  - [5.3 `formvars_main_output()`](#53-formvars_main_output)
  - [5.4 `min_timestep_output()`](#54-min_timestep_output)
  - [5.5 `mesh_visualization()`](#55-mesh_visualization)
- [6. 求解辅助 — `solvesupple.py`](#6-求解辅助--solvesupplepy)
  - [6.1 `formvars()` / `formvars_main()`](#61-formvars--formvars_main)
  - [6.2 `min_timestep()`](#62-min_timestep)
  - [6.3 `IM_wall()`](#63-im_wall)
  - [6.4 `IM_far()`](#64-im_far)
  - [6.5 `IM_LR()`](#65-im_lr)
  - [6.6 `formIM()`](#66-formim)
- [7. `output.txt` 解读](#7-outputtxt-解读)

---

## 项目概览

二维有限体积法 CFD 求解器,O 型结构化网格,求解 Navier-Stokes 方程.

**调用链**(`main.py`):

```
main.py
│
├─ mr.read_mesh("fangdata.txt")
│
├─ geo.geometry_main("output.txt")
│   ├─ calc_cell_vol()
│   ├─ calc_cell_center()
│   ├─ calc_face_direction_tau()
│   ├─ calc_face_direction_n()
│   ├─ calc_most_near_walldistance()
│   └─ output.geometry_debug("output.txt")
│
├─ ini.initialization_main()
│   ├─ initialization(T0, AOA, Ma, P0)
│   └─ output.initialize_output("output.txt")
│
├─ ss.formvars_main()
│
├─ ss.min_timestep()
│
└─ ss.formIM()
    ├─ IM_wall()
    ├─ IM_far()
    └─ IM_LR()

```

---

## 1. 配置 — `config.json` / `classconfig.py`

### 1.1 物理常数

| 键 | 值 | 含义 |
|----|-----|------|
| `gamma` | 1.4 | 比热比 |
| `R` | 287.06 | 气体常数 J/(kg·K) |
| `T0` | 288.16 | 来流总温 K |
| `Ts` | 110 | Sutherland 常数 K |
| `mu0` | 1.7894×10⁻⁵ | 参考动力粘度 Pa·s |
| `P0` | 101325 | 来流总压 Pa |
| `c0` | 340.28 | 参考声速 m/s |

### 1.2 模拟设置

| 键 | 默认值 | 含义 |
|----|--------|------|
| `AOA` | 0 | 攻角 ° |
| `Ma` | 0.2 | 马赫数 |
| `CFL` | 0.8 | Courant 数 |
| `IM` | 3 | 假想网格层数 (ghost cell layers) |

### 1.3 全局数据结构

所有列表为 1-indexed(`[0]` 为空占位).

| 变量 | 维度 | 说明 |
|------|------|------|
| `NodeList` | `[i_total+1][j_total+1]` | 网格节点,i=径向层,j=周向 |
| `CellList` | `[i_total][j_total+1]` | 计算单元,i=1..i_total-1 |
| `Facelist_tau` | `[i_total+1][j_total+1]` | 周向边的面数据 |
| `FaceList_n` | `[j_total+1][i_total]` | 径向边的面数据 |
| `totaltime` | 标量 | 全局累加模拟时间 |
| `density_table` | `[i_total+1][j_total+1]` | 密度快照 (残差计算用) |

### 1.4 数据类

- **`node_class`**:`x, y`(节点坐标)
- **`cell_class`**:
  - `index`(单元格索引);
  - `x, y, vol`(中心+面积);`sad`(壁面距离);
  - `rho, p, T, u, v, E, H, c, ma`(密度,压力,温度,速度,能量,焓,声速,马赫数);
  - `miu, miubl`(粘度);
  - `localdt`(当地时间步长);`dt`(实际推进时间步);
  - `U[6], U_former[6]`(守恒量); 

  方法:
  - `copy_flow_fields(src)` — 复制 9 个流场字段; 
  - `formvars()` — 根据原始变量计算守恒量 U[1..5]
- **`face_class`**:`ni, nj`(法向量,模长=边长),`mx, my`(中点)

---

## 2. 网格读取 — `meshreading.py`

### `read_mesh(meshfile)`

1. 从首行解析沿半径方向的点数`i_total`,每一圈的点数 `j_total`,这样就会有`i_total-1`层,每层`j_total`个网格.
2. `np.loadtxt(skiprows=1)` 读入全部 (x, y)
3. 校验 `点数 == i_total × j_total`
4. 逐层逐点构造 `NodeList[i][j]`
5. **闭合检测**:若每层首尾节点坐标差 < 1e-12,弹掉尾节点,`j_total -= 1`

---

## 3. 几何计算 — `geometry.py`

### `geometry_main(debugoutput, ifrender=False, showwhat=(True, True, True))`

顺序调用下面 5 个子函数,然后输出至文件.`ifrender` 控制是否可视化,`showwhat` 三元组控制显示内容 (单元中心, 周向法向, 径向法向).

### 3.1 `calc_cell_vol()`

四边形对角线叉积求面积:

$$V = \frac{1}{2}\left | (P_{i+1,j+1}-P_{i,j}) \times (P_{i+1,j}-P_{i,j+1}) \right |$$

使用 2D 标量叉积 `x1*y2 - y1*x2`(兼容 NumPy 2.x).`j=j_total` 时 `j+1` 回绕到 `1`.

### 3.2 `calc_cell_center()`

四边形重心公式(基于格林公式的面积加权平均).

### 3.3 `calc_face_direction_tau()`

周向边法向量:旋转切向量 90°,指向**径向外侧**(模长=边长).`j=j_total` 时回绕.

### 3.4 `calc_face_direction_n()`

径向边法向量:指向**逆时针切向**(模长=边长).索引顺序为 `[j][i]`.

### 3.5 `calc_most_near_walldistance()`

最内层 `Facelist_tau[1]` 为壁面,在 `[j±window]` 局部窗口内搜最近面中点距离.窗口 `max(15, j_total//5)`.

---

## 4. 流场初始化 — `initialize.py`

### `initialization(T0, AOA, Ma, P0)`

对所有单元施加均匀来流:

1. $T = \frac{T_0}{1 + \frac{\gamma-1}{2}Ma^2}$
2. $p = P_0 (\frac{T}{T_0})^{\frac{\gamma}{\gamma-1}}$
3. $c = \sqrt{\gamma RT},\rho = \frac{p}{RT}$
4. $u = c \cdot Ma \cdot \cos\alpha,v = c \cdot Ma \cdot \sin\alpha$
5. $E = \frac{p}{\rho (\gamma-1)} + \frac{u^2+v^2}{2},H = E + \frac{p}{\rho}$
6. $\mu = \mu_0 (\frac{T}{T_0})^{1.5} \left (\frac{T_0+T_s}{T+T_s}\right )$
7. $\tilde{\nu} = 0.1\frac{\mu}{\rho}$

### `initialization_main()`

调用 `initialization()` → `ot.initialize_output()`.

---

## 5. 输出 — `output.py`

### 5.1 `geometry_debug()`

- **`geometry_debug(path)`**:覆盖写入,输出所有单元(index / vol / center / sad)、周向面、径向面的信息

### 5.2 `initialize_output()`

- **`initialize_output(path)`**:追加写入,输出 CellList[1][1] 的全部流场量作为示例

### 5.3 `formvars_main_output()`

- **`formvars_main_output(path)`**:追加写入,输出所有单元的守恒量 U[1..5]

### 5.4 `min_timestep_output()`

- **`min_timestep_output(path)`**:追加写入,输出全局最小时间步及各单元 localdt

### 5.5 `mesh_visualization()`

- **`mesh_visualization(savepath, show_centers, show_tau, show_n)`**:绘制 O 型网格图(蓝色网格线 + 深红单元中心 + 深青/深橙法向量箭头)

---

## 6. 求解辅助 — `solvesupple.py`

### 6.1 `formvars_main()`

遍历所有物理单元, 调用 `cell_class.formvars()` 完成原始变量 → 守恒量变换:

$$\boldsymbol{U} = (\rho ,\rho u,\rho v,\rho E,\rho \tilde{\nu})\$$

### 6.2 `min_timestep()`

基于谱半径近似计算各单元当地时间步长, 找出全局最小值,
所有单元统一使用该最小值推进, 并累加到 `totaltime`:

$$\Delta t_{ij} = \frac{\text{CFL} \cdot V_{ij}}{|uA+vB| + |uC+vD| + c_{ij} \cdot (\sqrt{A^2+B^2} + \sqrt{C^2+D^2})}$$

其中 $(A,B)$ 和 $(C,D)$ 为相邻两面法向的平均值, 周向 `j=j_total` 时回绕到 `1`.

### 6.3 `IM_wall()`

设置内壁面假想网格边界条件 (slip wall). 对每一层假想网格, 创建 `cell_class` 的同时直接填入边界值, 然后整行追加到 `CellList` 末尾.

- **标量** (`rho, p, T, E, H, c`): 从壁面 `CellList[1][j]` 复制
- **速度** (`u, v`): 取对应内层 `CellList[im][j]` 的相反数 (镜像反射)
- **马赫数**: 由假想网格自身的 $u,v,T$ 重新计算
- **湍流变量** ($\tilde{\nu}$,`miubl`): 取对应内层的相反数

### 6.4 `IM_far()`

设置压力远场假想网格边界条件. 对每一层 ghost cell, 创建 `cell_class` 的同时从最外层物理单元 (`i_total-1`) 直接复制远场值, 然后整行追加到 `CellList` 末尾. 所有 ghost 层取值相同. 复制字段: `rho, p, E, T, H, u, v, ma, miubl`.

### 6.5 `IM_LR()`

设置 O 型网格切割线两侧的周期边界假想网格 (j 方向). 对每一径向层 `i`, 追加 `2 × IM` 个 ghost cell 到该行末尾.

- **左侧 ghost** (层 `k=1..IM`): 拷贝自右侧物理端 `CellList[i][j_total - k + 1]`
- **右侧 ghost** (层 `k=1..IM`): 拷贝自左侧物理端 `CellList[i][k]`
- 复制字段: `rho, p, E, T, H, u, v, ma, miubl`

### 6.6 `formIM()`

统一入口, 按顺序调用三个边界条件函数构建全部假想网格:

1. `IM_wall()` — 内壁面 ghost (i 方向下边界)
2. `IM_far()`  — 压力远场 ghost (i 方向上边界)
3. `IM_LR()`  — 周向周期 ghost (j 方向左右边界)

---

## 7. `output.txt` 解读

以 `fangdata.txt`(10×12)为例,文件按 `geometry_debug` → `initialize_output` 顺序生成:

- **单元段**:各个单元的索引`index`、面积`vol`、中心坐标`center`和壁面距离`sad`
- **面法向量段**:周向边和径向边的索引`index`、法向量`ni, nj`和中点坐标`mx, my`
- **流场段**:所有单元的原始变量和粘度,以及 `CellList[1][1]` 的全部流场示例