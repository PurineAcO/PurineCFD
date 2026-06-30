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
- [6. 求解辅助 — `solvesupple.py`](#6-求解辅助--solvesupplepy)
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
└─ ss.min_timestep()

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

### 1.3 全局数据结构

所有列表为 1-indexed(`[0]` 为空占位).

| 变量 | 维度 | 说明 |
|------|------|------|
| `NodeList` | `[i_total+1][j_total+1]` | 网格节点,i=径向层,j=周向 |
| `CellList` | `[i_total][j_total+1]` | 计算单元,i=1..i_total-1 |
| `Facelist_tau` | `[i_total+1][j_total+1]` | 周向边的面数据 |
| `FaceList_n` | `[j_total+1][i_total]` | 径向边的面数据 |

### 1.4 数据类

- **`node_class`**:`x, y`(节点坐标)
- **`cell_class`**:`x, y, vol`(中心+面积);`rho, p, T, u, v, E, H, c, ma`(原始变量);`miu, miubl`(粘度);`sad`(壁面距离);`localdt`(当地时间步长);`U[6], U_former[6]`(守恒量)
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

### `geometry_main(debugoutput, ifrender, showwhat)`

顺序调用下面 5 个子函数,然后输出至文件.`ifrender` 控制是否可视化,`showwhat` 三元组控制显示内容.

### 3.1 `calc_cell_vol()`

四边形对角线叉积求面积:$V = 0.5 \cdot | (P_{i+1,j+1}-P_{i,j}) \times (P_{i+1,j}-P_{i,j+1}) |$.使用 2D 标量叉积 `x1*y2 - y1*x2`(兼容 NumPy 2.x).`j=j_total` 时 `j+1` 回绕到 `1`.

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

1. $T = T_0 / (1 + \frac{\gamma-1}{2}Ma^2)$
2. $p = P_0 (T/T_0)^{\gamma/(\gamma-1)}$
3. $c = \sqrt{\gamma RT},\rho = \frac{p}{RT}$
4. $u = c \cdot Ma \cdot \cos\alpha,v = c \cdot Ma \cdot \sin\alpha$
5. $E = p/[\rho(\gamma-1)] + (u^2+v^2)/2,H = E + \frac{p}{\rho}$
6. $\mu = \mu_0 (T/T_0)^{1.5} \frac{T_0+T_s}{T+T_s}$
7. $\mu_{bl} = 0.1\frac{\mu}{\rho}$

### `initialization_main()`

调用 `initialization()` → `ot.initialize_output()`.

---

## 5. 输出 — `output.py`

- **`geometry_debug(path)`**:覆盖写入,输出所有单元(index / vol / center / sad)、周向面、径向面的信息
- **`initialize_output(path)`**:追加写入,输出 CellList[1][1] 的全部流场量作为示例
- **`mesh_visualization(savepath, show_centers, show_tau, show_n)`**:绘制 O 型网格图(蓝色网格线 + 深红单元中心 + 深青/深橙法向量箭头)

---

## 6. 求解辅助 — `solvesupple.py`

- **`formvars(cell)`**:原始变量 → 守恒量 `U[1..5]`
  - `U[1]=ρ`, `U[2]=ρu`, `U[3]=ρv`, `U[4]=ρE`, `U[5]=ρ·μ_bl`
- **`formvars_main()`**:遍历所有 CellList,逐单元调用 `formvars()`
- **`min_timestep()`**:计算各单元当地时间步长,返回全局最小值
  - 基于谱半径近似: $\Delta t = \frac{\text{CFL} \cdot V}{ \sum (|\mathbf{u}\cdot\mathbf{n}| + c|\mathbf{n}|)}$
  - 面法向取相邻两面的平均值,周向 `j=j_total` 时回绕到 `1`
- **`res_and_ustep()`**:保存密度到 `density_table`,备份 `U → U_former`(暂注释)

---

## 7. `output.txt` 解读

以 `fangdata.txt`(10×12)为例,文件按 `geometry_debug` → `initialize_output` 顺序生成:

- **单元段**:各个单元的索引`index`、面积`vol`、中心坐标`center`和壁面距离`sad`
- **面法向量段**:周向边和径向边的索引`index`、法向量`ni, nj`和中点坐标`mx, my`
- **流场段**:所有单元的原始变量和粘度,以及 `CellList[1][1]` 的全部流场示例