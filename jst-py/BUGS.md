# 缺陷清单 (Python 基线)

原始代码 **无法运行**：在 `import` 阶段即抛 `NameError`。下表是逐个定位并修复的全部问题，
按严重程度分组。每处修复在源码中都留有 `BUGFIX:` 注释。

## A. 致命缺陷（程序无法运行 / 结果完全错误）

| # | 位置 | 问题 | 后果 |
|---|------|------|------|
| A1 | `classconfig.py:137` | `def copy_flow_fields(self, src:cell_class)` 在 `cell_class` 类体内前向引用自身 | 导入即 `NameError`，**项目从未成功运行过**。加 `from __future__ import annotations` |
| A2 | `classconfig.py:159-169` | `cell_class.form_physic_vars()` 通篇引用 `self.FU`，但 `FU` 是 `face_class` 的属性，`cell_class` 只有 `U` | 第一次 RK 推进后必 `AttributeError` |
| A3 | `main.py:32` | 输出写 `cell.Dent`，该属性不存在（应为 `cell.rho`） | 写结果时 `AttributeError` |
| A4 | `solvemain.py:5-6` | `for i in (1, cc.i_total)` / `for j in (1, cc.j_total+1)` —— 遍历的是长度为 2 的**元组**而非 `range` | 只处理 4 个单元；且 `density_table[i][j_total+1]` 越界（数组第二维仅 `j_total+1`） |
| A5 | `solvesupple.py` | **`calc_grad()` 从未被调用** | `ugrad/vgrad/Tgrad/miublgrad` 恒为 0 → 粘性扩散项与 S-A 源项全部失效，N-S 方程静默退化为 Euler 方程 |
| A6 | `solvemain.py:14` | `if step==1: ss.imagination_mesh_create()` 位于 RK **级**循环内，`step==1` 对全部 5 级都成立 | ghost 行被重复 `append` 多次，`CellList` 结构被破坏 |
| A7 | `solvesupple.py:172` | `imagination_mesh_update()` 末尾多出一句 `cc.CellList[i].append(gcell)` | 每个 RK 级都让该行增长 3 个元素 —— 内存泄漏 + 索引错位 |

## B. 数值格式错误（能跑，但解不对）

| # | 位置 | 问题 | 修复 |
|---|------|------|------|
| B1 | `solvemain.py:11` | `for k in range(1,5)` 只推进 4 级，丢掉了末级 `RK[5]=1` | 改为 5 级。显式 RK 的末级系数必须为 1，否则格式不相容（不再是时间一致的推进） |
| B2 | `turbulence.py:11` | `fv1 = χ³/(χ³ + Cv1)`，漏了立方 | S-A 定义为 `χ³/(χ³ + Cv1³)`；分母 7.1 vs 357.9，近壁阻尼被严重削弱 |
| B3 | `turbulence.py:39-40` | `Ω = ½(∂u/∂y − ∂v/∂x)`、`S = √2·|Ω|` | 二维涡量模应为 `S = √(2ΩᵢⱼΩᵢⱼ) = |∂v/∂x − ∂u/∂y|`；原式小 √2 倍且符号相反 |
| B4 | `solvesupple.py:194` | 周期面 `FaceList_n[i][1]` 取 `CellList[i][j_total+IM+1]`（右侧 ghost，是单元 1 **自身**的副本） | 等于把单元 1 与它自己平均，**周期边界完全失效**。应取左侧 ghost `CellList[i][j_total+1]` |
| B5 | `solvesupple.py:327` | `calc_diffusion` 中 `j_right = j_total+IM+1 if j==j_total else j` | `j==j_total` 的面分隔单元 `j_total-1` 与 `j_total`，不涉及右 ghost。恒取 `j` |
| B6 | `solvesupple.py:402-407` | 三次 `shockwave_catcher` 索引全写成 `(i, j_total+2)` | 后两次覆盖第一次；`shockwave_n[i][j_total+3]` 恒为 0 → 切割线附近激波探测器失效。第三次调用还越界，已删除 |
| B7 | `initialize.py:21` | Sutherland 公式用了被形参 `T0`（来流静温 300 K）遮蔽的参考温度 | 应为 `cc.T0 = 288.16 K`，否则分子粘度错误并污染初始 ν̃ |
| B8 | `solvesupple.py:61-66` | 壁面 ghost 的标量恒取 `CellList[1][j]`，速度却取第 `im` 层 | 镜像不自洽；标量统一改取 `CellList[im][j]` |
| B9 | `solvesupple.py:91,153` | 远场 ghost `ma = (u²+v²)/(γRT)`，漏开方 | 算出的是 Ma²。（仅影响诊断输出） |
| B10 | `boundary.py:54,61` | 入流处强制 `face.miubl = 0` | S-A 远场应给定来流工作变量 ν̃∞；置零会持续冲刷湍流粘度，全场退化为层流。改用 `cc.miublll = 0.1·ν∞` |
| B11 | `config.json` | `Pr: 0.9, Prt: 0.71` —— 与同一文件注释（`Pr` 一般 0.71 / `Prt` 一般 0.9）恰好互换 | 空气的层流普朗特数 ≈0.71、湍流 ≈0.9，已对调 |
| B12 | `solvesupple.py:44` | `min_timestep()` 内累加 `cc.totaltime`，而该函数每个 RK 级都被调用 | 物理时间被放大 5 倍。累加移到 `solvemain.RK` 中每步一次；Δt 在 RK 步内也应冻结 |

## C. 健壮性 / 资源问题

| # | 位置 | 问题 | 修复 |
|---|------|------|------|
| C1 | `solvemain.py:9` | `cell.U_former = cell.U` 是引用绑定而非拷贝 | 改 `.copy()`。当前靠「右侧表达式新建数组」侥幸不出错，但极脆弱 |
| C2 | `classconfig.py` | `copy_grad` 直接绑定 `src` 的 ndarray | 改 `.copy()`，避免 ghost 与内部单元共享同一数组 |
| C3 | `turbulence.py:41-45` | `S̃` 可能为 0 或负（`fv2 < 0` 时），`r = ν̃/(S̃κ²d²)` 除零/发散 | 按 Allmaras (2012) 建议加下限截断 |
| C4 | `boundary.py:27-29` | 三点 Lagrange 外插分母在退化网格上为 0 | 加退化检测，回退到一阶外插 |
| C5 | `classconfig.py:160,215` | 密度非正时 `print` + `exit(6)` —— 库代码直接杀进程 | 改抛 `FloatingPointError`，并补充压力正性检查 |
| C6 | `main.py:39-40` | 每写一行结果都 `open`/`close` 一次文件 | 单次打开顺序写 |
| C7 | `main.py:17` | `res.log` 以 `"a"` 打开 | 多次运行残差历史首尾相接。改 `"w"` + 表头 |
| C8 | `main.py:13` | `range(1, cc.iteration)` 少跑一步 | 改 `range(1, iteration+1)` |
| C9 | `main.py:27` | 结果导出循环 `range(1, i_total+1)` 越过物理单元上界 `i_total-1` | 把壁面 ghost 也写了出来。改 `range(1, i_total)` |
| C10 | `geometry.py:114` | 壁面距离搜索窗口 `max(15, j_total//5)` 未按 `j_total` 封顶 | `j_total` 小时反复扫描同一批面。封顶到半周 |
| C11 | `output.py:98-99` | `cc.FaceList_n[j][i]` 索引颠倒 | 应为 `[i][j]`（仅影响网格可视化） |
| C12 | `classconfig.py:77-78` | `iteration` / `targetres` 硬编码，未走 `config.json` | 移入配置的 `solver` 段 |

## D. 已核查确认**正确**的部分

为免后续误改，记录几处初看可疑、核对后确认无误的实现：

- `Cw1 = Cb1/κ² + (1+Cb2)·sigma`：配置中 `sigma = 1.5` 是 **σ 的倒数**（σ_SA = 2/3），故与 `Cb1/κ² + (1+Cb2)/σ` 等价。
- `boundary.riemann` 用 `np.linalg.det` 取速度分量：`face.nx/ny` 实为**法向**（命名 `tauer`/`ner` 与几何含义相反），两次行列式恰好给出正确的法向/切向分量，重构公式也自洽。
- JST 激波探测器的索引偏移：`shockwave_tau[k]` 实际以虚拟单元 `k-2` 为中心，故 `adaptive_dissipation` 取 `max(sw[i..i+3])` 正好是关于面 `i` 对称的 4 点模板。
- `form_JST_dissipation_term` 的全部边界模板（`i==1,2,i_total-1,i_total` 与 `j==1,2,j_total`）经逐一展开核对，ghost 索引均正确。
- 几何度量闭合：单元四个面的外法向之和精确为 0（见 `tests/test_geometry.py::test_metric_closure`），保证自由来流可被离散格式精确保持。
