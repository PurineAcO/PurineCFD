"""
O 型网格可视化工具 —— 读取 fangdata.txt 并绘制环状网格
结构: i=环层(法向), j=周向点(切向), 内圈椭圆→外圈圆
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def read_mesh(meshfile):
    """读取网格文件，返回网格数据"""
    with open(meshfile, 'r', encoding='utf-8') as f:
        header = f.readline().strip()
        i_total, j_total = map(int, header.split())
        data = np.loadtxt(f)

    assert len(data) == i_total * j_total, \
        f"数据点数量不匹配: 期望 {i_total * j_total}, 实际 {len(data)}"

    grid = data.reshape(i_total, j_total, 2)
    return i_total, j_total, grid[:, :, 0], grid[:, :, 1]


def plot_mesh(x, y, i_total, j_total, title="O 型网格可视化"):
    """绘制 O 型网格 (i=环向, j=周向)"""
    fig, ax = plt.subplots(figsize=(8, 8))

    # ---- 绘制 i 方向线（环线 - 每个环层连起来） ----
    for i in range(i_total):
        # 首尾闭合
        xi = np.append(x[i, :], x[i, 0])
        yi = np.append(y[i, :], y[i, 0])
        if i == 0:
            ax.plot(xi, yi, 'r-', linewidth=2.5, alpha=0.9, label='第1环 (椭圆)')
        elif i == i_total - 1:
            ax.plot(xi, yi, 'g-', linewidth=2.5, alpha=0.9, label=f'第{i_total}环 (圆)')
        else:
            ax.plot(xi, yi, 'b-', linewidth=1.0, alpha=0.6)

    # ---- 绘制 j 方向线（径向线 - 连接相同角度各层） ----
    for j in range(j_total):
        ax.plot(x[:, j], y[:, j], 'b-', linewidth=0.8, alpha=0.4)

    # ---- 标记网格点 ----
    ax.scatter(x, y, c='red', s=15, zorder=5, label='网格点')

    # ---- 标注信息 ----
    # 内椭圆标注
    ax.annotate(f'内圈: 椭圆 (a={x[0,0]:.1f}, b={y[0, j_total//4]:.1f})',
                xy=(x[0, 0], y[0, 0]), fontsize=10, color='darkred',
                fontweight='bold',
                xytext=(x[0, 0] + 0.3, y[0, 0] + 0.3),
                arrowprops=dict(arrowstyle='->', color='darkred'),
                bbox=dict(boxstyle='round', fc='white', ec='red', alpha=0.8))
    # 外圆标注
    R_outer = np.sqrt(x[-1, 0]**2 + y[-1, 0]**2)
    ax.annotate(f'外圈: 圆 (R={R_outer:.1f})',
                xy=(x[-1, 0], y[-1, 0]), fontsize=10, color='darkgreen',
                fontweight='bold',
                xytext=(x[-1, 0] + 0.3, y[-1, 0] + 0.3),
                arrowprops=dict(arrowstyle='->', color='darkgreen'),
                bbox=dict(boxstyle='round', fc='white', ec='green', alpha=0.8))

    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(f'{title}\n'
                 f'{i_total} 环层 × {j_total} 周向点',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    return fig, ax


def print_mesh_info(x, y, i_total, j_total):
    """打印网格信息"""
    print(f"{'='*50}")
    print(f"  网格规模: {i_total} 环层 × {j_total} 周向点 = {i_total*j_total} 点")
    print(f"{'='*50}")

    # 第1环（椭圆）
    rx1 = np.max(x[0, :])
    ry1 = np.max(y[0, :])
    print(f"  第1环 (内椭圆): rx={rx1:.2f}, ry={ry1:.2f}")

    # 中间环
    for i in [1, 4, 7]:
        rx = np.max(np.abs(x[i, :]))
        ry = np.max(np.abs(y[i, :]))
        print(f"  第{i+1}环: rx={rx:.2f}, ry={ry:.2f}")

    # 最后一环（圆）
    rx_last = np.max(np.abs(x[-1, :]))
    ry_last = np.max(np.abs(y[-1, :]))
    print(f"  第{i_total}环 (外圆): rx={rx_last:.2f}, ry={ry_last:.2f}")
    print(f"{'='*50}")


if __name__ == '__main__':
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    meshfile = os.path.join(script_dir, 'fangdata.txt')

    i_total, j_total, x, y = read_mesh(meshfile)
    print_mesh_info(x, y, i_total, j_total)

    fig, ax = plot_mesh(x, y, i_total, j_total, "fangdata.txt — O 型网格")

    output_png = os.path.join(script_dir, 'fangdata_mesh.png')
    fig.savefig(output_png, dpi=150)
    print(f"\n网格图已保存: {output_png}")
    plt.show()

    # 保存图片
    output_png = os.path.join(script_dir, 'fangdata_mesh.png')
    fig.savefig(output_png, dpi=150)
    print(f"\n网格图已保存: {output_png}")
    plt.show()
