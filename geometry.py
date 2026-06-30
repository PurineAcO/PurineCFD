import numpy as np
import classconfig as cc
import output as ot

def calc_cell_vol():
    """新建网格类,计算每个单元的体积,并存储在`CellList` 中.\n"""
    for i in range(1,cc.i_total,1):
        circlecell = [[]]
        for j in range(1,cc.j_total,1):
            vec1 = np.array([cc.NodeList[i+1][j+1].x -cc.NodeList[i][j].x
                             ,cc.NodeList[i+1][j+1].y -cc.NodeList[i][j].y])
            vec2 = np.array([cc.NodeList[i+1][j].x -cc.NodeList[i][j+1].x
                             ,cc.NodeList[i+1][j].y -cc.NodeList[i][j+1].y])
            thisindex = (i,j)
            tempcell = cc.cell_class(thisindex)
            tempcell.vol = 0.5 * np.abs(vec1[0]*vec2[1] - vec1[1]*vec2[0])
            circlecell.append(tempcell)

        # wrap-around cell: j = j_total
        vec1 = np.array([cc.NodeList[i+1][1].x -cc.NodeList[i][cc.j_total].x
                         ,cc.NodeList[i+1][1].y -cc.NodeList[i][cc.j_total].y])
        vec2 = np.array([cc.NodeList[i+1][cc.j_total].x -cc.NodeList[i][1].x
                         ,cc.NodeList[i+1][cc.j_total].y -cc.NodeList[i][1].y])
        thisindex = (i,cc.j_total)
        tempcell = cc.cell_class(thisindex)
        tempcell.vol = 0.5 * np.abs(vec1[0]*vec2[1] - vec1[1]*vec2[0])
        circlecell.append(tempcell)

        cc.CellList.append(circlecell)

def calc_cell_center():
    """计算每个单元的中心,并存储在`CellList` 中."""
    for i in range(1, cc.i_total):
        for j in range(1, cc.j_total):
            x1,y1 = cc.NodeList[i][j].x, cc.NodeList[i][j].y
            x2,y2 = cc.NodeList[i+1][j].x, cc.NodeList[i+1][j].y
            x3,y3 = cc.NodeList[i+1][j+1].x, cc.NodeList[i+1][j+1].y
            x4,y4 = cc.NodeList[i][j+1].x, cc.NodeList[i][j+1].y
            c1 = x1*y2 - x2*y1
            c2 = x2*y3 - x3*y2
            c3 = x3*y4 - x4*y3
            c4 = x4*y1 - x1*y4
            signed_area = 0.5 * (c1 + c2 + c3 + c4)
            a = (x1+x2)*c1 + (x2+x3)*c2 + (x3+x4)*c3 + (x4+x1)*c4
            b = (y1+y2)*c1 + (y2+y3)*c2 + (y3+y4)*c3 + (y4+y1)*c4
            if abs(signed_area) > 1e-30:
                cc.CellList[i][j].x = a / (6 * signed_area)
                cc.CellList[i][j].y = b / (6 * signed_area)

        # wrap-around cell: j = j_total
        j = cc.j_total
        x1,y1 = cc.NodeList[i][j].x, cc.NodeList[i][j].y
        x2,y2 = cc.NodeList[i+1][j].x, cc.NodeList[i+1][j].y
        x3,y3 = cc.NodeList[i+1][1].x, cc.NodeList[i+1][1].y
        x4,y4 = cc.NodeList[i][1].x, cc.NodeList[i][1].y
        c1 = x1*y2 - x2*y1
        c2 = x2*y3 - x3*y2
        c3 = x3*y4 - x4*y3
        c4 = x4*y1 - x1*y4
        signed_area = 0.5 * (c1 + c2 + c3 + c4)
        a = (x1+x2)*c1 + (x2+x3)*c2 + (x3+x4)*c3 + (x4+x1)*c4
        b = (y1+y2)*c1 + (y2+y3)*c2 + (y3+y4)*c3 + (y4+y1)*c4
        if abs(signed_area) > 1e-30:
            cc.CellList[i][j].x = a / (6 * signed_area)
            cc.CellList[i][j].y = b / (6 * signed_area)

def calc_face_direction_tau():
    """计算周向网格的边的法向量(外法向,单位化),存储在`Facelist_tau` 中."""
    for i in range(1, cc.i_total + 1):
        circleface = [[]]
        for j in range(1, cc.j_total):
            dx = cc.NodeList[i][j+1].x - cc.NodeList[i][j].x
            dy = cc.NodeList[i][j+1].y - cc.NodeList[i][j].y
            mag = np.sqrt(dx**2 + dy**2)
            tempface_tau = cc.face_class((i, j))
            if mag > 1e-30:
                tempface_tau.ni = dy / mag
                tempface_tau.nj = -dx / mag
                tempface_tau.mx = (cc.NodeList[i][j].x + cc.NodeList[i][j+1].x) / 2
                tempface_tau.my = (cc.NodeList[i][j].y + cc.NodeList[i][j+1].y) / 2
            circleface.append(tempface_tau)

        # wrap-around: j = j_total → j+1 = 1
        dx = cc.NodeList[i][1].x - cc.NodeList[i][cc.j_total].x
        dy = cc.NodeList[i][1].y - cc.NodeList[i][cc.j_total].y
        mag = np.sqrt(dx**2 + dy**2)
        tempface_tau = cc.face_class((i, cc.j_total))
        if mag > 1e-30:
            tempface_tau.ni = dy / mag
            tempface_tau.nj = -dx / mag
            tempface_tau.mx = (cc.NodeList[i][cc.j_total].x + cc.NodeList[i][1].x) / 2
            tempface_tau.my = (cc.NodeList[i][cc.j_total].y + cc.NodeList[i][1].y) / 2
        circleface.append(tempface_tau)

        cc.Facelist_tau.append(circleface)

def calc_face_direction_n():
    """计算径向网格的边的法向量(外法向,单位化),存储在`FaceList_n` 中."""
    for j in range(1, cc.j_total + 1):
        circleface = [[]]
        for i in range(1, cc.i_total):
            dx = cc.NodeList[i+1][j].x - cc.NodeList[i][j].x
            dy = cc.NodeList[i+1][j].y - cc.NodeList[i][j].y
            mag = np.sqrt(dx**2 + dy**2)
            tempface_n = cc.face_class((i, j))
            if mag > 1e-30:
                tempface_n.ni = -dy / mag
                tempface_n.nj = dx / mag
                tempface_n.mx = (cc.NodeList[i][j].x + cc.NodeList[i+1][j].x) / 2
                tempface_n.my = (cc.NodeList[i][j].y + cc.NodeList[i+1][j].y) / 2
            circleface.append(tempface_n)

        cc.FaceList_n.append(circleface)

def calc_most_near_walldistance():
    """计算每个单元中心到壁面的最近距离, 存储在 `cell.sad` 中.\n
    优化: 仅搜索周向 `w=max(5,j_total//20)` 范围内的壁面, 而非全局遍历.
    """
    # window: ~20% of circumference, at least 15
    window = max(15, cc.j_total // 5)
    j_total = cc.j_total
    wall_faces = cc.Facelist_tau[1]  # first ring = wall

    for i in range(1, cc.i_total):
        for j in range(1, j_total + 1):
            cell = cc.CellList[i][j]
            min_dist = float('inf')
            # search k in [j-window, j+window] with wrap-around
            for dk in range(-window, window + 1):
                k = (j - 1 + dk) % j_total + 1  # 1-indexed wrap
                face = wall_faces[k]
                dist = np.sqrt((cell.x - face.mx) ** 2 + (cell.y - face.my) ** 2)
                if dist < min_dist:
                    min_dist = dist
            cell.sad = min_dist

def geometry_main(debugoutput, ifrender=False, showwhat=(True, True, True)):
    """几何结构的初始化,共包含以下步骤:\n
    (1)计算单元体积和中心;(2)计算周向\径向网格的边的法向量;\n 
    (3)计算每个单元中心到壁面的最近距离.\n 
    `ifrender` 控制是否进行网格可视化,\n 
    `showwhat` 是一个三元组,分别控制显示单元中心、周向法向、径向法向."""
    calc_cell_vol()
    calc_cell_center()
    calc_face_direction_tau()
    calc_face_direction_n()
    calc_most_near_walldistance()
    ot.geometry_debug(debugoutput)
    if ifrender:
        ot.mesh_visualization("mesh_visual.svg",
                           show_centers=showwhat[0],
                           show_tau=showwhat[1],
                           show_n=showwhat[2])
