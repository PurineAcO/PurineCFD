import numpy as np
import classconfig as cc

def read_mesh(meshfile):
    """
    阅读**O型网格**,并将网格点信息存储在**class** `NodeList` 中.\n
    网格文件的第一行应包含两个整数,分别表示环层数 `i_total` 和周向点数 `j_total`,
    后续行包含每个网格点的x和y坐标,保证按照逆时针排列.\n
    你既可以将末端封闭也可以不封闭.\n
    """

    with open(meshfile, 'r') as f:
        header = f.readline().strip()
        try:
            cc.i_total, cc.j_total= map(int, header.split())  
        except ValueError:
            print(f"Error: The header of {meshfile} is not in the expected format. Please check the file.")
            return 

    data = np.loadtxt(meshfile, skiprows=1)
    cc.meshcnt = len(data)

    if cc.meshcnt == 0:
        print(f"Error: No data found in {meshfile}. Please check the file format and content.")
        return 

    if cc.meshcnt != cc.i_total * cc.j_total:
        print(f"Error: Mismatch in number of points in {meshfile}.")
        return 

    readindex = 0
    # cc.NodeList = [[]]
    for i in range(1, cc.i_total + 1):
        circle = [[]]
        for j in range(1, cc.j_total + 1):
            tempnode = cc.node_class(readindex)
            tempnode.x = data[readindex, 0]
            tempnode.y = data[readindex, 1]
            circle.append(tempnode)
            readindex += 1
        cc.NodeList.append(circle)

    # examine if the mesh is closed
    tol = 1e-12
    closed = True
    for i in range(1, cc.i_total + 1):
        n1 = cc.NodeList[i][1]        # the first node of each ring
        n_last = cc.NodeList[i][-1]   # the last node of each ring
        if abs(n1.x - n_last.x) > tol or abs(n1.y - n_last.y) > tol:
            # judge if the first and last node of each ring are the same (closed)
            closed = False
            break
    if closed:
        for i in range(1, cc.i_total + 1):
            cc.NodeList[i].pop()       # remove the last node of each ring
        cc.j_total -= 1
        cc.meshcnt -= cc.i_total
        print(f"Detected closed data, automatically corrected: j_total → {cc.j_total}, meshcnt → {cc.meshcnt}")

    print(f"successfully read {meshfile} with {cc.meshcnt} points.")
    print(f"i_total: {cc.i_total}, j_total: {cc.j_total}, meshcnt: {cc.meshcnt}")


def mesh_visualization():
    """可视化网格点,以验证网格数据的正确性.\n"""

    x=[]; y=[]
    for i in range(len(cc.NodeList)):
        for j in range(len(cc.NodeList[i])):
            x.append(cc.NodeList[i][j].x)
            y.append(cc.NodeList[i][j].y)       
    
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 8), dpi=100)
    plt.scatter(x, y, s=5, c='steelblue', alpha=0.7, edgecolors='none')

    plt.xlabel('X Coordinate', fontsize=12)
    plt.ylabel('Y Coordinate', fontsize=12)
    plt.title('Grid Points Visualization', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.axis('equal')

    plt.show()

# if __name__ == "__main__":
#     read_mesh("yuanzhudata.txt")
#     mesh_visualization()
