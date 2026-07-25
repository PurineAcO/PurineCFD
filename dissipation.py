import math
import classconfig as cc

def Spectral_Radius(face:cc.face_class,cell_1:cc.cell_class,cell_2:cc.cell_class):
    """形成面单元谱半径*λf*  (`lambda_f`)"""
    face.lambda_f = 0.5 * cc.CFL *(cell_1.vol/cell_1.localdt 
                                   + cell_2.vol/cell_2.localdt)

def shockwave_catcher(index:tuple,direction:str,cell_1:cc.cell_class,
                      cell_2:cc.cell_class,cell_3:cc.cell_class):
    """激波捕捉因子,其中单元排列为`cell_1|(face)|cell_2|cell_3`"""
    i = index[0]
    j = index[1]
    if direction == "tau":
        cc.shockwave_tau[i][j] = abs((cell_1.p - 2*cell_2.p + cell_3.p)/
                                 (cell_1.p + 2*cell_2.p + cell_3.p))
    elif direction== "n":
        cc.shockwave_n[i][j] = abs((cell_1.p - 2*cell_2.p + cell_3.p)/
                                 (cell_1.p + 2*cell_2.p + cell_3.p))

def adaptive_dissipation(face:cc.face_class,direction:str):
    """自适应耗散因子,其中`face`是当前面,其余网格均在前面"""
    i,j = face.index
    if direction == "tau":
        face.epsilon[1] = cc.k2 * max(cc.shockwave_tau[i][j],cc.shockwave_tau[i+1][j],
                                      cc.shockwave_tau[i+2][j],cc.shockwave_tau[i+3][j])
        face.epsilon[2] = max(0,cc.k4-face.epsilon[1])
    if direction == "n":
        face.epsilon[1] = cc.k2 * max(cc.shockwave_n[i][j],cc.shockwave_n[i][j+1],
                                      cc.shockwave_n[i][j+2],cc.shockwave_n[i][j+3])
        face.epsilon[2] = max(0,cc.k4-face.epsilon[1])

    

# def form_JST_dissipation_term(face : cc.face_class,
#                               cell_b:cc.cell_class,cell_bb:cc.cell_class,
#                               cell_f:cc.cell_class,cell_ff:cc.cell_class):
#     """*JST*人工粘性项形成,其中单元排列为:`cell_bb|cell_b|(face)|cell_f|cell_ff`"""
#     d1U = cell_f.U-cell_b.U
#     d3U = cell_ff.U - 3*cell_f.U + 3*cell_b.U -cell_bb.U
#     face.Dissipation = face.lambda_f*(face.epsilon[2]*d1U - face.epsilon[3]*d3U)
