import classconfig as cc
import output as ot
import numpy as np
import math

def initialization(T0=cc.T,AOA=cc.AOA,Ma=cc.Ma,P0=cc.P):
    """标准初始化,使用入口条件,需要给定:\n 
    来流总温`T0`、马赫数`Ma`、压力`P0`和攻角`AOA`(单位:°)"""
    cc.totaltime = 0
    for i in range(1, cc.i_total):
        for j in range(1, cc.j_total + 1):
            cc.CellList[i][j].ma = Ma
            cc.CellList[i][j].T = T0
            cc.CellList[i][j].p = P0
            cc.CellList[i][j].c = math.sqrt(cc.gamma*cc.R*cc.CellList[i][j].T)
            cc.CellList[i][j].rho = cc.CellList[i][j].p/(cc.R*cc.CellList[i][j].T)
            cc.CellList[i][j].u = cc.CellList[i][j].c * Ma * math.cos(math.radians(AOA))
            cc.CellList[i][j].v = cc.CellList[i][j].c * Ma * math.sin(math.radians(AOA))
            cc.CellList[i][j].E = cc.CellList[i][j].p/(cc.CellList[i][j].rho*(cc.gamma-1))+(cc.CellList[i][j].u**2+cc.CellList[i][j].v**2)/2
            cc.CellList[i][j].H = cc.CellList[i][j].E + cc.CellList[i][j].p/cc.CellList[i][j].rho
            # BUGFIX: Sutherland 公式的参考温度是 cc.T0(288.16 K),而非被同名
            #         形参 T0(来流静温)遮蔽的那个值 —— 原式在 T≠cc.T0 时给出
            #         错误的分子粘度,进而污染初始 ν̃.
            cc.CellList[i][j].miu = cc.mu0 * (cc.CellList[i][j].T/cc.T0)**1.5 * (cc.T0+cc.Ts)/(cc.CellList[i][j].T+cc.Ts)
            cc.CellList[i][j].miubl = cc.CellList[i][j].miu *0.1/cc.CellList[i][j].rho

def initialization_main():
    """执行标准初始化,并将部分结果输出到文件中."""
    # some temp variables
    cc.shockwave_tau = np.zeros((cc.i_total+cc.IM+1,cc.j_total+1))
    cc.shockwave_n = np.zeros((cc.i_total,cc.j_total+cc.IM+1))
    cc.density_table = np.zeros((cc.i_total+1,cc.j_total+1))
    initialization()
    ot.initialize_output()