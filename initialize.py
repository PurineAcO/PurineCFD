import classconfig as cc
import output as ot
import math

def initialization(T0=cc.T0,AOA=cc.AOA,Ma=cc.Ma,P0=cc.P0):
    """标准初始化,使用入口条件,需要给定:\n 
    来流总温`T0`、马赫数`Ma`、压力`P0`和攻角`AOA`(单位:°)"""
    cc.totaltime = 0
    for i in range(1, cc.i_total):
        for j in range(1, cc.j_total + 1):
            cc.CellList[i][j].ma = Ma
            cc.CellList[i][j].T = T0/(1 + (cc.gamma-1)/2 * (Ma**2))
            cc.CellList[i][j].p = P0 * (cc.CellList[i][j].T / T0) ** (cc.gamma / (cc.gamma - 1))
            cc.CellList[i][j].c = math.sqrt(cc.gamma*cc.R*cc.CellList[i][j].T)
            cc.CellList[i][j].rho = cc.CellList[i][j].p/(cc.R*cc.CellList[i][j].T)
            cc.CellList[i][j].u = cc.CellList[i][j].c * Ma * math.cos(AOA * math.pi / 180)
            cc.CellList[i][j].v = cc.CellList[i][j].c * Ma * math.sin(AOA * math.pi / 180)
            cc.CellList[i][j].E = cc.CellList[i][j].p/(cc.CellList[i][j].rho*(cc.gamma-1))+(cc.CellList[i][j].u**2+cc.CellList[i][j].v**2)/2
            cc.CellList[i][j].H = cc.CellList[i][j].E + cc.CellList[i][j].p/cc.CellList[i][j].rho
            cc.CellList[i][j].miu = cc.mu0 * (cc.CellList[i][j].T/T0)**1.5 * (T0+cc.Ts)/(cc.CellList[i][j].T+cc.Ts)
            cc.CellList[i][j].miubl = cc.CellList[i][j].miu *0.1/cc.CellList[i][j].rho

def initialization_main():
    """执行标准初始化,并将部分结果输出到文件中."""
    initialization()
    ot.initialize_output()