import numpy as np
import classconfig as cc

def green_gauss(cell:cc.cell_class,face1:cc.face_class,face2:cc.face_class,
                    face3:cc.face_class,face4:cc.face_class):
        """基于Green-Guass的梯度构建"""
        u_vec = np.array([face1.u,face2.u,face3.u,face4.u])
        v_vec = np.array([face1.v,face2.v,face3.v,face4.v])
        miubl_vec = np.array([face1.miubl,face2.miubl,face3.miubl,face4.miubl])
        T_vec = np.array([face1.T,face2.T,face3.T,face4.T])
        nx_vec = np.array([face1.nx,-face2.nx,face3.nx,-face4.nx])
        ny_vec = np.array([face1.ny,-face2.ny,face3.ny,-face4.ny])
        cell.ugrad = np.array([0,np.dot(u_vec,nx_vec),np.dot(u_vec,ny_vec)])/cell.vol
        cell.vgrad = np.array([0,np.dot(v_vec,nx_vec),np.dot(v_vec,ny_vec)])/cell.vol
        cell.miublgrad = np.array([0,np.dot(miubl_vec,nx_vec),np.dot(miubl_vec,ny_vec)])/cell.vol
        cell.Tgrad = np.array([0,np.dot(T_vec,nx_vec),np.dot(T_vec,ny_vec)])/cell.vol
