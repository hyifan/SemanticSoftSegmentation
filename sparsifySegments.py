# -*- coding:utf-8 -*-
import numpy as np
from scipy.sparse import spdiags
from scipy.sparse.linalg import cg


# This function implements the relaxed sparsification descibed in Section 3.4
# 此函数实现了第3.4节中描述的放宽的稀疏化。
def sparsifySegments(softSegments, Laplacian, highLevelIters):
    sigmaS = 1   # sparsity
    sigmaF = 1   # fidelity
    delta = 100  # constraint
    h, w, compCnt = softSegments.shape
    N = h * w * compCnt

    spPow = 0.90

    # Get rid of very low/high alpha values and normalize
    # 去掉非常低/高的alpha值并归一化
    softSegments[softSegments < 0.1] = 0
    softSegments[softSegments > 0.9] = 1
    softSegments = softSegments / np.tile(np.reshape(np.sum(softSegments, 2), [h, w, 1]), [1, 1, compCnt])

    # Construct the linear system
    # 构造线性系统
    lap = Laplacian
    Laplacian = np.zeros([N, N])
    for i in range(compCnt):
        Laplacian[i * h * w : (i + 1) * h * w, i * h * w : (i + 1) * h * w] = lap # 变成5个拉普拉斯矩阵，即N×N

    # The alpha constraint
    # alpha约束
    C = np.tile(np.eye(h * w), [1, compCnt]) # 水平连接5个(h*w)×(h*w)的单位矩阵得到(h*w)×N的矩阵
    C = np.dot(C.T, C)
    Laplacian = Laplacian + delta * C # L = L + λC'C

    # The sparsification optimization
    # 稀疏化优化
    softSegments = softSegments[:]
    compInit = softSegments # needed for fidelity energy
    for iter in range(highLevelIters):
        u, v = getUandV(softSegments, spPow) # The sparsity energy
        # L + Du + Dv + I + λC'C，式（17）左边部分
        A = Laplacian + sigmaS * (spdiags(np.reshape(u, [N, 1]).T, 0, N, N) + spdiags(np.reshape(v, [N, 1]).T, 0, N, N)) + sigmaF * np.eye(N)
        b = sigmaS * v + sigmaF * compInit + delta # v + /hat a + λlip，式（17）右边部分
        # pcg 预处理共轭梯度法
        softSegments, info = cg(A, np.reshape(b, [N, 1]), x0=np.reshape(softSegments, [N, 1]))
        softSegments = np.reshape(softSegments, [h, w, compCnt])

    # One final iter for good times (everything but sparsity)
    # 最后一次迭代
    A = Laplacian + sigmaF * np.eye(N)
    b = sigmaF * softSegments + delta
    softSegments, info = cg(A, b, softSegments)

    # Ta-dah
    softSegments = np.reshape(softSegments, [h, w, compCnt])
    return softSegments


def getUandV(comp, spPow):
    # Sparsity terms in the energy
    # 能量中的稀疏项
    eps = 1e-2
    maxU = abs(comp)
    maxU[maxU < eps] = eps
    maxV = abs(comp - 1)
    maxV[maxV < eps] = eps
    u = np.power(maxU, spPow - 2)
    v = np.power(maxV, spPow - 2)
    return u, v