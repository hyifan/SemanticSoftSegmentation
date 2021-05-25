# -*- coding:utf-8 -*-
import numpy as np
import cv2
from scipy import sparse
from operation import add_T


# 计算抠图亲和性
# 该函数实现了Anat Levin, Dani Lischinski, Yair Weiss描述的图像抠图方法
# 除图像以外的所有参数都是可选的。
# 输出是一个稀疏矩阵，它为二进制映射inMap给出的像素的非局部邻域提供非零元素。
def mattingAffinity(image):
    windowRadius = 1 # 估算局部正态分布的窗口大小
    epsilon = 1e-7   # 定义协方差矩阵反转前使用的正则化系数，对于有噪声的图像，它应该更大
    windowSize = 2 * windowRadius + 1
    neighSize = pow(windowSize, 2)
    h, w, d = image.shape
    N = h * w
    epsilon = epsilon / neighSize

    # meanImage 窗口中像素3*1平均值向量
    # covarMat 3*3像素颜色的协方差矩阵
    meanImage, covarMat = localRGBnormalDistributions(image, windowRadius, epsilon)

    # Determine pixels and their local neighbors
    # 确定像素及其局部邻域
    indices = np.reshape(np.array(range(h * w)), [h, w])

    # 以子块滑动的方式（子块先在行上滑动，再在列上滑动）将indices分解成m×n的子矩阵，并将分解以后的子矩阵沿行的方向转换成neighInd的列。
    neighInd = im2col(indices, [windowSize, windowSize]) # windowSize=3时9列，(h-2)*(w-2)行

    # neighInd = neighInd[windowRadius : len(neighInd) - windowRadius, windowRadius : len(neighInd) - windowRadius]
    inInd = neighInd[:, np.int((neighSize + 1) / 2 - 1)] # 矩阵中心位置的索引
    pixCnt = inInd.shape[0] # 中心位置数量

    # Prepare in & out data
    image = np.reshape(image, [N, d])
    meanImage = np.reshape(meanImage, [N, d])
    flowRows = np.zeros([neighSize, neighSize, pixCnt])
    flowCols = np.zeros([neighSize, neighSize, pixCnt])
    flows = np.zeros([neighSize, neighSize, pixCnt])

    # Compute matting affinity
    for i in range(pixCnt):
        neighs = neighInd[i, :] # 位置 inInd[i] 对应的窗口索引
        littleImage = np.zeros([neighs.shape[0], image.shape[1]])
        for j in range(neighs.shape[0]):
            littleImage[j, :] = image[np.int(neighs[j])]
        shiftedWinColors = littleImage - np.tile(meanImage[np.int(inInd[i]), :], [neighs.shape[0], 1]) # Ii-uk
        inv = np.linalg.inv(covarMat[:, :, np.int(inInd[i])])
        flows[:, :, i] = np.dot(np.dot(shiftedWinColors, inv), shiftedWinColors.T)
        neighs = np.tile(neighs, [neighs.shape[0], 1])
        flowRows[:, :, i] = neighs
        flowCols[:, :, i] = neighs.T

    flows = (flows + 1) / neighSize

    flows = np.reshape(flows, [neighSize*neighSize*pixCnt])
    flowRows = np.reshape(flowRows, [neighSize*neighSize*pixCnt])
    flowCols = np.reshape(flowCols, [neighSize*neighSize*pixCnt])

    W = sparse.csr_matrix((flows, (flowRows, flowCols)), shape = (N, N)).todense()
    W = add_T(W)
    W = W / 2
    return W


# meanImage 窗口中像素3*1平均值向量
# covarMat 3*3像素颜色的协方差矩阵
def localRGBnormalDistributions(image, windowRadius, epsilon):
    h, w, d = image.shape
    N = h * w
    windowSize = 2 * windowRadius + 1

    # 窗口中像素3*1平均值向量
    meanImage = np.zeros([h, w, d])
    meanImage[:, :, 0] = cv2.blur(image[:, :, 0], (windowSize, windowSize))
    meanImage[:, :, 1] = cv2.blur(image[:, :, 1], (windowSize, windowSize))
    meanImage[:, :, 2] = cv2.blur(image[:, :, 2], (windowSize, windowSize))

    # 3*3像素颜色的协方差矩阵
    covarMat = np.zeros([3, 3, N])
    for i in range(3):
        for j in range(i, 3):
            temp = cv2.blur(image[:, :, i] * image[:, :, j], (windowSize, windowSize)) - meanImage[:, :, i] *  meanImage[:, :, j]
            covarMat[i, j, :] = np.reshape(temp, [h*w])

    for i in range(3):
        covarMat[i, i, :] = covarMat[i, i, :] + epsilon

    return meanImage, covarMat


# 实现 Matlab im2col功能
def im2col(A, size):
    h, w = A.shape
    x = h - size[0] + 1
    y = w - size[1] + 1
    result = np.zeros([x * y, size[0] * size[1]])
    for i in range(x):
        for j in range(y):
            result[j * x + i, :] = A[i : i + size[0], j : j + size[1]].ravel()
    return result
