# -*- coding:utf-8 -*-
import numpy as np
from skimage.segmentation import slic
from skimage import measure
from scipy.special import erf
from scipy import sparse
from operation import add_T


# 该类用于计算本文描述的基于超像素的亲和性。
class Superpixels(object):
    def __init__(self, image, spcnt):
        self.image = image
        self.spcnt = spcnt # 理想的超像素数
        L = slic(image, n_segments=spcnt, compactness=10) + 1 # 返回【标签矩阵】，标签值从1开始
        self.labels = L
        self.spcount = np.int(np.max(L)) # 实际超级像素数

        h, w = L.shape

        # Find neighboring superpixels 寻找相邻的超像素
        neigh = []
        for i in range(h):
            for j in range(w):
                # 不是最后一行
                if (i != h - 1):
                    # 向下对比
                    if (L[i][j] != L[i+1][j]):
                        neigh.append(sorted([L[i][j], L[i+1][j]]))
                    # 不是第一列，向左对比
                    if (j != 0):
                        if (L[i][j] != L[i+1][j-1]):
                            neigh.append(sorted([L[i][j], L[i+1][j-1]]))
                    # 不是最后一列，向右对比
                    if (j != w - 1):
                        if (L[i][j] != L[i][j+1]):
                            neigh.append(sorted([L[i][j], L[i][j+1]]))
                        if (L[i][j] != L[i+1][j+1]):
                            neigh.append(sorted([L[i][j], L[i+1][j+1]]))
                # 最后一行
                else:
                    # 不是最后一列，向右对比
                    if (j != w - 1):
                        if (L[i][j] != L[i][j+1]):
                            neigh.append(sorted([L[i][j], L[i][j+1]]))
        neigh = list(set([tuple(n) for n in neigh])) # 去重
        neigh.sort() # 排序
        self.neigh = np.array(neigh) # 包含每个相邻区域对的【标签】

        # Find centroids 寻找质心
        s = measure.regionprops(L)
        cent = [] # 将质心的x和y坐标存储到两列矩阵中
        for i in range(len(s)):
            x = s[i].centroid[0]
            y = s[i].centroid[1]
            index = np.int(h*y + x) # 坐标索引，0开始
            cent.append([x, y, index])
        self.centroids = np.array(cent)


    def computeRegionMeans(self, image):
        # 计算区域平均值
        d = image.shape[2]
        regmeans = np.zeros([self.spcount, d])
        for i in range(self.spcount):
            label = np.where(self.labels == i + 1, 1, 0)
            for j in range(d):
                regmeans[i, j] = np.mean(image[:, :, j] * label)
        return regmeans


    # This is for the semantic affinity, generates affinities in [-1, 1]
    # 计算语义亲和性，范围[-1, 1]
    def neighborAffinities(self, features):
        erfSteepness = 20 # as
        erfCenter = 0.85  # 1 - erfCenter === bs

        h, w, d = features.shape
        N = h * w
        num = self.neigh.shape[0]

        spMeans = self.computeRegionMeans(features) # 大小为[self.spcount, d]
        affs = np.zeros(num)  # 大小为[相邻区域对标签数量 ,1]
        inds1 = np.zeros(num)
        inds2 = np.zeros(num)
        for i in range(num):
            # 循环标签对： ind1 <==> ind2
            ind1 = self.neigh[i, 0]
            ind2 = self.neigh[i, 1]
            # aff = abs(spMeans(ind1, :) - spMeans(ind2, :))
            # erf(erfSteepness * (1 - sqrt(aff * aff') - erfCenter)) = erf(erfSteepness * (1 - erfCenter) - sqrt(aff * aff'))
            affs[i] = sigmoidAff(spMeans[ind1 - 1, :], spMeans[ind2 - 1, :], erfSteepness, erfCenter)
            inds1[i] = self.centroids[ind1 - 1, 2] # 标签ind1对应的质心所对应的索引
            inds2[i] = self.centroids[ind2 - 1, 2] # 标签ind2对应的质心所对应的索引
        # sparse([1,2,3,4],[1,2,3,4],[0,0,1,1])：i=[1,2,3,4]，对应要形成矩阵的行位置；j=[1,2,3,4]，对应要形成矩阵的列位置；s=[0,0,1,1]，对应要形成矩阵对应位置的值。
        # i 和j 的位置为一一对应，即(1,1)(2,2)(3,3)(4,4),将s中的值赋给这四个坐标的位置。
        # 根据 inds1、inds2 和 affs 三元组生成稀疏矩阵 S，将 S 的大小指定为 N×N
        W = sparse.csr_matrix((affs, (inds1, inds2)), shape = (N, N)).todense()
        W = add_T(W) # 将W沿对角线复制变为对称矩阵
        return W

    # This is for the nonlocal color affinity, generates affinities in [0, 1]
    # 计算非局部颜色亲和性，范围[0, 1]
    def nearbyAffinities(self, image):
        erfSteepness = 50 # ac
        erfCenter = 0.95  # 1 - erfCenter === bc
        proxThresh = 0.2  # 半径

        h, w, d = image.shape
        N = h * w

        spMeans = self.computeRegionMeans(image)
        combinationCnt = self.spcount
        combinationCnt = np.int(combinationCnt * (combinationCnt - 1) / 2) # 每个超像素一一对比的次数
        affs = np.zeros(combinationCnt)
        inds1 = np.zeros(combinationCnt)
        inds2 = np.zeros(combinationCnt)
        cnt = 0
        cents = self.centroids[:, 0:2] # 质心对应的xy坐标
        cents[:,0] = cents[:,0] / h # 将质心预处理到[0,1]
        cents[:,1] = cents[:,1] / w
        for i in range(self.spcount):
            for j in range(i + 1, self.spcount):
                centdist = cents[i, 0:2] - cents[j, 0:2] # 质心 i 和 j 的距离
                centdist = np.sqrt(np.dot(centdist, centdist.T))
                if centdist > proxThresh:
                    # 不在 proxThresh 半径内
                    affs[cnt] = 0
                else:
                    # 在 proxThresh 半径内，计算非局部颜色亲和性
                    affs[cnt] = sigmoidAffPos(spMeans[i, :], spMeans[j, :], erfSteepness, erfCenter)
                inds1[cnt] = self.centroids[i, 2]
                inds2[cnt] = self.centroids[j, 2]
                cnt = cnt + 1
        W = sparse.csr_matrix((affs, (inds1, inds2)), shape = (N, N)).todense()
        W = add_T(W)
        return W


def sigmoidAff(feat1, feat2, steepness, center):
    aff = abs(feat1 - feat2)
    aff = 1 - np.sqrt(np.dot(aff, aff.T))
    aff = erf(steepness * (aff - center))
    return aff


def sigmoidAffPos(feat1, feat2, steepness, center):
    aff = abs(feat1 - feat2)
    aff = 1 - np.sqrt(np.dot(aff, aff.T))
    aff = (erf(steepness * (aff - center)) + 1) / 2
    return aff