# -*- coding:utf-8 -*-
import heapq
import scipy
import numpy as np
import operation
from Superpixels import Superpixels
from mattingAffinity import mattingAffinity
from softSegmentsFromEigs import softSegmentsFromEigs
from groupSegments import groupSegments
from sparsifySegments import sparsifySegments


def SemanticSoftSegmentation(image, features):
    '''
    生成超像素，计算 Wc: non-local color affinity 和 Ws: semantic affinity
    '''
    superpixels = Superpixels(image, 2500)
    h, w, d = image.shape

    '''
    计算拉普拉斯矩阵L
    '''
    affinities1 = mattingAffinity(image) # 抠图亲和性
    affinities2 = superpixels.neighborAffinities(features) # 语义亲和性(semantic affinity)
    affinities3 = superpixels.nearbyAffinities(image) # 非局部颜色亲和性(non-local color affinity)

    aff2 = operation.multiply(affinities2, 0.01)
    aff3 = operation.multiply(affinities3, 0.01)
    aff = operation.add(affinities1, aff2, aff3)

    N = aff.shape[0]
    D = scipy.sparse.spdiags(np.sum(aff, 0), 0, N, N)
    Laplacian = operation.subtract(D.toarray(), aff) # Equation 6，L = D - W


    '''
    受约束的稀疏化
    '''
    # 提取最小特征向量
    eigCnt = 100
    vals, vecs = scipy.sparse.linalg.eigs(Laplacian, k = eigCnt, which = 'SM') # 计算拉普拉斯矩阵L前100个最小特征向量
    eigenvectors = vecs.real
    eigenvalues = np.zeros([eigCnt, eigCnt])
    for i in range(eigCnt):
        eigenvalues[i, i] = vals[i].real

    # 40层 => 留下15-25个非平凡层
    initialSegmCnt = 2
    sparsityParam = 0.8
    iterCnt = 2
    initSoftSegments = softSegmentsFromEigs(eigenvectors, eigenvalues, Laplacian, h, w, features, initialSegmCnt, iterCnt, sparsityParam)

    # 15-25个非平凡层 => 5层
    segmCnt = 2
    groupedSegments = groupSegments(initSoftSegments, features, segmCnt)

    '''
    放宽的稀疏化
    '''
    # iterCnt = 1
    # softSegments = sparsifySegments(groupedSegments, Laplacian, iterCnt)

    return groupedSegments
