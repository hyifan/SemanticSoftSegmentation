# -*- coding:utf-8 -*-
import numpy as np
from sklearn.cluster import KMeans


# 此函数实现第3.4节中描述的受约束的稀疏化。
# Levin等人介绍了这种方法。在“光谱铺垫”中，这个函数是其原始源代码的一个改编版本。请访问：http://www.vision.huji.ac.il/spectralmatting/
# eigVecs 特征向量
# eigVals 特征值
# 拉普拉斯矩阵 Laplacian
# features 语义图像
# compCnt, maxIter, sparsityParam = initialSegmCnt 40, iterCnt 40, sparsityParam 0.8
def softSegmentsFromEigs(eigVecs, eigVals, Laplacian, h, w, features, compCnt, maxIter, sparsityParam):
    eigValCnt = eigVecs.shape[1] # 特征向量/特征值个数（100）

    '''
    使用features用于k均值，生成 initialSegments
    '''
    features = np.reshape(features, [features.shape[0] * features.shape[1], features.shape[2]])
    # 将 features 聚类为 40 层，返回一个包含每个点的簇索引的 N×1 向量
    initialSegments = KMeans(n_clusters = compCnt).fit_predict(features)

    '''
     生成 softSegments
    '''
    softSegments = np.zeros([len(initialSegments), compCnt])
    for i in range(compCnt):
        softSegments[:, i] = np.double(initialSegments == i) # N行40列，每一列中属于i分类的点为1

    '''
    Newtons迭代
    '''
    spMat = sparsityParam # γ=0.8

    # Second derivative of first and second terms in the sparsity penalty
    # 稀疏惩罚中第一和第二个术语的二阶导数
    thr_e = 1e-10
    w1 = 0.3
    w0 = 0.3

    s1 = abs(softSegments - 1)
    s1[s1 < thr_e] = thr_e
    s0 = abs(softSegments)
    s0[s0 < thr_e] = thr_e
    e1 = np.dot(np.power(w1, sparsityParam), np.power(s1, spMat - 2)) # |1-alpha|^(γ-2)
    e0 = np.dot(np.power(w0, sparsityParam), np.power(s0, spMat - 2)) # |alpha|^(γ-2)

    scld = 1
    eig_vectors = eigVecs[:, 0 : eigValCnt] # 特征向量的 N × K 矩阵
    eig_values = eigVals[0 : eigValCnt, 0 : eigValCnt]  # 特征值的 K × K 矩阵

    # First iter no for removing zero components
    # 第一次迭代没有移除0分量
    # Y = ceil(X) 将 X 的每个元素四舍五入到大于或等于该元素的最接近整数。
    removeIter = np.ceil(maxIter / 4)
    removeIterCycle = np.ceil(maxIter / 4)

    # Compute matting component with sparsity prior
    # 通过稀疏先验计算抠图分量
    for iter in range(1, maxIter + 1):
        print('迭代次数', iter)
        # Construct the matrices in Eq(9) in Spectral Matting
        # 用光谱抠图中的公式（9）构造矩阵 http://webee.technion.ac.il/people/anat.levin/papers/spectral-matting-levin-etal-pami08.pdf
        tA = np.zeros([(compCnt - 1) * eigValCnt, (compCnt - 1) * eigValCnt])
        tb = np.zeros([(compCnt - 1) * eigValCnt, 1])
        for k in range(compCnt - 1):
            weighted_eigs = np.tile(np.reshape(e1[:, k] + e0[:, k], [e1.shape[0], 1]), [1, eigValCnt]) * eig_vectors
            tA[k * eigValCnt : (k + 1) * eigValCnt, k * eigValCnt : (k + 1) * eigValCnt] = np.dot(eig_vectors.T, weighted_eigs) + scld * eig_values
            tb[k * eigValCnt : (k + 1) * eigValCnt] = np.dot(eig_vectors.T, np.reshape(e1[:, k], [e1.shape[0], 1]))

        k = compCnt - 1
        weighted_eigs = np.tile(np.reshape(e1[:, k] + e0[:, k], [e1.shape[0], 1]), [1, eigValCnt]) * eig_vectors
        ttA = np.dot(eig_vectors.T, weighted_eigs) + scld * eig_values
        ttbSum = scld * np.sum(np.dot(eig_vectors.T, Laplacian), 1)
        ttb = np.dot(eig_vectors.T, np.reshape(e0[:, k], [e0.shape[0], 1])) + np.reshape(ttbSum, [ttbSum.shape[0], 1])

        tA = tA + np.tile(ttA, [compCnt - 1, compCnt - 1])
        tb = tb + np.tile(ttb, [compCnt - 1, 1])

        # Solve for weights
        y = np.reshape(np.dot(np.linalg.inv(tA), tb), [eigValCnt, compCnt - 1])

        # Compute the matting comps from weights
        softSegments[:, 0 : compCnt - 1] = np.dot(eigVecs[:, 0 : eigValCnt], y)
        softSegments[:, compCnt - 1] = 1 - np.sum(softSegments[:, 0 : compCnt - 1], 1) # Sets the last one as 1-sum(others), guaranteeing \sum(all) = 1

        # Remove matting components which are close to zero every once in a while
        # 每隔一段时间移除接近零的抠图分量
        if (iter > removeIter):
            nzii = []
            for i in range(compCnt):
                if (np.max(softSegments[:, i]) > 0.001):
                    nzii.append(i)
            compCnt = len(nzii)
            newSoftSegments = np.zeros([softSegments.shape[0], compCnt])
            for i in range(compCnt):
                newSoftSegments[:, i] = softSegments[:, nzii[i]]
            softSegments = newSoftSegments
            removeIter = removeIter + removeIterCycle

        # Recompute the derivatives of sparsity penalties
        # 重新计算稀疏惩罚的导数
        s1 = abs(softSegments - 1)
        s1[s1 < thr_e] = thr_e
        s0 = abs(softSegments)
        s0[s0 < thr_e] = thr_e
        e1 = np.dot(np.power(w1, sparsityParam), np.power(s1, spMat - 2))
        e0 = np.dot(np.power(w0, sparsityParam), np.power(s0, spMat - 2))

    softSegments = np.reshape(softSegments, [h, w, softSegments.shape[1]])
    return softSegments