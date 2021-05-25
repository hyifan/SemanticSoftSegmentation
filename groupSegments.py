# -*- coding:utf-8 -*-
import numpy as np
from sklearn.cluster import KMeans


# A simple grouping of soft segments w.r.t. their semantic features
# as described in Section 3.4.
# 软分割w.r.t.它们的语义特征的简单分组，如第3.4节所述。
def groupSegments(segments, features, segmCnt):
    h, w, d = segments.shape # 高、宽、上个步骤分的层数（该例为40层）
    compFeatures = np.zeros([d, features.shape[2]]) # 40 × 3

    for i in range(d):
        s = np.reshape(segments[:, :, i], [h, w, 1])
        cc = np.tile(s, [1, features.shape[2]]) * features
        # cc中属于特征i的所有颜色总和/属于特征i的像素数量（segments中所有属于i的所有元素的和）
        cc = np.sum(np.sum(cc, 1), 0) / np.sum(np.sum(s, 1), 0)
        compFeatures[i, :] = cc # 平均颜色

    ids = KMeans(n_clusters = segmCnt).fit_predict(compFeatures) # 用KMeans将颜色相近的分为一组
    groupedSegments = np.zeros([h, w, segmCnt])

    for i in range(segmCnt):
        groupedSegments[:, :, i] = np.sum(segments[:, :, ids==i], 2)

    return groupedSegments