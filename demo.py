# -*- coding:utf-8 -*-
import numpy as np
import cv2
import scipy
import datetime
from preprocessFeatures import preprocessFeatures
from SemanticSoftSegmentation import SemanticSoftSegmentation


start = datetime.datetime.now()


'''
获取 image 和 features
'''
imageOrigin = cv2.imread('images/img.png') # 读取3维image
image = imageOrigin/255 # 归一化[0,1]
features = scipy.io.loadmat('images/img.mat')['embedmap'] # 读取128维features
features = preprocessFeatures(features, image) # 使用引导滤波器优化特征使其与图像边缘对齐，再使用PCA降到3维，并归一化[0,1]
cv2.imwrite('images/features.jpg', features * 255)


'''
计算语言软分割得到的alpha遮罩sss
'''
sss = SemanticSoftSegmentation(image, features)


end = datetime.datetime.now()
print(end - start)


'''
根据sss展示最终的png图片
'''
h, w, d = imageOrigin.shape

for index in range(sss.shape[2]):
    res = np.zeros([h, w, 3])
    for i in range(3):
        res[:, :, i] = sss[:, :, index] * imageOrigin[:, :, i]
    b_channel, g_channel, r_channel = cv2.split(res)
    alpha_channel = sss[:, :, index] * 255
    res = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    cv2.imwrite('res/res_' + str(index) + '.png', res)
