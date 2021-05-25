# -*- coding:utf-8 -*-
import numpy as np
from imguidedfilter import imguidedfilter
from sklearn.decomposition import PCA
from sklearn import preprocessing


def preprocessFeatures(features, image):
	# 由于网络中的一些不稳定因素，过滤掉超高的数字
	features[features < -5] = -5
	features[features > 5] = 5

	'''
	使用引导滤波器优化features使其与image边缘对齐
	'''
	fd = features.shape[2]
	maxfd = fd - fd % 3
	for i in range(0, maxfd, 3):
		features[:, :, i : i + 3] = imguidedfilter(features[:, :, i : i + 3], image, (10, 10), 0.01)
	for i in range(maxfd, fd):
		features[:, :, i] = imguidedfilter(features[:, :, i], image, (10, 10), 0.01)

	'''
	使用PCA降到3维
	'''
	h, w, d = features.shape
	features = np.reshape(features, [h*w, d])
	pca = PCA(n_components=3)
	pca.fit(features)
	simp = pca.fit_transform(features)
	simp = np.reshape(simp, [h, w, 3])

	'''
	归一化[0,1]
	'''
	for i in range (0, 3):
		simp[:,:,i] = simp[:,:,i] - np.min(simp[:,:,i]);
		simp[:,:,i] = simp[:,:,i] / np.max(simp[:,:,i]);

	return simp
