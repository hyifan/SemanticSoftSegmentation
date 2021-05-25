# -*- coding:utf-8 -*-
import numpy as np


# 将W沿对角线复制变为对称矩阵
def add_T(W):
	allLen = np.int(W.shape[0])
	length = np.int(W.shape[0]/100)

	for i in range(100):
		for j in range(100):
			start1 = i * length
			start2 = j * length
			end1 = (i + 1) * length if (i + 1) * length < allLen else allLen
			end2 = (j + 1) * length if (j + 1) * length < allLen else allLen
			W[start1 : end1, start2 : end2] = W[start1 : end1, start2 : end2] + W[start1 : end1, start2 : end2].T

	return W


def multiply(arr, num):
	allLen = np.int(arr.shape[0])
	length = np.int(arr.shape[0]/100)

	for i in range(100):
		for j in range(100):
			start1 = i * length
			start2 = j * length
			end1 = (i + 1) * length if (i + 1) * length < allLen else allLen
			end2 = (j + 1) * length if (j + 1) * length < allLen else allLen
			arr[start1 : end1, start2 : end2] = num * arr[start1 : end1, start2 : end2]

	return arr


def add(arr1, arr2, arr3):
	allLen = np.int(arr1.shape[0])
	length = np.int(arr1.shape[0]/100)

	for i in range(100):
		for j in range(100):
			start1 = i * length
			start2 = j * length
			end1 = (i + 1) * length if (i + 1) * length < allLen else allLen
			end2 = (j + 1) * length if (j + 1) * length < allLen else allLen
			arr1[start1 : end1, start2 : end2] = arr1[start1 : end1, start2 : end2] + arr2[start1 : end1, start2 : end2] + arr3[start1 : end1, start2 : end2]

	return arr1


def subtract(arr1, arr2):
	allLen = np.int(arr1.shape[0])
	length = np.int(arr1.shape[0]/100)
	for i in range(100):
		for j in range(100):
			start1 = i * length
			start2 = j * length
			end1 = (i + 1) * length if (i + 1) * length < allLen else allLen
			end2 = (j + 1) * length if (j + 1) * length < allLen else allLen
			arr1[start1 : end1, start2 : end2] = arr1[start1 : end1, start2 : end2] - arr2[start1 : end1, start2 : end2]
	return arr1
