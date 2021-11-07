from cv2 import cv2
import numpy as np
from utils.keyPoint import KeyPoint
# import argparse
# # 设置参数
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image")
# args = vars(ap.parse_args())
keyPoint = KeyPoint

def changeView(img,pointDst,point1ImgNp):
	#对原图进行透视变换
	#img： 输入图像的位置
	#pointDst： 透视变换后的四个角点坐标
	#point1ImgNp:原图落点坐标
	# 输出
	#point1ImgOrd： 原图落点对应的透视图坐标

	srcImg = keyPoint.imgPretreat(img)   #srcImg是形态学处理后的二值图像

	r, c = img.shape[:2]
	#所有靶标形心坐标
	point = keyPoint.centroidCoordinates(img,srcImg,True)
	fourPoints,targetNum = KeyPoint.fourPoint(point,point1ImgNp)
	targetNum = targetNum//2
	# 原图上的四个点
	pointSrc = [[fourPoints[0][0], fourPoints[0][1]], [fourPoints[1][0], fourPoints[1][1]],
				[fourPoints[3][0], fourPoints[3][1]],[fourPoints[2][0], fourPoints[2][1]]]
	# pointSrc = [[413,655],[1524,668],[1880,932],[55,915]]
	# 目标图中的个点
	# pointDst = [[0, 0], [1000, 0], [0, 600], [1000, 600]]
	srcPoints = np.float32(pointSrc)
	dstPoints = np.float32(pointDst)
	# 1、求解变换矩阵M
	m = cv2.getPerspectiveTransform(srcPoints, dstPoints)
	# mInv = cv2.getPerspectiveTransform(dstPoints, srcPoints)
	# 2、求解鸟瞰图

	resultImg = cv2.warpPerspective(img, m, (int(dstPoints[1][0]), int(dstPoints[2][1])), flags=cv2.INTER_AREA, borderMode=None)

	cv2.imshow("change_view", resultImg)
	cv2.waitKey(0)
	# print(point1ImgNp[0])

	#求解其在鸟瞰图上的对应坐标
	pointPersperct = cv2.perspectiveTransform(point1ImgNp[np.newaxis],m)[0]

	# print("point2Dst%d"%point2Dst)

	cv2.imwrite("E:\Desktop/test/1/result/resultImg.jpg", resultImg)

	return pointPersperct,targetNum



def findSamllest(arr):
	if (len(arr) == 0):
		print("数据为空")
		return 0, 0
	smallest = 0
	smallest_index = 0



	for i in range(1, len(arr) - 1):
		if arr[i] >= arr[i + 1] and arr[i] >= arr[i - 1] and arr[i] > 700:
			smallest = arr[i]
			smallest_index = i
			break
	return smallest, smallest_index
