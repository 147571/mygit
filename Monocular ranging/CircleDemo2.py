from cv2 import cv2
import matplotlib
import numpy as np
import os

from utils.keyPoint import KeyPoint
from utils.PerspectiveTransform import changeView
import argparse

keyPoint = KeyPoint



matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def save_image(image, addr, num):
	address = addr + str(num) + '.jpg'
	cv2.imwrite(address, image)


def findSamllest(arr):
	smallest = arr[0]
	smallest_index = 0
	for i in range(1, len(arr)):
		if arr[i] < smallest:
			smallest = arr[i]
			smallest_index = i
	return smallest, smallest_index



def main(file_pathname):
	# 对球落点坐标追踪
	for filename in os.listdir(file_pathname):
		lenth = filename.split('.')
		lenthDistance = int(lenth[0])

		print(filename)
		img = cv2.imread(file_pathname + '/' + filename)
		sp = img.shape
		i = 0
		print("图像分辨率:{:d},{:d}".format(sp[1], sp[0]))

		# 真实场景长宽cm
		w, h = 200, 200
		pointDst = [[0, 0], [800, 0], [800, 800], [0, 800]]
		# 球的落点
		srcImg = keyPoint.imgPretreat(img)  # srcImg是形态学处理后的二值图像

		# 得到排好序的角点
		point = keyPoint.centroidCoordinates(img, srcImg, False)
		point1ImgNp = []
		point1ImgNp.append([point[0][0], point[0][1]])
		point1ImgNp = np.float32(point1ImgNp)
		print("变换前落点({:.2f},{:.2f})".format(point1ImgNp[0][0],point1ImgNp[0][1]))
		# 落点经过透视变换的坐标
		pointPersperct,targetNum = changeView(img, pointDst, point1ImgNp)
		print("变换后({:.2f},{:.2f})".format(pointPersperct[0][0], pointPersperct[0][1]))

		print("透视变换后图像大小:(%d,%d)" % (pointDst[1][0], pointDst[3][1]))
		hx, hy = w / pointDst[1][0], h / pointDst[3][1]
		# 图像中点1,2 的物理坐标
		distanceReal = lenthDistance
		distanceTest = [pointPersperct[0][0] * hx, pointPersperct[0][1] * hy]

		erro = abs(distanceTest[0]+ 200 *(targetNum-1) - distanceReal) / distanceReal
		print("测量x:%.4fcm" % (distanceTest[0]+ 200 *(targetNum-1)))
		print("真实值x%.2fcm" % distanceReal)
		print("误差%.2f%%" % (erro * 100))


if __name__ == '__main__':
	# 设置参数
	ap = argparse.ArgumentParser()
	ap.add_argument('--source', type=str, default='data/image', help='source')  # file/folder, 0 for webcam

	args = ap.parse_args()
	source = args.source
	main(source)
