# from keyPoint import KeyPoint
# from PerspectiveTransform import changeView

from cv2 import cv2
import numpy as np
import argparse
# 设置参数
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image")
# args = vars(ap.parse_args())
# keyPoint = KeyPoint


def imgPretreat(img):
	# img = cv2.imread("E:\Desktop/test/1/redBall/1.jpg")
	b, g, r = cv2.split(img)
	# cv2.imshow("Red 1", r)
	# 二值化
	ret, bin = cv2.threshold(r, 220, 240, cv2.THRESH_BINARY)
	cv2.namedWindow('bin', 0)
	cv2.resizeWindow('bin', 640, 480)
	cv2.imshow("bin", bin)
	# 中值滤波
	blur = cv2.medianBlur(bin, 3)
	cv2.namedWindow('blur', 0)
	cv2.resizeWindow('blur', 640, 480)
	cv2.imshow("blur", blur)
	# 形态学开运算
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
	open = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel)
	cv2.namedWindow('open', 0)
	cv2.resizeWindow('open', 640, 480)
	cv2.imshow("open", open)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	return open

if __name__ == "__main__":

	img = cv2.imread("E:\Desktop/test/1/redBall/347.jpg")
	open = imgPretreat(img)
	# 真实场景长宽cm
	# w, h = 400, 200
# 	# img = cv2.imread(args["image"])
# 	# pointDst = [[0, 0], [1600, 0], [1600, 800], [0, 800]]
# 	# # 球的落点
# 	# point1ImgNp = []
# 	# point1ImgNp.append([975, 762])
# 	# point1ImgNp = np.float32(point1ImgNp)
# 	# # 落点经过透视变换的坐标
# 	# pointPersperct = changeView(img, pointDst, point1ImgNp)
# 	# print(pointPersperct)
# 	# print("透视变换后图像大小:(%d,%d)" % (pointDst[1][0], pointDst[3][1]))
# 	# hx, hy = w / pointDst[1][0], h / pointDst[3][1]
# 	# # 图像中点1,2 的物理坐标
# 	# pointReal = (pointPersperct[0][0] * hx, pointPersperct[0][1] * hy)
# 	# print("测量x:%.4f 测量y:%.4f" % (pointReal[0], pointReal[1]))