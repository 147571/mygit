from cv2 import cv2
import numpy
from imutils import contours
import numpy as np
from scipy.spatial import distance as dist
from utils import myutils
import  time



def get_red(img):
	redImg = img[:, :, 2]
	return redImg


def get_green(img):
	greenImg = img[:, :, 1]
	return greenImg


def get_blue(img):
	blueImg = img[:, :, 0]
	return blueImg
	return blueImg

def zuobiaopaixu(a):
	b = []
	l = len(a)
	for i in range(l):
		j = i
		for j in range(l):
			if (a[i][0] < a[j][0]):
				a[i], a[j] = a[j], a[i]
			# if (a[i][1] < a[j][1]):
			# 	a[i], a[j] = a[j], a[i]

	for k in range(len(a)):
		b.append(a[k])
	return b
class KeyPoint:
	def imgPretreat(img):

		# img = cv2.imread("E:\Desktop/test/1/redBall/1.jpg")
		b, g, r = cv2.split(img)
		# cv2.imshow("Red 1", r)
		# 二值化
		ret, bin = cv2.threshold(r, 217, 250, cv2.THRESH_BINARY)
		# cv2.namedWindow('bin', 0)
		# cv2.resizeWindow('bin', 640, 480)
		# cv2.imshow("bin", bin)
		# 中值滤波
		blur = cv2.medianBlur(bin, 3)
		# cv2.namedWindow('blur', 0)
		# cv2.resizeWindow('blur', 640, 480)
		# cv2.imshow("blur", blur)
		# 形态学开运算
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
		open = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel)
		cv2.namedWindow('open', 0)
		cv2.resizeWindow('open', 640, 480)
		cv2.imshow("open", open)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

		return open
	#找到所有靶标中点
	def centroidCoordinates(img,open,target=True):


		countours, _ = cv2.findContours(open.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
		# t1 = time.time()
		# countours = myutils.sort_contours(countours, method="left-to-right")[0]
		# digits = {}
		img1 = img.copy()
		# img2 = img1.copy()
		locs = []


		# t2 = time.time()
		if target:
			for (i, c) in enumerate(countours):
				# 计算矩形
				(x, y, w, h) = cv2.boundingRect(c)
				M = cv2.moments(c)
				cx = M['m10'] / M['m00']
				cy = M['m01'] / M['m00']
				area = cv2.contourArea(c)
				# cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 0, 255), 1)

				ar = w / float(h)
				# cv2.putText(img1, 'w:{} h:{} a:{}'.format(int(w), int(h), int(area)),
				# 			(x - 10, y - 5),
				# 			cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
				# 选择合适的区域，根据实际任务来，这里的基本都是四个数字一组
				if ar > 1.6 and ar < 3.5 and cy>600:

					if (w > 22 and w < 55) and (h > 8 and h < 25):
						# 符合的留下来
						# locs.append((x, y, w, h))
						# tx = x + w/2
						# ty = y + h/2

						cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 0, 255), 1)
						cv2.putText(img1, 'cx:{} cy:{}'.format(int(cx), int(cy)),
									(x - 10, y - 5),
									cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
						cv2.putText(img1, 'w:{} h:{} a:{}'.format(int(w), int(h), int(area)),
									(x - 10, y - 35),
									cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
						# cv2.putText(img2, 'tx:{} ty:{}'.format(int(tx), int(ty)), (x + 5, y - 5),
						# 			cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
						cv2.circle(img1, (int(cx), int(cy)), 1, (0, 0, 255), 2)
						# cv2.circle(img2,(int(tx),int(ty)),1,(0,0,255),2)
						locs.append((cx, cy))
				# cv2.drawContours(img1, c, 3, (0, 0, 255), 3)
			t3 = time.time()
			locs = zuobiaopaixu(locs)
			# print("locs:"%locs)
			cv2.imwrite("E:\Desktop/test/1/keypoint.jpg",img1)
		else:
			for (i, c) in enumerate(countours):
				# 计算矩形
				(x, y, w, h) = cv2.boundingRect(c)
				M = cv2.moments(c)
				cx = M['m10'] / M['m00']
				cy = M['m01'] / M['m00']
				area = cv2.contourArea(c)
				# cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 0, 255), 1)
				ar = w / float(h)
				# cv2.putText(img1, 'w:{} h:{} a:{}'.format(int(w), int(h), int(area)),
				# 			(x - 10, y - 5),
				# 			cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
				# 选择合适的区域，根据实际任务来，这里的基本都是四个数字一组
				if ar > 1 and ar < 2 and cy > 600:

					if (w > 25 and w < 50) and (h > 22 and h < 40):
						# 符合的留下来
						# locs.append((x, y, w, h))
						# tx = x + w/2
						# ty = y + h/2

						cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 0, 255), 1)
						cv2.putText(img1, 'cx:{} cy:{}'.format(int(cx), int(cy)),
									(x - 10, y - 5),
									cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
						# cv2.putText(img1, 'w:{} h:{} a:{}'.format(int(w), int(h), int(area)),
						# 			(x - 10, y - 35),
						# 			cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
						# cv2.putText(img2, 'tx:{} ty:{}'.format(int(tx), int(ty)), (x + 5, y - 5),
						# 			cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
						cv2.circle(img1, (int(x+w/2), int(y+h)), 1, (0, 0, 255), 2)
						# cv2.circle(img2,(int(tx),int(ty)),1,(0,0,255),2)
						locs.append((x+w/2, y+h))
			cv2.imwrite("E:\Desktop/test/1/keypoint1.jpg", img1)
			# print("locs:"%locs)


		# print("time:%.10f,%.10f" % (t2 - t1, t3 - t2))
		# cv2.imshow("coun", img1)
		# cv2.waitKey(0)
		cv2.destroyAllWindows()
		return locs

	# 针对落点坐标 找离最近的四个靶点坐标
	def fourPoint(targetPoint, ballPoint):
		X_train = np.array(targetPoint)
		distance1 = []
		targetNum = 0
		for i in range(len(X_train)):
			if X_train[i][0] < ballPoint[0][0]:
				targetNum+=1
				# top1.append(X_train[i])
			distance = dist.euclidean(ballPoint, X_train[i])

			# distance = abs(X_train[i] - ballPoint)
			# xAddy = distance[0][0] + distance[0][1]

			distance1.append((distance))

		# distance1.append((distance))
		# distance1 = sorted(distance1, key=lambda x: (x[0]))
		##np.argsort()返回样本点的索引位置 即得到距离最近的四个坐标
		sort = np.argsort(distance1)
		# top1 = []
		# top2 = []
		# for i in sort:
		# 	# print(X_train[i][0])
		# 	if X_train[i][1] > ballPoint[0][1]:
		# 		top1.append(X_train[i])
		#
		# 		# top1 = [X_train[i] for i in sort[:2]]
		# 	else:
		# 		top2.append(X_train[i])
		#
		# 		# top2 = [X_train[i] for i in sort[:2]]
		# top1 = top1[:2]
		# top2 = top2[:2]
		# top = np.vstack((top1, top2))
		top1 = []
		top2 = []
		top3 = []
		top4 = []
		for i in sort:
			# print(X_train[i][0])
			# 左上
			if X_train[i][1] < ballPoint[0][1] and X_train[i][0] < ballPoint[0][0]:
				top1.append(X_train[i])
			elif X_train[i][1] > ballPoint[0][1] and X_train[i][0] < ballPoint[0][0]:
				top2.append(X_train[i])
			elif X_train[i][1] > ballPoint[0][1] and X_train[i][0] > ballPoint[0][0]:
				top3.append(X_train[i])
			elif X_train[i][1] < ballPoint[0][1] and X_train[i][0] > ballPoint[0][0]:
				top4.append(X_train[i])

		# top1 = [X_train[i] for i in sort[:2]]

		# top2 = [X_train[i] for i in sort[:2]]

		top1 = top1[:1]
		top2 = top2[:1]
		top3 = top3[:1]
		top4 = top4[:1]

		top = top1 + top2 + top3 + top4
		top = sorted(top, key=lambda x: (x[1],x[0]))

		# top=[distance2[i] for  i in sort[:4]]

		# 找投掷点左侧有几个靶点坐标

		return top,targetNum




#
if __name__ == '__main__':
	keypoint = KeyPoint
	img = cv2.imread("E:\Desktop/test/1/redBall/347.jpg")
	open = keypoint.imgPretreat(img)
	locs = keypoint.centroidCoordinates(img,open,True)
	print(locs)

