import numpy  as  np
import matplotlib
from scipy.spatial import distance as dist

##有了x,y坐标就可以绘制一张散点图
matplotlib.use('TkAgg')
import matplotlib.pyplot  as  plt


# , X_train[i][0], X_train[i][1]
# 针对落点坐标 找离最近的四个靶点坐标
def fourPoint(targetPoint, ballPoint):
	X_train = np.array(targetPoint)
	distance1 = []
	targetNum = 0
	for i in range(len(X_sample)):
		if X_train[i][0] < ballPoint[0]:
			targetNum += 1
		# distance = dist.euclidean(ballPoint, X_train[i])
		distance = abs(X_train[i] - ballPoint)
		xAddy = distance[0] + distance[1]

		distance1.append((xAddy))


		# distance1 = sorted(distance1, key=lambda x: (x[0]))
		##np.argsort()返回样本点的索引位置 即得到距离最近的四个坐标
	sort = np.argsort(distance1)
	top1 = []
	top2 = []
	top3 = []
	top4 = []
	for i in sort:
		print(X_train[i][0])
		# 左上
		if X_train[i][1] < ballPoint[1] and X_train[i][0] < ballPoint[0]:
			top1.append(X_train[i])
		elif X_train[i][1] > ballPoint[1] and X_train[i][0] < ballPoint[0]:
			top2.append(X_train[i])
		elif X_train[i][1] > ballPoint[1] and X_train[i][0] > ballPoint[0]:
			top3.append(X_train[i])
		elif X_train[i][1] < ballPoint[1] and X_train[i][0] > ballPoint[0]:
			top4.append(X_train[i])



			# top1 = [X_train[i] for i in sort[:2]]

			# top2 = [X_train[i] for i in sort[:2]]

	top1 = top1[:1]
	top2 = top2[:1]
	top3 = top3[:1]
	top4 = top4[:1]

	top = top1+top2+top3+top4
	top = sorted(top, key=lambda x: (x[1], x[0]))
	if(targetNum%2):
		targetNum = targetNum//2+1
	else:
		targetNum = targetNum/2
	print(targetNum)
	# top=[distance2[i] for  i in sort[:4]]
	return top

if __name__ == "__main__":
	X_sample = [[54, 914],
				[411, 655],
				[972.7, 661],
				[980, 920],
				[1522, 668],
				[1879, 933]]
	X_train = np.array(X_sample)
	x_test = np.array([972.50,765])

	fourpoint = fourPoint(X_train, x_test)

	print(fourpoint)
