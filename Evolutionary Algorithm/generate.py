DNA_SIZE =  55#避难点 DNA序列     
POP_SIZE = 100          
CROSS_RATE = 0.8         
MUTATION_RATE = 0.1    
N_GENERATIONS = 2000
S = df1['S'].dropna(axis=0).values #居住面积
D = df2.values #距离矩阵
P = df1['P'].values # 每个居住地人口数
Dmax = df1['Dmax'].values
B_toSub = []
def solve(Y):
	# Y -->[55,1]
	res = []
	for i in range(243):
		b = np.zeros((1,55))
		while np.dot(b,Y)[0]!=1:
			b = np.zeros((1,55))
			ind = np.random.randint(0,55)
			b[:,ind] =1
		res.append(b[0])
		# b[0] -->[55,1]
	B = np.array(res)
	# B-->[243,55]
	return B
def generate(pop):
	# 这样计算出B实际上在变向满足st03
	Tag = False
	while not Tag:
		# pop = np.random.randint(2,size = (100,55))
		B_copy = B1
		B = solve(item.T)
		print('----------矩阵B改变？',(B==B_copy).all())
		B_copy = B.copy()
		#计算当前方案55个避难所的容量要求
		capacity_need = np.dot(P.T,B)
		capacity = item*S
		st1 = np.sum((capacity_need.reshape((1,55))-capacity)>0)
		# 计算限制条件2，距离条件
		dist_sum = np.sum(B*D,axis=1)		
		st2 = np.sum((dist_sum-Dmax)>0)
		if st1  == 0 and st2  == 0:
			print('找到满足的条件的',B)
			B_toSub.append((item,B))
			Tag =  True
		else:
			print('不合格！',st1,st2)
			continue