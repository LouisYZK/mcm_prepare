# 多目标遗传算法模型
import numpy as np 
import pandas as pd 
#df = pd.read_excel('模型参数.xls',header =0 )
# print(df.head())
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
def F1(pop):
	# pop --> [100,55]
	# S -->[55,1]
	return np.dot(pop,S)
def F2(pop):
	Dist = []
	for j in range(POP_SIZE):
		B = solve(pop[j,:])
		dist = np.sum(B*D)
		Dist.append(dist)
	return np.array(Dist)
def get_fitness(v1,v2,pop):
	# v1/2 -->[100,]
	# 计算基于帕累托原理的适应度
	nq = []
	for x in range(POP_SIZE):
		compare = 0
		count = 0 
		for xx in range(POP_SIZE):
			if v1[x] < v1[xx]:
				compare +=1
			if v1[x]>v1[xx]:
				compare  = -100
			if v2[x] < v2[xx]:
				compare  +=1
			if v2[x]>v2[xx]:
				compare = -100
			if compare >=1:
				count = count+1
		if isSubjectTo(pop[x,:]):
			nq.append(count+1+80)
		else:
			nq.append(1)
	return nq
def select(pop,fitness):
	idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,p=fitness/fitness.sum())
	return pop[idx]
def crossover(parent, pop):     # mating process (genes crossover)
    # 判断族群是否发生交配
    if np.random.rand() < CROSS_RATE:
        # 在pop中随机挑选与parent交配的个体
        i_ = np.random.randint(0, POP_SIZE, size=1)                             # select another individual from pop
        # 随机生成基因序列中发生交配的基因位置
        cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool)   # choose crossover points
        # 替换基因位
        parent[cross_points] = pop[i_, cross_points]                            # mating and produce one child
    return parent
def mutate(child):
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            child[point] = 1 if child[point] == 0 else 0
    return child
# 解矩阵方程，化两参数为一个参数
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
def isSubjectTo(item):
	# 这样计算出B实际上在变向满足st03
	Tag = False
	while not Tag:
		B = solve(item.T)
		#计算当前方案55个避难所的容量要求
		capacity_need = np.dot(P.T,B)
		capacity = item*S
		st1 = np.sum((capacity_need.reshape((1,55))-capacity)>0)
		# 计算限制条件2，距离条件
		dist_sum = np.sum(B*D,axis=1)		
		st2 = np.sum((dist_sum-Dmax)>0)
		if st1  == 0 and st2  == 0:
			print('找到满足的条件的',item)
			B_toSub.append(item)
			Tag =  True
		else:
			print('不合格！',st1,st2)
			continue
pop = np.random.randint(2,size = (POP_SIZE,243,DNA_SIZE))
for index in range(POP_SIZE):

for _ in range(N_GENERATIONS):
	v1  = F1(pop)
	v2 = F2(pop)
	fitness = np.array(get_fitness(v1,v2,pop))
	# print("Most fitted DNA: ", pop[np.argmax(fitness), :])
	pop = select(pop,fitness)
	pop_copy = pop.copy()
	for parent in pop:
		child = crossover(parent,pop_copy)
		child = mutate(child)
		parent[:] = child
	print("已经循环",_,'轮次')
