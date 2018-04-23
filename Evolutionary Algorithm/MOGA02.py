# 多目标遗传算法模型
# 尝试从矩阵B为初始化集群，看是否能加快收敛速度
import numpy as np 
import pandas as pd 
df1 = pd.read_excel('模型参数.xls',sheetname='Sheet1',header =0 )
df2 = pd.read_excel('模型参数.xls',sheetname='Dij',header=None)
print(df.head())
DNA_SIZE =  55#避难点 DNA序列   
POP_SIZE = 1000          
CROSS_RATE = 0.5         
MUTATION_RATE = 0.5    
N_GENERATIONS = 1000
S = df1['S'].dropna(axis=0).values #居住面积
D = df2.values #距离矩阵
P = df1['P'].values # 每个居住地人口数
Dmax = df1['Dmax'].values
B_toSub = []
def F1(pop):
	# pop --> [100,243,55]
	YY = []
	for ii in range(len(pop)):
		YY.append(solve_Y(pop[ii]))
	# S -->[55,1]
	Y = np.array(YY) 
	# Y -->[POP_size,55]
	return np.dot(Y,S)
def F2(pop):
	Dist = []
	for j in range(len(pop)):
		B = pop[j]
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
				continue
			if v2[x] < v2[xx]:
				compare  +=1
			if v2[x]>v2[xx]:
				continue
			if compare >=1:
				count = count+1
		fitness_add = int(POP_SIZE*0.8)
		if isSubjectTo(pop[x]):
			nq.append(fitness_add)
		else: 
			nq.append(count+1)
	return nq
def select(pop,fitness):
	global POP_SIZE
	idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,p=fitness/fitness.sum())
	return pop[idx]
def crossover(parent, pop):     
    # 判断族群是否发生交配
    if np.random.rand() < CROSS_RATE :
    	# 在pop中随机挑选与parent交配的个体
	    i_ = np.random.randint(0, POP_SIZE, size=1)                            
	    # 随机生成基因序列中发生交配的基因位置
	    cross_points = np.random.randint(0, 2, size=243).astype(np.bool)
	    # 替换基因位
	    parent[cross_points] = pop[i_, cross_points,:]	
    return parent

def mutate(child):
	if np.random.rand() < MUTATION_RATE:
		for point in range(243):
		    if np.random.rand() < MUTATION_RATE:
		        child[point,:] = np.zeros((1,55))
		        dna_point = np.random.choice(np.arange(DNA_SIZE), size=1, replace=True,p=S/S.sum())
		        child[point,dna_point] = 1
	return child
def solve_Y(B):
	# 解出方程BY=E的解Y,其实Y很好解，只要B某列存在不为0的值，那么Y这一列就是1
	# Y-->[55,]
	Y = np.zeros((55))
	B_sum = B.sum(axis=0)
	Y[B_sum>0] = 1
	return Y

def isSubjectTo(B):
	#计算当前方案55个避难所的容量要求
	capacity_need = np.dot(P.T,B)
	Y = solve_Y(B)
	capacity = Y*S
	diff = capacity_need-capacity
	st1 = diff[diff>0].size
	# 计算限制条件2，距离条件
	dist_sum = np.sum(B*D,axis=1)	
	diff2 = dist_sum-Dmax
	st2 = diff2[diff2>0].size
	if st1  == 0 and st2  == 0:
		print('找到满足的条件的',Y)
		B_toSub.append(B)
		Tag =  True
	else:
		print(st1,st2)
		Tag = False
	return Tag
def init_pop():
	pop = np.zeros((POP_SIZE,243,DNA_SIZE))
	for index in range(POP_SIZE):
		for index_ in range(243):
			one_ind = np.random.randint(0,DNA_SIZE,1)
			pop[index,index_,one_ind] = 1
	return pop
pop = init_pop()
count_all = []
for _ in range(N_GENERATIONS):
	count = 0
	v1 = F1(pop)
	v2 = F2(pop)
	fitness = np.array(get_fitness(v1,v2,pop))
	# print("Most fitted DNA: ", pop[np.argmax(fitness), :])
	pop = select(pop,fitness)
	pop_copy = pop.copy()
	for parent in pop:
		if isSubjectTo(parent):
			count +=1
		child = crossover(parent,pop_copy)
		child = mutate(child)
		parent[:] = child
	count_all.append(count/POP_SIZE)
	print("已经循环",_,'轮次')
	if POP_SIZE > 100:
		POP_SIZE = POP_SIZE-1 
	else:
		POP_SIZE = 100
