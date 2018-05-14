import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
data = pd.DataFrame(np.random.randn(1000,4),index = np.arange(1000),columns = list('ABCD')) 
data = data.cumsum()
ax = data.plot.scatter(x = 'A',y='B',color = 'blue')
data.plot.scatter(x = 'A',y = 'C',color = 'red',ax = ax)
data.plot.scatter(x = 'A',y = 'D',color='yellow',ax=ax)
ax2 = data.plot()
plt.show()