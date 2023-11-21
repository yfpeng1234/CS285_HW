'''
I save the results of the two tasks in DAgger in a single .xlsx file.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# read Excel
file_path = 'data.xlsx'
df = pd.read_excel(file_path)

# to numpy array
data= df.to_numpy()
data[2,:]=np.ones_like(data[2,:])*data[2,0]
data[3,:]=np.ones_like(data[3,:])*data[3,0]
data[6,:]=np.ones_like(data[6,:])*data[6,0]
data[7,:]=np.ones_like(data[7,:])*data[7,0]

x=np.arange(data.shape[-1])+1

plt.plot(x, data[0,:], label='ours policy', color='blue')

# 绘制第二条数据线，使用橙色
plt.plot(x, data[2,:], label='expert policy', color='orange')

# 给每条线添加标准差的阴影区域
plt.fill_between(x, data[0,:] - data[1,:], data[0,:] + data[1,:], color='blue', alpha=0.2)
plt.fill_between(x, data[2,:] - data[3,:], data[2,:] + data[3,:], color='orange', alpha=0.2)

# 添加图例
plt.legend()

# 添加标题和标签（可选）
plt.title('Learning Curve in Ant-v4 task')
plt.xlabel('iteration')
plt.ylabel('reward')

plt.savefig('ant.png')

plt.clf()

plt.plot(x, data[4,:], label='ours policy', color='blue')

# 绘制第二条数据线，使用橙色
plt.plot(x, data[6,:], label='expert policy', color='orange')

# 给每条线添加标准差的阴影区域
plt.fill_between(x, data[4,:] - data[5,:], data[4,:] + data[5,:], color='blue', alpha=0.2)
plt.fill_between(x, data[6,:] - data[7,:], data[6,:] + data[7,:], color='orange', alpha=0.2)

# 添加图例
plt.legend()

# 添加标题和标签（可选）
plt.title('Learning Curve in Hopper-v4 task')
plt.xlabel('iteration')
plt.ylabel('reward')

plt.savefig('hopper.png')
