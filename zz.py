import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. 准备数据
data = {
    'Dataset': ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'ECL', 'Weather', 'Exchange'],
    'our': [0.432, 0.379, 0.371, 0.275, 0.169, 0.243, 0.354],
    'S-Mamba': [0.455, 0.381, 0.398, 0.288, 0.170, 0.251, 0.367],
    'TimeMixer': [0.459, 0.390, 0.381, 0.277, 0.182, 0.245, 0.369],
    'iTransformer': [0.454, 0.383, 0.407, 0.288, 0.178, 0.258, 0.360],
    'PatchTST': [0.446, 0.387, 0.387, 0.281, 0.205, 0.259, 0.367],
    'TimesNET': [0.458, 0.414, 0.400, 0.291, 0.192, 0.259, 0.416],
    'Dlinear': [0.456, 0.399, 0.403, 0.386, 0.212, 0.256, 0.354],
}

df = pd.DataFrame(data)

# 2. 设置绘图参数（已调整颜色）
methods = ['our', 'PatchTST', 'Dlinear', 'iTransformer',  'S-Mamba', 'TimeMixer', ]

# 修改了这里的颜色，使用了对比度更清晰的色系
colors = {
    'our': '#ff0000',      # 红色 (保持醒目)
    'PatchTST': '#1F77B4', # 蓝色
    'Dlinear': '#FF7F0E',  # 橙色
    'iTransformer': '#2CA02C', # 绿色
    
    'S-Mamba': '#8C564B',    # 棕色
    'TimeMixer': '#E377C2',    # 粉色
    
}

# 线条样式
line_styles = {
    'our': {'width': 4, 'zorder': 10}, # 我们的方法最粗、最上层
    'iTransformer': {'width':3, 'zorder': 6},
    'PatchTST': {'width':3, 'zorder': 5},
    'Dlinear': {'width':3, 'zorder': 4},
    'S-Mamba': {'width': 3,'zorder': 2},
    'TimeMixer': {'width':3, 'zorder': 1}, 
   
}

# 3. 数据处理：反向归一化 (外小里大)
df_norm = df.copy()
for i, row in df.iterrows():
    vals = [row[m] for m in methods]
    min_val, max_val = min(vals), max(vals)
    for m in methods:
        val = row[m]
        # 归一化：最小值 -> 1.0 (最外圈), 最大值 -> 0.1 (接近圆心)
        norm = 1 - (val - min_val) / ((max_val - min_val) if max_val != min_val else 1)
        df_norm.at[i, m] = 0.1 + 0.9 * norm

# 4. 绘图
categories = df['Dataset'].tolist()
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1] # 闭合圆环

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

# 绘制数据线
for method in methods:
    values = df_norm[method].tolist()
    values += values[:1]
    color = colors.get(method, 'grey')
    style = line_styles.get(method, {'width': 1, 'zorder': 1})
    ax.plot(angles, values, linewidth=style['width'], linestyle='solid', label=method, color=color, zorder=style['zorder'])




ax.set_theta_offset(np.pi / 2) # 设置起始角度
ax.set_theta_direction(-1) # 顺时针
plt.xticks(angles[:-1], categories, color='black', size=13, weight='bold') # 维度标签(ETTh1等)保留
# pad 参数控制距离，数值越大距离越远
ax.tick_params(axis='x', pad=100)
ax.set_rlabel_position(0)
plt.yticks([], []) # 隐藏径向刻度
ax.spines['polar'].set_visible(False) # 隐藏外圈圆
ax.grid(False) # 隐藏默认圆形网格

# 手动绘制多边形网格线
grid_levels = [0.2, 0.4, 0.6, 0.8, 1.0]
for r in grid_levels:
    for i in range(N):
        ax.plot([angles[i], angles[i+1]], [r, r], color='grey', linewidth=0.5, zorder=0)

# 绘制轴线
for ang in angles[:-1]:
    ax.plot([ang, ang], [0, 1], color='grey', linewidth=0.5, linestyle='--')

# 图例
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4, fancybox=True, shadow=True, prop={'weight':'bold', 'size': 12})

plt.tight_layout()
plt.show()