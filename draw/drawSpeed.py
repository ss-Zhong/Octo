import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'serif'  # 选择字体家族
plt.rcParams['font.serif'] = 'Times New Roman'  # 指定字体
plt.rcParams['font.size'] = 12  # 指定字号

categories = ['VGG11', 'AlexNet']
color=['sandybrown', 'limegreen', 'cornflowerblue', 'darkorange', 'forestgreen', 'royalblue']
data = np.array([[287, 148, 155, 352, 162, 172],
                 [123, 75, 81, 144, 82, 88]])
data = np.array([50000, 60000]) // np.transpose(data)


# 创建图形
fig, ax = plt.subplots()

# 绘制柱状图
bar_width = 0.3
bar_start = 2 * np.arange(len(categories))
for i in range(6):
    bar_positions = bar_start + i * bar_width

    # 第一大块的柱状图
    bars = ax.bar(bar_positions, data[i], bar_width, label='AlexNet', color=color[i], edgecolor='black')
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

# 设置坐标轴标签
ax.set_xticks(bar_start + bar_width *2.5)
ax.set_xticklabels(categories)
ax.set_ylabel('Speed(images/sec)')

# 显示图例
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(["CIFAR-10, FP32","CIFAR-10, Symm", "CIFAR-10, Asymm+PRC", 
           "Fashion MNIST, FP32","Fashion MNIST, Symm", "Fashion MNIST, Asymm+PRC"],frameon=False, fontsize = 9)

# 显示图形
# plt.show()
plt.savefig(f'./result/img/speed.svg', format='svg')
