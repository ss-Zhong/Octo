import matplotlib.pyplot as plt
import csv

csvPath = 'result/data/saved/20240115_1139.csv'

# 从CSV文件中读取Loss和Accuracy数据
losses = []
accuracies = []

with open(csvPath, 'r') as file:
    csv_reader = csv.DictReader(file)
    
    for row in csv_reader:
        # 将Loss和Accuracy数据添加到列表中
        if row['Epoch'] != '0' and row['Loss'] != '':
            losses.append(float(row['Loss']))
        if row['Accuracy']:
            accuracies.append(float(row['Accuracy']))

# 创建图形对象和子图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

# 绘制损失图
ax1.plot(losses, label='Loss', color='blue')
ax1.set_ylabel('Loss')
ax1.set_xlabel('Batch')
ax1.legend()
ax1.set_ylim(0, None)

# 绘制准确度图
ax2.plot(accuracies, label='Accuracy', color='green')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()

ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax2.set_xlim(0, 19)
ax2.set_ylim(0, 1)
for i, acc in enumerate(accuracies):
    ax2.annotate(f'{acc:.2f}', (i, acc), textcoords="offset points", xytext=(0,10), ha='center')

# 调整子图之间的间距
plt.tight_layout()

# 显示图形
plt.show()