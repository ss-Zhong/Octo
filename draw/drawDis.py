import pickle
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

file_name = f"./result/save/vgg/cifar10/FP32.pkl"

with open(file_name, 'rb') as f:
    params = pickle.load(f)
    for key in params.keys():
        print(key)

    values = params['Conv_b_22'].get().flatten()

    # 创建分布图
    sns.histplot(values, kde=True, color='skyblue')

    # 设置坐标轴标签和标题
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Distribution')

    # 显示图形
    plt.show()
