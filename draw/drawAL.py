import matplotlib.pyplot as plt
import csv

# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'serif'  # 选择字体家族
plt.rcParams['font.serif'] = 'Times New Roman'  # 指定字体
plt.rcParams['font.size'] = 12  # 指定字号

def getLA(csvPath):
    losses, accuracies = [], []
    with open(csvPath, 'r') as file:
        csv_reader = csv.DictReader(file)
        for i, row in enumerate(csv_reader):
            if row['Epoch'] != '0' and row['Loss'] != '' and i % 50 == 0:
                losses.append(float(row['Loss']))
            if row['Accuracy']:
                accuracies.append(float(row['Accuracy']))
    
    return losses, accuracies

if __name__ == '__main__':
    modelList = ["vgg", "alexnet"]
    datasetList = ["cifar10", "fashionmnist"]
    dict = {"vgg": "VGG11", "alexnet": "AlexNet", "cifar10": "CIFAR-10", "fashionmnist": "Fashion Mnist"}
    fig, axs = plt.subplots(2, 2, figsize=(13, 8))

    for i, model in enumerate(modelList):
        for j, dataset in enumerate(datasetList):
            Path = f"./result/save/{model}/{dataset}/"
            csvPath_asymm = Path + "Asymm.csv"
            csvPath_fp32 = Path + "FP32.csv"
            csvPath_symm = Path + "Symm.csv"

            losses_asymm, accuracies_asymm = getLA(csvPath_asymm)
            losses_fp32, accuracies_fp32 = getLA(csvPath_fp32)
            losses_symm, accuracies_symm = getLA(csvPath_symm)
            
            # 计算训练进度百分比
            training_progress = [i * 100 / 25 for i in range(26)]
            training_progress_loss = [(i+1) * 100 / len(losses_asymm) for i in range(len(losses_asymm))]

            # 绘制损失图
            axs[i, j].plot(training_progress_loss, losses_fp32, label="FP32",color='sandybrown', linewidth = 1)
            axs[i, j].plot(training_progress_loss, losses_symm, label="Symm",color='limegreen', linewidth = 1)
            axs[i, j].plot(training_progress_loss, losses_asymm, label="Asymm+PRC",color='cornflowerblue', linewidth = 1)
            axs[i, j].set_ylabel('Training Loss')
            axs[i, j].set_xlabel('Training Progress (%)')
            axs[i, j].set_ylim(0, 2.5)
            axs[i, j].set_xlim(0, 100)
            axs[i, j].set_title(f"{dict[model]}, {dict[dataset]}", fontsize=13, pad=10)
            axs[i, j].grid(True, linestyle='--', linewidth=0.5, color='#f2f2f2')

            # 创建第二个y轴用于绘制准确度
            ax_ = axs[i, j].twinx()
            ax_.set_ylabel('Accuracy')
            ax_.plot(training_progress, accuracies_fp32, label="FP32", color='sandybrown', linewidth = 1)
            ax_.plot(training_progress, accuracies_symm, label="Symm",color='limegreen', linewidth = 1)
            ax_.plot(training_progress, accuracies_asymm, label="Asymm+PRC", color='cornflowerblue', linewidth = 1)
            ax_.set_ylim(0, 1)

            plt.legend(["FP32","Symm", "Asymm+PRC"],loc="center right",frameon=False)
            plt.xticks([i * 10 for i in range(11)])  # 根据需要调整刻度
            

    # 显示图形
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    # plt.show()
    plt.savefig(f'./result/img/Trainingconvergence.svg', format='svg')