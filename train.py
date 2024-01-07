from libs.datasets_lib import cifar10
from models.vgg11 import *
from libs.ml_lib.trainer import *
from datetime import datetime
import time

'''
Parameter Setting
'''
# train setting
batch_size = 64
epoch_num = 20
lr = 0.1

# model setting
model = VGG11(optimizer=optimizer.SGD(lr=lr))
dataset=cifar10.loadCIFAR10()
num_classes = 1000

# experiment setting
date_time = datetime.now().strftime("%Y%m%d_%H%M")
csvPath = 'F:\\Programs\\Octo\\Octo\\exp_result\\' + date_time + '.csv'


'''
Train
'''
if __name__ == '__main__':
    # write csv
    csvFile = open(csvPath, mode='w', newline='')
    csvWriter = csv.writer(csvFile)

    # 开始计时
    start = time.perf_counter() 

    # train begin
    trainer = Trainer(model=model, dataset=dataset, num_classes=num_classes, 
                      batch_size=batch_size, epoch_num=epoch_num,
                      csvWriter = csvWriter)
    trainer.train()

    # 结束计时
    end = time.perf_counter() 
    print(end-start)

    csvFile.close()