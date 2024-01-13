from libs.datasets_lib import cifar10
from libs.ml_lib import optimizer
from models.vgg11 import *
from libs.ml_lib.trainer import *
from datetime import datetime
import time

'''
Parameter Setting
'''
# train setting
batch_size = 32
epoch_num = 20
lr = 0.01
quant_mode = QuantMode.FullPrecision

# model setting
model = VGG11(optimizer=optimizer.SGD(lr=lr), quant_mode = quant_mode)
dataset = cifar10.loadCIFAR10(normalize=True)
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
    csvWriter.writerow(['Epoch', 'Batch', 'Loss', 'Accuracy'])

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
    csvWriter.writerow(end-start)
    csvFile.close()