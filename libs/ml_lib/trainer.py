import csv
from tqdm import tqdm
from . import *

import time

def one_hot_encode(labels, class_num):
    batch_size = len(labels)
    one_hot_labels = mypy.zeros((batch_size, class_num))
    one_hot_labels[mypy.arange(batch_size), labels] = 1
    return one_hot_labels

class Trainer:
    def __init__(self, model, dataset, batch_size, epoch_num, class_num, csvWriter = None, save_file = None):
        self.model = model
        (self.train_images, labelTr), (self.test_images, self.test_labels) = dataset
        self.train_labels = one_hot_encode(labelTr, class_num)
        self.batch_size = batch_size
        self.epoch_num = epoch_num

        self.csvWriter = csvWriter
        self.save_file = save_file

    def train(self):
        num_batches = len(self.train_labels) // self.batch_size

        if self.csvWriter is not None:
            acc = self.accuracy(self.batch_size)
            print("acc: ", acc)
            self.csvWriter.writerow([0, 0, 0, acc])

        for i in range(self.epoch_num):
            for batch_index in range(num_batches):
                s = time.perf_counter()

                start_index = batch_index * self.batch_size
                end_index = start_index + self.batch_size

                batch_images = self.train_images[start_index:end_index]
                batch_labels = self.train_labels[start_index:end_index]

                loss = self.model.forward(batch_images, batch_labels)
                self.model.backward()

                e = time.perf_counter()
                print(f"[E:{i+1}, B:{batch_index+1}, T:{format(e-s, '.4f')}s]\t", loss)
                
                # 记录实验数据
                if batch_index < num_batches - 1 and self.csvWriter is not None:
                        self.csvWriter.writerow([i+1, batch_index+1, loss, ''])
                elif batch_index == num_batches - 1:
                    acc = self.accuracy(self.batch_size)
                    print("acc: ", acc)
                    if self.csvWriter is not None:
                        self.csvWriter.writerow([i+1, batch_index+1, loss, acc])

        if self.save_file:
            self.model.save_params(self.save_file)
                
    def accuracy(self, batch_size):
        num_batches = len(self.test_labels) // batch_size
        acc = 0.0

        for batch_index in tqdm(range(num_batches)):
            start_index = batch_index * self.batch_size
            end_index = start_index + self.batch_size

            batch_images = self.test_images[start_index:end_index]
            batch_labels = self.test_labels[start_index:end_index]

            y = self.model.predict(batch_images)
            y = mypy.argmax(y, axis=1)
            acc += mypy.sum(y == batch_labels)
        
        return acc / (len(self.test_labels) - len(self.test_labels) % batch_size)
