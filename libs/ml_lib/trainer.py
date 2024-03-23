import csv
from tqdm import tqdm
from . import *
import time

class Trainer:
    def __init__(self, model, dataset, batch_size, epoch_num, class_num, csvWriter = None, save_file = None):
        self.model = model
        (self.train_images, labelTr), (self.test_images, self.test_labels) = dataset
        self.train_labels = mypy.eye(class_num)[labelTr]
        self.batch_size = batch_size
        self.epoch_num = epoch_num

        self.csvWriter = csvWriter
        self.save_file = save_file

    def train(self):
        num_batches = len(self.train_labels) // self.batch_size

        if self.csvWriter is not None:
            acc = self.accuracy(self.batch_size)
            print(f"[E: 0] acc:{acc}")
            self.csvWriter.writerow([0, 0, 0, acc])

        for i in range(self.epoch_num):
            loss = 0.
            loss_E = 0.
            acc_trE = 0.
            for batch_index in tqdm(range(num_batches)):

                start_index = batch_index * self.batch_size
                end_index = start_index + self.batch_size

                batch_images = self.train_images[start_index:end_index]
                batch_labels = self.train_labels[start_index:end_index]

                acc_tr, loss = self.model.forward(batch_images, batch_labels)
                loss_E += loss
                acc_trE += acc_tr
                self.model.backward()
                
                # print(f"[E:{i+1}, B:{batch_index+1}]\t", loss)
                
                # 记录实验数据
                if batch_index < num_batches - 1 and self.csvWriter is not None:
                    self.csvWriter.writerow([i+1, batch_index+1, loss, ''])

            acc = self.accuracy(self.batch_size)
            loss_E /= num_batches
            acc_trE /= len(self.train_labels)
            print(f"[E: {i+1}] train-acc:{acc_trE} test-acc:{acc} avg-loss:{loss_E} loss:{loss}")
            if self.csvWriter is not None:
                self.csvWriter.writerow([i+1, num_batches, loss, acc])

        if self.save_file:
            self.model.save_params(self.save_file)
                
    def accuracy(self, batch_size):
        num_batches = len(self.test_labels) // batch_size
        acc = 0.
        loss = 0.

        for batch_index in tqdm(range(num_batches)):
            start_index = batch_index * self.batch_size
            end_index = start_index + self.batch_size

            batch_images = self.test_images[start_index:end_index]
            batch_labels = self.test_labels[start_index:end_index]

            y, l = self.model.predict(batch_images, mypy.eye(10)[batch_labels])
            if l:
                loss += l
            y = mypy.argmax(y, axis=1)
            acc += mypy.sum(y == batch_labels)

        print("test-loss:", loss / num_batches)
        
        return acc / (len(self.test_labels) - len(self.test_labels) % batch_size)
