import numpy as np
import csv
from tqdm import tqdm

def one_hot_encode(labels, num_classes):
    batch_size = len(labels)
    one_hot_labels = np.zeros((batch_size, num_classes))
    one_hot_labels[np.arange(batch_size), labels] = 1
    return one_hot_labels

class Trainer:
    def __init__(self, model, dataset, batch_size, epoch_num, num_classes, csvWriter = None):
        self.model = model
        (self.train_images, labelTr), (self.test_images, self.test_labels) = dataset
        self.train_labels = one_hot_encode(labelTr, num_classes)
        self.batch_size = batch_size
        self.epoch_num = epoch_num

        self.csvWriter = csvWriter
        if csvWriter is not None:
            self.csvWriter.writerow(['Epoch', 'Loss', 'Accuracy'])

    def train(self):
        num_batches = len(self.train_images) // self.batch_size

        for i in range(self.epoch_num):
            loss_E = 0.0
            for batch_index in range(num_batches):
                start_index = batch_index * self.batch_size
                end_index = start_index + self.batch_size

                batch_images = self.train_images[start_index:end_index]
                batch_labels = self.train_labels[start_index:end_index]

                loss = self.model.forward(batch_images, batch_labels)
                acc = self.accuracy()
                print("acc: ", acc)
                
                loss_E += loss
                self.model.backward()

                print(f"epoch: {i}, batch: {batch_index}", loss)

            acc = self.accuracy()
            print("acc: ", acc)

            # 记录实验数据
            if self.csvWriter is not None:
                self.csvWriter.writerow([i+1, loss_E / num_batches, acc])
                
    def accuracy(self, batch_size = 64):
        num_batches = len(self.test_images) // batch_size
        acc = 0.0

        for batch_index in tqdm(range(num_batches)):
            start_index = batch_index * self.batch_size
            end_index = start_index + self.batch_size

            batch_images = self.test_images[start_index:end_index]
            batch_labels = self.test_labels[start_index:end_index]

            y = self.model.predict(batch_images)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == batch_labels)
        
        return acc / (len(self.test_images) - len(self.test_images) % batch_size)
