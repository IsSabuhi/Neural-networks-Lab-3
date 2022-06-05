import numpy as np
from random import shuffle
from PIL.Image import open
from os import listdir
from typing import List

class KohonenNeural:
    learning_rate: float
    D: float = 1
    

    def __init__(self, input, clasters): 
        self.weights = np.random.uniform(low=-0.3, high=0.3, size=(clasters, input))

    def predict(self, vector: np.ndarray):
        dist: np.ndarray =  np.power((vector - self.weights), 2).sum(axis=1)
        winner_index = dist.argmin()
        return winner_index

    def train(self, vector: np.ndarray):
        winner_index = self.predict(vector)

        all_dists: np.ndarray = np.zeros(5, dtype=np.float32)
        rows, _ = self.weights.shape
        for index in range(0, rows):
            if index == winner_index:
                continue
            else:
                all_dists[index] = (np.power((vector - self.weights[index]), 2).sum())

        if self.D is None:
            max_dist_index = all_dists.argmax()
            self.D = all_dists[max_dist_index]
        
        all_errors = []
        for index in range(0, len(all_dists)):
            if index == winner_index or all_dists[index] < kn.D:
                delta: np.ndarray = self.learning_rate * (vector - self.weights[index])
                self.weights[index] += delta
                all_errors.append(np.abs(delta))
            
        all_errors = np.array(all_errors)
        return all_errors.sum()

#Загрузка датасета

def normalize(image: np.ndarray):
    new_image = []
    for rgb in image:
        rgb: np.ndarray
        if (rgb == [255,255,255]).all():
            new_image.append(0)
        else:
            new_image.append(1)

    return np.array(new_image)

dataset: List[tuple] = []
for file in listdir('data'):
    image = np.array(open(f'data/{file}'))
    x_max, y_max, _ = image.shape
    image = image.reshape((x_max*y_max, 3))
    image = normalize(image)
    dataset.append(tuple((file, image)))

test: List[tuple] = []
for file in listdir('test'):
    image = np.array(open(f'test/{file}'))
    x_max, y_max, _ = image.shape
    image = image.reshape((x_max*y_max, 3))
    image = normalize(image)
    test.append(tuple((file, image)))

#Обучение

kn = KohonenNeural(2500, 5)
epoch = 200
kn.learning_rate = 0.8

all_deltas = []
epoch_count = 0
error_counter = np.zeros(shape=5)

for i in range(epoch):
    shuffle(dataset)
    delta: float = 0
    for _, image in dataset:
        delta += kn.train(image)
    
    delta = delta / len(dataset)
    all_deltas.append(round(delta, 5))
    if (delta < 0.05): break

    epoch_count += 1
    kn.learning_rate *= 0.9
    kn.D *= 0.9


#Кластеры

print('Обучающая выборка:')
all_class = { 0: {}, 1: {}, 2: {}, 3: {}, 4: {} }
for filename, image in dataset:
    classes = kn.predict(image)
    default_value = all_class[classes].get(filename.split(' ')[0], 0)
    new_value = default_value + 1
    all_class[classes][filename.split(' ')[0]] = new_value
    print(f'{filename}: Класс {classes}')

correct_classes = {}
for _class in range(0, len(all_class)):
    max_key = max(all_class[_class], key=all_class[_class].get)
    correct_classes[_class] = max_key
print(correct_classes)

print(f'\nИзменения на эпохе {all_deltas}')
print(f'Прошло эпох: {epoch_count}')

print('Тестовая выборка:')
error = 0
for filename, image in test:
    classes = kn.predict(image)
    if filename.split(' ')[0] != correct_classes[classes]: error += 1  
    print(f'{filename}: Класс {classes}')
    
        

print(f'Ошибка на тестовой выборке: {error / len(test)}')