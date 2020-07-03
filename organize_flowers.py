import os
import scipy.io
from shutil import copyfile

test_size = 0.3

image_labels_file = scipy.io.loadmat('data/imagelabels.mat')
labels = list(image_labels_file['labels'][0])
dict = {}
for label in set(labels):
    dict[label] = set()
for index, filename in enumerate(os.listdir('data/flowers_photos')):
    dict[labels[index]].add(filename)
try:
    os.mkdir('data/organized_flowers_photos')
    os.mkdir('data/organized_flowers_photos/train')
    os.mkdir('data/organized_flowers_photos/test')
except:
    print("error")
for label in dict.keys():
    try:
        os.mkdir('data/organized_flowers_photos/train/' + str(label))
        os.mkdir('data/organized_flowers_photos/test/' + str(label))
    except:
        print("error")
    length = len(dict[label]) - 1
    for i, filename in enumerate(dict[label]):
        if i <= test_size * length:
            copyfile('data/flowers_photos/' + filename, 'data/organized_flowers_photos/test/' + str(label) + '/' + filename)
        else:
            copyfile('data/flowers_photos/' + filename, 'data/organized_flowers_photos/train/' + str(label) + '/' + filename)
