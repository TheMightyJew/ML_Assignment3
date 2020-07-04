import os
import scipy.io
from shutil import copyfile

test_size = 0.2

image_labels_file = scipy.io.loadmat('data/imagelabels.mat')
labels = list(image_labels_file['labels'][0])
dict = {}
for label in set(labels):
    dict[label] = set()
for index, filename in enumerate(os.listdir('data/flowers_photos')):
    dict[labels[index]].add(filename)

base_folder_name = 'organized_flowers_photos_' + str(test_size)
try:
    os.mkdir('data/' + base_folder_name)
    os.mkdir('data/' + base_folder_name + '/train')
    os.mkdir('data/' + base_folder_name + '/test')
except:
    print("error")
for label in dict.keys():
    try:
        os.mkdir('data/' + base_folder_name + '/train/' + str(label))
        os.mkdir('data/' + base_folder_name + '/test/' + str(label))
    except:
        print("error")
    length = len(dict[label]) - 1
    for i, filename in enumerate(dict[label]):
        if i <= test_size * length:
            copyfile('data/flowers_photos/' + filename, 'data/' + base_folder_name + '/test/' + str(label) + '/' + filename)
        else:
            copyfile('data/flowers_photos/' + filename, 'data/' + base_folder_name + '/train/' + str(label) + '/' + filename)
