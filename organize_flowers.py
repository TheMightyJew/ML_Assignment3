import os
import scipy.io
from shutil import copyfile

image_labels_file = scipy.io.loadmat('data/imagelabels.mat')
labels = list(image_labels_file['labels'][0])
dict = {}
for label in set(labels):
    dict[label] = set()
for index, filename in enumerate(os.listdir('data/flowers_photos')):
    dict[labels[index]].add(filename)
try:
    os.mkdir('data/organized_flowers_photos')
except:
    print("error")
for label in dict.keys():
    try:
        os.mkdir('data/organized_flowers_photos/' + str(label))
    except:
        print("error")
    for filename in dict[label]:
        copyfile('data/flowers_photos/' + filename, 'data/organized_flowers_photos/' + str(label) + '/' + filename)
