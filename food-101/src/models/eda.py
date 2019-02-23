'''
This file will be used for Exploratory Data Analysis
'''

# import necessary libraries
import numpy as np
import os
import random
from .visualize import show_img
from pathlib import Path

# path to dataset
path = Path('data/food-101/images')

# plot class distribution for all labels 
# number of files in directory which is also number of classes
files = os.listdir(path)
num_classes = len(files)
print ('Number of classes:', num_classes)

#distribution dict from labels to number of images
dist = dict()
# class distribution
for f in files:
    dist[f] = len(os.listdir(path/f))

# all 101 food-categories have 1000 images each     
print ('Class distribution:')
print (dist)

# plot distribution of random 10 classes
# why 10? visualizing 101 at once is difficult 
keys = random.sample(list(dist.keys()), 10)
values = [dist[k] for k in keys]
plt.bar(keys, values)
plt.xticks(rotation=270)
plt.show()

# plot 10 random images from dataset
show_img(rows=2, cols=5, figsize=(15, 5), path)
