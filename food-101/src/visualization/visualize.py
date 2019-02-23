'''
This file contains function to plot sample images in dataset
'''

# import necessary libraries
from PIL import Image
from pathlib import Path
import os
import random
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt

# create a visualization function to show sample dataset
# number of rows, cols and path to dataset required to plot sample dataset
def show_img(rows, cols, figsize=None, path):
      
    fig, ax = plt.subplots(rows, cols, figsize=figsize)
    files = os.listdir(path)
    for i in range(rows):
        for j in range(cols):
            random_folder = random.choice(files)
            random_file = random.choice(os.listdir(path/random_folder))
            im = Image.open(path/random_folder/random_file)
            ax[i, j].imshow(im)
            ax[i, j].set_xlabel(str(random_folder) + " " + str(im.size))
            ax[i, j].set_yticklabels([])
            ax[i, j].set_xticklabels([])
            ax[i, j].grid(False)
