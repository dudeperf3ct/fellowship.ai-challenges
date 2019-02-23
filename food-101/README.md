Food-101
==============================

- [Problem Statement](#problem-statement)
  - [Previous SoTA Results](#previous-sota-results)
  - [References](#references)
- [Introduction](#introduction)
- [Approach](#approach)
- [Conclusion and Result](#conclusion-and-result)
- [Improvements](#improvements)
- [Project Organization](#project-organization)


## Problem Statement

[Food-101](https://www.vision.ee.ethz.ch/datasets_extra/food-101/) is a challenging vision problem, but everyone can relate to it. Recent SoTA is ~80% top-1, 90% top-5.  These approaches rely on lots of TTA, large networks and  even novel architectures.

Train a decent model >85% accuracy for top-1 for the test set, using a ResNet50 or smaller network with a reasonable set of augmentations. 

![sample image from dataset](notebooks/images/food-101.jpg "Food-101 sample images")


---

### Previous SoTA Results

Following is a comparison of the previous SoTA classification results for the Food-101 dataset.

| Model                    |  Augmentations           |  Epochs  |  Top-1 Accuracy  % |  Top-5 Accuracy %  |
| ------------------------|----------------------------------| --------------|------------------------------|------------------------------ |
| InceptionV3      | Flip, Rotation, Color, Zoom | 32   |                 88.28           |            96.88                 |
|WISeR                    | Flip, Rotation, Color, Zoom |  ~ 32   |               90.27    |           98.71                   |
| ResNet+fastai   | Optional Transformations |  16   |                 90.52           |            98.34                 |


---

### References

[1] **Inception V3 Approach** Hassannejad, Hamid, et al. [Food image recognition using very deep convolutional networks](https://dl.acm.org/citation.cfm?id=2986042). Proceedings of the 2nd International Workshop on Multimedia Assisted Dietary Management . ACM, 2016.

[2 ] **WISeR Approach** Martinel, Niki, Gian Luca Foresti, and Christian Micheloni. [Wide-slice residual networks for food recognition](https://arxiv.org/pdf/1612.06543.pdf) . Applications of Computer Vision (WACV), 2018 IEEE Winter Conference on . IEEE, 2018.

[3] **ResNet + fastai Approach** [platform.ai](https://platform.ai/blog/page/3/new-food-101-sota-with-fastai-and-platform-ais-fast-augmentation-search/) 



---

We will tackle this problem using two frameworks, 

1. [Keras](http://keras.io/) 

2. [Fastai](https://docs.fast.ai/).

 ---
 
 ## Introduction
 
 Our objective is to classify 101,000 food images in 101 categories.
 
This is very so ImageNet like where we had 1.2  million images to classify into 1000 categories. There we saw explosion of different architectures starting from AlexNet, ZFNet, VGG, GoogLeNet, Inception v3, ResNet, SqueezeNet, and many other Nets to tackle this problem *better than humans*.
 

Now it's time to *stand on shoulder of these Nets* and use it to solve our classification problem into 101 food categories.

We have already seen a great length why **CNNs** are great at these jobs and we looked at many other things in [this blog](https://dudeperf3ct.github.io/cnn/mnist/2018/10/17/Force-of-Convolutional-Neural-Networks/).

---

## Approach

We will use ResNet50 as base architecture.


![resnet50](notebooks/images/resnet-50.png "ResNet50")

**Steps**

- We will add transformations like brightness, contrast, zoom, etc and resize all the images to size 224 for passing to the base architecture of ResNet50.
- Find a proper learning rate using LR Finder, an approach proposed by Leslie Smith in awesome paper [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186). To further peeking in how it works, look [here](https://sgugger.github.io/the-1cycle-policy.html), [here](http://teleported.in/posts/cyclic-learning-rate/) and [here](https://www.jeremyjordan.me/nn-learning-rate/) and [Super-Convergence paper](https://arxiv.org/abs/1708.07120) .
- Train by keeping the weights of ResNet50 architecture (*excluding last FC layer*) fixed for 5 epochs.
- Unfreeze (*we can change the frozen weights of resnet50*), and train again for 4 epochs. This involves approach called discrimative fine-tuning(differential learning) which that the initial layers in CNN architectures better identify basic patterns like edge, textures and we don't wan't to drastically change that learning and hence very low learning rate, on other hand the final layers can be changed using higher learning rate. To see further, look [here](https://towardsdatascience.com/transfer-learning-using-differential-learning-rates-638455797f00), [here](https://towardsdatascience.com/transfer-learning-using-differential-learning-rates-638455797f00).
- Then, we will change the size of images from 224 to 512. (*Wooh bigger images*) and use the same model above train further.
- Again we approach the same methods above, freeze for 4 epochs and unfreeze and train for 3 epochs.


*Simple Enough?*

What result do we obtain after going through all this? Let's have a look

All results obtained are using Google Colab(*thanks Google!*).


|  Phase                       |   Time Taken (hrs)          |  Epochs  |  Top-1 Accuracy  % |  Top-5 Accuracy %  |
| ------------------------     |----------------------------------| --------------|------------------------------|------------------------------ |
| Freeze and Train on 224 size images  |  4 | 5   |                 75           |            92.36                |
|  Unfreeze and Train on 224 size images | 4  |  4   |               85    |           95.11                   |
|  Freeze and Train on 512 size images  | 7 |  4  |                 70          |            90.24                 |
|  Unfreeze and Train on 512 size images  | 7 |  3 |                 83           |            95.98                 |

---

**Conclusion**

*Phew!* 

**A lot of patience** - 22 hrs (*and dealing with colab is not easy, poor connections will lead to always disconnecting and 12 hrs timeout*)

**Very Cool Classifier** - Top-1 Accuracy = 83% and Top-5 Accuracy = 96%

---

**Improvements**

Results can further be improved

- Train longer (*as always, I need more power*)
- Experimenting with more transformations like skewness, jitter, etc can lead to more roubstness

---

*Don't forget to check out the blog in case you need a refresher on CNN, Transfer Learning and Visualizing CNN.*

Blog Links: 

[1] [CNNs](https://dudeperf3ct.github.io/cnn/mnist/2018/10/17/Force-of-Convolutional-Neural-Networks/).

[2] [Transfer Learning](https://dudeperf3ct.github.io/transfer/learning/catsvsdogs/2018/11/20/Power-of-Transfer-Learning/).

[3] [Visualizing CNN](https://dudeperf3ct.github.io/visualize/cnn/catsvsdogs/2018/12/02/Power-of-Visualizing-Convolution-Neural-Networks/).

---

## Project Organization
    
    ├── docs
    │   ├── commands.rst
    │   ├── conf.py
    │   ├── getting-started.rst
    │   ├── index.rst
    │   ├── make.bat
    │   └── Makefile
    ├── LICENSE
    ├── Makefile
    ├── notebooks
    │   ├── fastai_food_101.ipynb             <- Implementation and results using fastai library
    │   └── images
    │       ├── food-101.jpg
    │       └── resnet-50.png
    ├── README.md
    ├── references
    ├── reports
    │   └── figures
    ├── requirements.txt                      <- The requirements file for reproducing the analysis environment
    ├── setup.py
    ├── src
    │   ├── models                            <- Keras Scripts to train, finetune models and make predictions
    │   │   ├── eda.py
    │   │   ├── find_lr_model.py
    │   │   ├── find_lr.py
    │   │   ├── finetune_model.py
    │   │   ├── __init__.py
    │   │   ├── model.py
    │   │   ├── one_cycle_lr.py
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   └── visualization                    <- Keras Scripts to create exploratory visualizations
    │       ├── __init__.py 
    │       └── visualize.py
    ├── test_environment.py
    └── tox.ini



--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
