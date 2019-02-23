# Select Your Challenge Problem

## One-shot Learning

[Omniglot](https://github.com/brendenlake/omniglot), the “transpose” of MNIST, with 1623 character classes, each with 20 examples. 

Use background set of [30 alphabets](https://github.com/brendenlake/omniglot/blob/master/python/images_background.zip) for training and evaluate on set of [20 alphabets](https://github.com/brendenlake/omniglot/blob/master/python/images_evaluation.zip). Refer to this [script](https://github.com/brendenlake/omniglot/blob/master/python/one-shot-classification/demo_classification.py) for sampling setup.

Report one-shot classification (20-way) results using a meta learning approach like [MAML](https://arxiv.org/pdf/1703.03400.pdf).

## Image Segmentation

Apply an automatic portrait segmentation model (aka image matting) to [celebrity face](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset. 


## Food-101

[Food-101](https://www.vision.ee.ethz.ch/datasets_extra/food-101/) is a challenging vision problem, but everyone can relate to it.  Recent SoTA is ~80% top-1, 90% top-5.  These approaches rely on lots of TTA, large networks and  even novel architectures.

Train a decent model >85% accuracy for top-1 for the test set, using a ResNet50 or smaller network with a reasonable set of augmentations.   


## ULMFiT Sentiment 

Apply a supervised or semi-supervised [ULMFiT](http://nlp.fast.ai/classification/2018/05/15/introducting-ulmfit.html) model to [Twitter US Airlines Sentiment](https://www.kaggle.com/crowdflower/twitter-airline-sentiment#Tweets.csv).
