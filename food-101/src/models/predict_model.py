'''
This file is used for testing images on finetuned model
'''
import numpy as np
from .model import build_model
from PIL import Image
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import json

image_path = 'data/test/test-1.jpg'
name = 'xception'

print ('Loading model...')
#create the same model used for training
model = build_model(name) 
#load the finetuned weights of the model
model.load_weights(name+'_finetune_model-224_weights.h5')
print ('Model Loaded...')

print ('Getting class to index mapping...')
#get class to index mapping
with open('class_mapping.json') as infile:
    label_map = json.load(infile)

def test_image(model, image_path):
	#preprocess test image
	img = load_img(image_path, target_size=(224, 224))
	x = img_to_array(img)
	x /= 255.
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(img)
	#predict probabilities 
	predictions = model.predict_proba(x)
	print ('Probabilities:', predictions)
	#find the index to highest probability
	pred_class = predictions.argmax(axis=-1)
	print ('Predicted Class:', pred_class)
	#print class to index mapping
	print ('Class to Index Mapping:', label_map)
	img.show()

print ('Predicting...')
test_image(model, image_path)