'''
This file is used to create a base model for training and
also a fine tuning model 
'''

# import necessary libraries
from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras.layers import Dense, Flatten, Dropout, BatchNormalization
from keras.models import Model, load_model, Sequential

# we will create a base architecture model
def build_model(name):
    
    # vgg19
    if name == 'vgg19':
        base_model = VGG19(weights='imagenet', include_top=False, pooling='avg')
        for layer in base_model.layers:
            layer.trainable = False
        x = base_model.output
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)

    # inception v3
    elif name == 'Inceptionv3':
        base_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
        for layer in base_model.layers:
            layer.trainable = False
        x = base_model.output
        x = Dense(1024, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.75)(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)     
        
    # densenet 201
    elif name == 'densenet201':
        base_model = DenseNet201(weights='imagenet', include_top=False, pooling='avg')
        for layer in base_model.layers:
            layer.trainable = False
        x = base_model.output
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(1024, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.75)(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        
    # xception
    elif name == 'xception':
        base_model = Xception(weights='imagenet', include_top=False, pooling='avg')
        for layer in base_model.layers:
            layer.trainable = False
        x = base_model.output
        
        x = Dense(1024, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.75)(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)        
        
    # resnet50
    elif name == 'resnet50':
        base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        for layer in base_model.layers:
            layer.trainable = False
        x = base_model.output
        
        x = Dense(1024, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.75)(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        
#     elif name == 'custom':
      # don't be a hero by inventing a new architecture
      # use pretrained architecture if the dataset is similar 
        
    return model


# we will finetune model based on predefined number of layer to freeze and unfreeze
def finetune_model(model, name):
    
    # vgg19
    if name == 'vgg19':
        for layer in model.layers[:12]:
            layer.trainable = False
        for layer in model.layers[12:]:
            layer.trainable = True

    # inceptionv3            
    elif name == 'Inceptionv3':
        for layer in model.layers[:299]:
            layer.trainable = False
        for layer in model.layers[299:]:
            layer.trainable = True   
    
    # densenet201    
    elif name == 'densenet201':
        for layer in model.layers[:700]:
            layer.trainable = False
        for layer in model.layers[700:]:
            layer.trainable = True
    
    # xception  
    elif name == 'xception':
        for layer in model.layers[:122]:
            layer.trainable = False
        for layer in model.layers[122:]:
            layer.trainable = True       
    
    # resnet50    
    elif name == 'resnet50':
        for layer in model.layers[:165]:
            layer.trainable = False
        for layer in model.layers[165:]:
            layer.trainable = True
        
#     elif name == 'custom':
      # don't be a hero by inventing a new architecture
      # use pretrained architecture if the dataset is similar 
        
    return model    