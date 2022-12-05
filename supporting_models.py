import tensorflow as tf
#import tensorflow_probability as tfp
from tensorflow import keras
from keras import layers


def Simple_CNN(num_classes,in_shape):
    model = tf.keras.Sequential([
        layers.Conv2D(32,(3,3), activation='relu',input_shape=in_shape),
        layers.MaxPool2D((2,2)),
        layers.Flatten(),
        layers.Dense(100,activation='relu',name="penultimate_layer"),
        layers.Dense(num_classes,name="last_layer"),
        layers.Softmax()
    ])
    return model

def AlexNet (num_classes,in_shape):
    model = tf.keras.Sequential([
        layers.Conv2D(96, 11, strides=4, activation='relu',input_shape=in_shape),
        layers.BatchNormalization(),

        layers.MaxPool2D(2, strides=2),
        
        layers.Conv2D(256,11,strides=1,activation='relu',padding='same'),
        layers.BatchNormalization(),

        layers.Conv2D(384, (3,3),strides=(1,1), activation='relu',padding="same"),
        layers.BatchNormalization(),
    
        layers.Conv2D(384, (3,3),strides=(1,1), activation='relu',padding="same"),
        layers.BatchNormalization(),

        layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu',padding="same"),
        layers.BatchNormalization(),

        layers.MaxPooling2D(2, strides=(2, 2)),

        layers.Flatten(),

        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),

        layers.Dense(4096, activation='relu',name="penultimate_layer"),
        layers.Dropout(0.5),
        layers.Dense(num_classes,name="last_layer"),
        layers.Softmax()
    ])
    return model

def EfficientNetV2B0_pretrained(num_classes,in_shape):
    model = tf.keras.Sequential([
        tf.keras.applications.EfficientNetV2B0(include_top=False,weights='imagenet',input_shape=in_shape,classifier_activation='None'),
        layers.Flatten(),
        layers.Dense(100,activation='relu',name="penultimate_layer"),
        layers.Dense(num_classes,name="last_layer"),
        layers.Softmax()
    ])
    return model

def ResNet50(num_classes,in_shape):
    model = tf.keras.applications.resnet50.ResNet50(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=in_shape,
        pooling=None,
        classes=num_classes)
    return model

def ResNet101(num_classes,in_shape):
    model =  tf.keras.applications.resnet.ResNet101(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=in_shape,
        pooling=None,
        classes=num_classes)
    return model


def select_model(model_name,num_classes,img_shape):
    if model_name == 'Simple_CNN':
        return Simple_CNN(num_classes,img_shape)
    if model_name == 'AlexNet':
        return AlexNet(num_classes,img_shape)
    if model_name == 'EfficientNetV2B0_pretrained':
        return EfficientNetV2B0_pretrained(num_classes,img_shape)
    if model_name == 'ResNet50':
        return ResNet50(num_classes,img_shape)
    if model_name == 'ResNet101':
        return ResNet101(num_classes,img_shape)