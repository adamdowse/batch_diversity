import tensorflow as tf
#import tensorflow_probability as tfp
from tensorflow import keras
from keras import layers
import math


def Simple_CNN(num_classes,in_shape,REG):
    model = tf.keras.Sequential([
        layers.Conv2D(32,(3,3), activation='relu',input_shape=in_shape,kernel_regularizer=tf.keras.regularizers.l2(REG)),
        layers.MaxPool2D((2,2)),
        layers.Flatten(),
        layers.Dense(100,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(REG),name="penultimate_layer"),
        layers.Dense(num_classes,name="last_layer"),
        layers.Softmax()
    ])
    return model


def All_CNN_noBN(num_classes,in_shape):
    #cnn arch used in crittical learnning point paper
    model = tf.keras.Sequential([
        layers.Conv2D(96,(3,3), activation='relu',input_shape=in_shape),
        layers.Conv2D(96,(3,3), activation='relu'),
        layers.MaxPool2D((3,3),strides=2),
        layers.Conv2D(192,(3,3), activation='relu'),
        layers.Conv2D(192,(3,3), activation='relu'),
        layers.MaxPool2D((3,3),strides=2),
        layers.Conv2D(192,(1,1), activation='relu'),
        layers.Conv2D(10,(1,1), activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(num_classes,name="last_layer"),
        layers.Softmax()
    ])
    return model

def FullyConnected(num_classes,in_shape):
    model = tf.keras.Sequential([
        layers.Flatten(input_shape=in_shape),
        layers.Dense(2000,activation='relu'),
        layers.Dense(1500,activation='relu'),
        layers.Dense(1000,activation='relu'),
        layers.Dense(500,activation='relu'),
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

def ResNet18(num_classes,in_shape,REG):
    inputs = keras.Input(shape=in_shape)
    outputs = build_resnet(inputs, [2, 2, 2, 2], num_classes,REG)
    model = keras.Model(inputs, outputs)
    return model
    




def select_model(model_name,num_classes,img_shape,REG=0):
    if model_name == 'Simple_CNN':
        return Simple_CNN(num_classes,img_shape,REG)
    if model_name == 'AlexNet':
        return AlexNet(num_classes,img_shape)
    if model_name == 'EfficientNetV2B0_pretrained':
        return EfficientNetV2B0_pretrained(num_classes,img_shape)
    if model_name == 'ResNet50':
        return ResNet50(num_classes,img_shape)
    if model_name == 'ResNet101':
        return ResNet101(num_classes,img_shape)
    if model_name == 'ResNet18':
        return ResNet18(num_classes,img_shape,REG)
    if model_name == 'All_CNN_noBN':
        return All_CNN_noBN(num_classes,img_shape)
    if model_name == 'FullyConnected':
        return FullyConnected(num_classes,img_shape)



def build_resnet(x,vars,num_classes,REG=0):
    kaiming_normal = keras.initializers.VarianceScaling(scale=2.0, mode='fan_out', distribution='untruncated_normal')

    def conv3x3(x, out_planes, stride=1, name=None,REG=0):
        x = layers.ZeroPadding2D(padding=1, name=f'{name}_pad')(x)
        return layers.Conv2D(filters=out_planes, kernel_size=3, strides=stride, use_bias=False, kernel_initializer=kaiming_normal,kernel_regularizer=keras.regularizers.l2(REG), name=name)(x)

    def basic_block(x, planes, stride=1, downsample=None, name=None):
        identity = x

        out = conv3x3(x, planes, stride=stride, name=f'{name}.conv1')
        out = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.bn1')(out)
        out = layers.ReLU(name=f'{name}.relu1')(out)

        out = conv3x3(out, planes, name=f'{name}.conv2')
        out = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.bn2')(out)

        if downsample is not None:
            for layer in downsample:
                identity = layer(identity)

        out = layers.Add(name=f'{name}.add')([identity, out])
        out = layers.ReLU(name=f'{name}.relu2')(out)

        return out

    def make_layer(x, planes, blocks, stride=1, name=None,REG=0):
        downsample = None
        inplanes = x.shape[3]
        if stride != 1 or inplanes != planes:
            downsample = [
                layers.Conv2D(filters=planes, kernel_size=1, strides=stride, use_bias=False, kernel_initializer=kaiming_normal,kernel_regularizer=keras.regularizers.l2(REG), name=f'{name}.0.downsample.0'),
                layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.0.downsample.1'),
            ]

        x = basic_block(x, planes, stride, downsample, name=f'{name}.0')
        for i in range(1, blocks):
            x = basic_block(x, planes, name=f'{name}.{i}')

        return x

    def resnet(x, blocks_per_layer, num_classes,REG):
        x = layers.ZeroPadding2D(padding=3, name='conv1_pad')(x)
        x = layers.Conv2D(filters=64, kernel_size=7, strides=2, use_bias=False, kernel_initializer=kaiming_normal,kernel_regularizer=keras.regularizers.l2(REG), name='conv1')(x)
        x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='bn1')(x)
        x = layers.ReLU(name='relu1')(x)
        x = layers.ZeroPadding2D(padding=1, name='maxpool_pad')(x)
        x = layers.MaxPool2D(pool_size=3, strides=2, name='maxpool')(x)

        x = make_layer(x, 64, blocks_per_layer[0], name='layer1')
        x = make_layer(x, 128, blocks_per_layer[1], stride=2, name='layer2')
        x = make_layer(x, 256, blocks_per_layer[2], stride=2, name='layer3')
        x = make_layer(x, 512, blocks_per_layer[3], stride=2, name='layer4')

        x = layers.GlobalAveragePooling2D(name='avgpool')(x)
        initializer = keras.initializers.RandomUniform(-1.0 / math.sqrt(512), 1.0 / math.sqrt(512))
        x = layers.Dense(units=num_classes, kernel_initializer=initializer, bias_initializer=initializer, name='fc')(x)
        x = layers.Softmax(name='softmax')(x)
        return x
    return resnet(x, vars,num_classes)
